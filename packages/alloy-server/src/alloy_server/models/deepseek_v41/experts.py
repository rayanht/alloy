"""SSD-streamed routed experts and the mmap'd Engram tables.

The routed experts (306 GiB at Q4_K_M) cannot be resident, so they live in the
GGUF shards on disk and are read on demand into a resident **arena**: a Metal-shared
buffer of expert-sized slots the gathered MoE kernels index by slot instead of by
expert id. An LRU over (layer, expert) -> slot keeps hot experts resident between
tokens; misses are `preadv` reads straight into the slot's memory (the page cache is
bypassed — the arena is the cache). Layers whose down projection is Q6_K and layers
whose down is Q4_K have different slot byte layouts, so each down type gets its own
arena.

The Engram tables (103 GiB) are addressed by deterministic hash: 24 rows per token
per layer, gathered on the CPU through a memmap of the shard.
"""

from __future__ import annotations

import fcntl
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np

from alloy._compiler.dtypes import uint8
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy_server.gguf.split import SplitGGUF, TensorLocation
from alloy_server.models.deepseek_v41.config import DeepseekV41Config

Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K

# macOS fcntl: bypass the unified buffer cache for this descriptor.
F_NOCACHE = 48


def row_bytes(qtype: gguf.GGMLQuantizationType, n_in: int) -> int:
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    return n_in // block_size * type_size


@dataclass(frozen=True)
class ExpertLayout:
    """Byte layout of one expert of a layer: gate/up rows (Q4_K) then down rows."""

    inter: int
    hidden: int
    gu_row_bytes: int
    down_qtype: gguf.GGMLQuantizationType
    down_row_bytes: int

    @property
    def gate_bytes(self) -> int:
        return self.inter * self.gu_row_bytes

    @property
    def down_bytes(self) -> int:
        return self.hidden * self.down_row_bytes

    @property
    def slot_bytes(self) -> int:
        return 2 * self.gate_bytes + self.down_bytes


class ExpertArena:
    """Resident slots for one expert layout: `gate_up` (S, 2I, gu_row_bytes) and
    `down` (S, H, down_row_bytes), the shapes the gathered MoE kernels index."""

    def __init__(self, layout: ExpertLayout, slots: int) -> None:
        self.layout = layout
        self.slots = slots
        self.gate_up = _alloc_aligned((slots, 2 * layout.inter, layout.gu_row_bytes), uint8)
        self.down = _alloc_aligned((slots, layout.hidden, layout.down_row_bytes), uint8)
        self.gate_up_np = self.gate_up.numpy
        self.down_np = self.down.numpy
        # LRU of resident experts: (layer, expert) -> slot, oldest first
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.free: list[int] = list(range(slots))
        self.pinned: set[int] = set()

    def slot_for(self, key: tuple[int, int]) -> int | None:
        slot = self.lru.get(key)
        if slot is not None:
            self.lru.move_to_end(key)
        return slot

    def take_slot(self, key: tuple[int, int]) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            for old_key, old_slot in self.lru.items():
                if old_slot not in self.pinned:
                    del self.lru[old_key]
                    slot = old_slot
                    break
            else:
                raise RuntimeError("expert arena: every slot is pinned")
        self.lru[key] = slot
        return slot


class ExpertStore:
    def __init__(
        self,
        split: SplitGGUF,
        cfg: DeepseekV41Config,
        *,
        budget_bytes: int,
        io_threads: int = 8,
        bypass_page_cache: bool = True,
    ) -> None:
        self.cfg = cfg
        self.locs: dict[tuple[int, str], TensorLocation] = {}
        self.layouts: dict[int, ExpertLayout] = {}
        for i in range(cfg.n_layers):
            gate = split.tensors[f"blk.{i}.ffn_gate_exps.weight"]
            up = split.tensors[f"blk.{i}.ffn_up_exps.weight"]
            down = split.tensors[f"blk.{i}.ffn_down_exps.weight"]
            if gate.tensor_type != Q4_K or up.tensor_type != Q4_K:
                raise ValueError(f"layer {i}: gate/up experts must be Q4_K")
            if down.tensor_type not in (Q4_K, Q6_K):
                raise ValueError(f"layer {i}: down experts must be Q4_K or Q6_K")
            self.locs[(i, "gate")] = gate
            self.locs[(i, "up")] = up
            self.locs[(i, "down")] = down
            self.layouts[i] = ExpertLayout(
                inter=cfg.moe_inter_dim,
                hidden=cfg.dim,
                gu_row_bytes=row_bytes(Q4_K, cfg.dim),
                down_qtype=down.tensor_type,
                down_row_bytes=row_bytes(down.tensor_type, cfg.moe_inter_dim),
            )
        self.fds: dict[Path, int] = {}
        for loc in self.locs.values():
            if loc.path not in self.fds:
                fd = os.open(loc.path, os.O_RDONLY)
                if bypass_page_cache:
                    fcntl.fcntl(fd, F_NOCACHE, 1)
                self.fds[loc.path] = fd
        # one arena per distinct slot layout, the budget split by each layout's
        # share of the model's experts
        by_layout: dict[ExpertLayout, int] = {}
        for layout in self.layouts.values():
            by_layout[layout] = by_layout.get(layout, 0) + cfg.n_routed_experts
        total = sum(by_layout.values())
        self.arenas: dict[ExpertLayout, ExpertArena] = {}
        for layout, n_experts in by_layout.items():
            # a whole layer's experts must fit at once (a chunk can route to every expert)
            slots = max(cfg.n_routed_experts, int(budget_bytes * n_experts / total) // layout.slot_bytes)
            slots = min(slots, n_experts)
            self.arenas[layout] = ExpertArena(layout, slots)
        self.pool = ThreadPoolExecutor(max_workers=io_threads)
        self.misses = 0
        self.hits = 0
        self.bytes_read = 0

    def arena(self, layer: int) -> ExpertArena:
        return self.arenas[self.layouts[layer]]

    def read_into(self, view: np.ndarray, loc: TensorLocation, byte_offset: int) -> None:
        fd = self.fds[loc.path]
        mv = memoryview(view).cast("B")
        off = loc.data_offset + byte_offset
        done = 0
        while done < len(mv):
            n = os.preadv(fd, [mv[done:]], off + done)
            if n <= 0:
                raise OSError(f"short read at {loc.name}+{off + done}")
            done += n
        self.bytes_read += len(mv)

    def fetch(self, layer: int, expert: int, slot: int) -> None:
        arena = self.arena(layer)
        layout = arena.layout
        gu = arena.gate_up_np[slot]
        self.read_into(gu[: layout.inter], self.locs[(layer, "gate")], expert * layout.gate_bytes)
        self.read_into(gu[layout.inter :], self.locs[(layer, "up")], expert * layout.gate_bytes)
        self.read_into(arena.down_np[slot], self.locs[(layer, "down")], expert * layout.down_bytes)

    def acquire(self, layer: int, experts: list[int]) -> dict[int, int]:
        """Make `experts` of `layer` resident; returns expert -> arena slot. The
        slots stay pinned until `release(layer)`."""
        arena = self.arena(layer)
        slots: dict[int, int] = {}
        futures = []
        for e in dict.fromkeys(experts):
            key = (layer, e)
            slot = arena.slot_for(key)
            if slot is None:
                slot = arena.take_slot(key)
                self.misses += 1
                futures.append(self.pool.submit(self.fetch, layer, e, slot))
            else:
                self.hits += 1
            arena.pinned.add(slot)
            slots[e] = slot
        for f in futures:
            f.result()
        return slots

    def release_all(self) -> None:
        """Unpin every slot — valid once every queued kernel that reads them has run."""
        for arena in self.arenas.values():
            arena.pinned.clear()

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


class EngramTables:
    """Per engram layer, the (rows, row_bytes) memmap of the hash table; rows are
    gathered by id on the CPU (24 random rows per token per layer)."""

    def __init__(self, split: SplitGGUF, cfg: DeepseekV41Config) -> None:
        self.tables: dict[int, np.ndarray] = {}
        self.row_bytes = 0
        if cfg.engram is None:
            return
        for i in cfg.engram.layer_ids:
            name = f"blk.{i}.engram_embd.weight"
            if split.tensors[name].tensor_type != Q4_K:
                raise ValueError(f"{name}: expected Q4_K engram table")
            self.tables[i] = split.packed_rows(name)
            self.row_bytes = self.tables[i].shape[1]

    def gather(self, layer: int, ids: np.ndarray) -> np.ndarray:
        """(len(ids), row_bytes) uint8 for the given table rows."""
        return np.ascontiguousarray(self.tables[layer][np.asarray(ids, dtype=np.int64).reshape(-1)])

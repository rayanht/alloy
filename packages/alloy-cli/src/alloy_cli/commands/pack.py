"""`alloy pack` — build a distributable `.alloypack` for on-device Apple inference.

Captures a model's prefill + decode CompiledPlans, ships the weights + plan
structure + a precompiled Metal library (one `default.<plat>.metallib` per
platform, never MSL source — App Store 2.5.2 safety), bundles the tokenizer, and
records a `catalog.json` entry (params, quant, native context, download size, and
`min_ram` = the full resident set the engine allocates on-device).

    alloy pack qwen2.5:3b --target apple --out dist/
    alloy pack qwen2.5:0.5b --out dist/ --no-validate
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
import typer
from huggingface_hub import HfApi

from alloy._runtime import _metal_ext
from alloy_torch import backend
from alloy_torch.backend import InputSlot, IntermediateSlot, OutputSlot, WeightSlot
from alloy_server.generation import decode, prefill
from alloy_server.generation.generator import AlloyGenerator, resolve_eos_tokens
from alloy_server.gguf import ResolvedGGUF
from alloy_server.gguf.tokenizer import load_gguf_tokenizer
from alloy_server.models import load_native_causal_lm, resolve_model

# GGUF general.file_type → quant label (llama.cpp ftype enum).
_FTYPE_QUANT = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
}

# Per-platform metallib: (xcrun sdk, pack filename tag).
_PLATFORMS = {"macos": "macosx", "ios": "iphoneos"}

_KERNEL_RE = re.compile(r"(kernel\s+void\s+)(\w+)(\s*\()")

# `pack.bin` byte layout (little-endian; offsets absolute from file start):
#   magic "ALYB" | u32 version | u32 n_files | u32 n_weights | u32 n_consts | u32 names_len
#   names:        n_files null-terminated UTF-8 paths, padded to 8 bytes (names_len total)
#   files index:  n_files   × (u64 offset, u64 size)   — structure (manifest, metallibs, tokenizer)
#   weights index:n_weights × (u64 offset, u64 size)
#   consts index: n_consts  × (u64 offset, u64 size)
#   payloads (files, then weights, then consts).
_PACK_MAGIC = b"ALYB"
_PACK_VERSION = 2


def _write_pack(path: Path, files: dict[str, bytes],
                weights: list[bytes], consts: list[bytes]) -> None:
    """Serialize structure files + weights + consts into a single `pack.bin`."""
    names = b"".join(name.encode() + b"\0" for name in files)
    pad = (-len(names)) % 8
    names_len = len(names) + pad
    n_files, n_w, n_c = len(files), len(weights), len(consts)
    blobs = [*files.values(), *weights, *consts]
    offset = 24 + names_len + len(blobs) * 16
    index = bytearray()
    for blob in blobs:
        index += struct.pack("<QQ", offset, len(blob))
        offset += len(blob)
    with open(path, "wb") as f:
        f.write(_PACK_MAGIC)
        f.write(struct.pack("<IIIII", _PACK_VERSION, n_files, n_w, n_c, names_len))
        f.write(names + b"\0" * pad)
        f.write(index)
        for blob in blobs:
            f.write(blob)


def _read_pack_index(path: Path) -> tuple[dict[str, tuple[int, int]],
                                          list[tuple[int, int]], list[tuple[int, int]]]:
    """Read `pack.bin`'s header → (files {name: (off,size)}, weights, consts)."""
    with open(path, "rb") as f:
        head = f.read(24)
        assert head[:4] == _PACK_MAGIC, f"bad pack.bin magic in {path}"
        _ver, n_files, n_w, n_c, names_len = struct.unpack("<IIIII", head[4:24])
        names = f.read(names_len).split(b"\0")[:n_files]
        index = f.read((n_files + n_w + n_c) * 16)
    ents = [struct.unpack_from("<QQ", index, i * 16) for i in range(n_files + n_w + n_c)]
    files = {names[i].decode(): ents[i] for i in range(n_files)}
    return files, ents[n_files:n_files + n_w], ents[n_files + n_w:]


def _read_pack_entry(path: Path, entry: tuple[int, int]) -> bytes:
    """Read one (offset, size) entry from `pack.bin`."""
    offset, size = entry
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def _read_addr(ptr: int, n: int) -> bytes:
    return bytes((ctypes.c_char * n).from_address(ptr)) if ptr and n else b""


def _rename_kernels(src: str, shader_idx: int) -> str:
    """Prefix every kernel entry point with `k{idx}_` so all sources combine into
    one metallib without name collisions (the same fn name recurs across sources
    with different constexpr specializations)."""
    return _KERNEL_RE.sub(lambda m: f"{m.group(1)}k{shader_idx}_{m.group(2)}{m.group(3)}", src)


def _renamed_fn(shader_idx: int, fn: str) -> str:
    return f"k{shader_idx}_{fn}"


def _build_metallib(shader_srcs: dict[int, str], out_dir: Path, sdk: str, tag: str) -> Path:
    """Rename kernels uniquely, compile each source to AIR, link into one metallib."""
    airs: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, src in sorted(shader_srcs.items()):
            mp = tdp / f"{i:03d}.metal"
            mp.write_text(_rename_kernels(src, i))
            air = tdp / f"{i:03d}.air"
            subprocess.run(
                ["xcrun", "-sdk", sdk, "metal", "-c", str(mp), "-o", str(air), "-w"],
                check=True, capture_output=True, text=True,
            )
            airs.append(str(air))
        lib = out_dir / f"default.{tag}.metallib"
        subprocess.run(
            ["xcrun", "-sdk", sdk, "metallib", *airs, "-o", str(lib)],
            check=True, capture_output=True, text=True,
        )
    return lib


def _gguf_meta(path: Path) -> dict:
    """Read the architecture metadata needed for the catalog, without a full model load."""
    reader = gguf.GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()

    def field(key: str, default=None):
        f = reader.fields.get(key)
        return f.contents() if f is not None else default

    head_count = int(field(f"{arch}.attention.head_count", 0))
    embed = int(field(f"{arch}.embedding_length", 0))
    head_dim = int(field(f"{arch}.attention.key_length", 0)) or (embed // head_count if head_count else 0)
    ftype = int(field("general.file_type", -1))
    return {
        "arch": arch,
        "name": field("general.name", ""),
        "size_label": field("general.size_label", ""),
        "quant": _FTYPE_QUANT.get(ftype, f"ftype{ftype}"),
        "native_ctx": int(field(f"{arch}.context_length", 0)),
        "n_layers": int(field(f"{arch}.block_count", 0)),
        "n_kv_heads": int(field(f"{arch}.attention.head_count_kv", 0)),
        "head_dim": head_dim,
    }


def _capture_plans(model: str):
    """Run a real prefill→decode generation, capturing both CompiledPlans, their
    pre-call input snapshots, and the golden tokens."""
    prompt = [100 + (i * 137) % 30000 for i in range(200)]  # > chunk: exercises cold+warm
    max_new = 8

    loaded = load_native_causal_lm(model)
    hf = loaded.model.eval()
    vocab = int(hf.config.vocab_size)
    gen = AlloyGenerator.from_model(hf, cache_dtype=torch.float16)
    gen.eager_compile_all()

    cap: dict[str, dict] = {}

    def classify(args):
        m = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.dim() == 2 and a.shape[0] == 1 \
                    and a.dtype in (torch.long, torch.int32, torch.int64) and a.shape[1] <= 4096:
                if m is None or a.shape[1] > m:
                    m = a.shape[1]
        return None if m is None else ("decode" if m == 1 else "prefill")

    def make_wrap(orig):
        def wrap(plan, args, pre_copies=None, wanted_outputs=None, args_stable=False, grid_updates=None):
            kind = classify(args)
            snap, info = {}, []
            for si, slot in enumerate(plan.slots):
                if isinstance(slot, InputSlot):
                    a = args[slot.arg_idx]
                    st = a.untyped_storage()
                    sp = st.data_ptr()
                    if sp not in snap:
                        snap[sp] = _read_addr(sp, st.nbytes())
                    info.append((si, slot.arg_idx, sp, a.data_ptr() - sp, slot.nbytes))
            out = orig(plan, args, pre_copies=pre_copies, wanted_outputs=wanted_outputs,
                       args_stable=args_stable, grid_updates=grid_updates)
            if kind is not None and kind not in cap:
                post = {si: args[s.arg_idx].untyped_storage().data_ptr()
                        for si, s in enumerate(plan.slots) if isinstance(s, InputSlot)}
                cap[kind] = {"plan": plan, "snap": snap, "info": info, "post": post}
            return out
        return wrap

    orig = backend._execute_plan
    backend._execute_plan = decode._execute_plan = prefill._execute_plan = make_wrap(orig)
    try:
        with torch.inference_mode():
            golden = gen.generate(torch.tensor([prompt], dtype=torch.long), max_new_tokens=max_new)
    finally:
        backend._execute_plan = decode._execute_plan = prefill._execute_plan = orig

    golden = [int(t) for t in golden[0].tolist()]
    for k in ("prefill", "decode"):
        if k not in cap:
            raise typer.Exit(f"capture failed: missing {k} plan")

    pf_chunk = next(iter(gen.plans.prefill_inputs))
    pf_ids, pf_pos, pf_lastreal, pf_mask = gen.plans.prefill_inputs[pf_chunk]
    roles = {
        "decode": {"token": gen.plans.find_plan_input_slot(cap["decode"]["plan"], gen.plans.token_input),
                   "cache_position": gen.plans.find_plan_input_slot(cap["decode"]["plan"], gen.plans.cache_position)},
        "prefill": {"token": gen.plans.find_plan_input_slot(cap["prefill"]["plan"], pf_ids),
                    "cache_position": gen.plans.find_plan_input_slot(cap["prefill"]["plan"], pf_pos),
                    "attention_mask": gen.plans.find_plan_input_slot(cap["prefill"]["plan"], pf_mask),
                    "last_real": gen.plans.find_plan_input_slot(cap["prefill"]["plan"], pf_lastreal)},
    }
    return cap, hf, roles, pf_chunk, golden, vocab, prompt, max_new


def pack(
    model: str = typer.Argument(..., help="Model ref (Ollama name, HF repo:quant, or local .gguf)."),
    out: Path = typer.Option(Path("dist"), "--out", "-o", help="Output directory for the pack."),
    target: str = typer.Option("apple", "--target", help="Build target (apple)."),
    platforms: str = typer.Option("macos,ios", "--platforms", help="Comma-separated metallib platforms."),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Replay the pack and check golden logits."),
) -> None:
    """Build a distributable `.alloypack` for on-device Apple inference."""
    if target != "apple":
        raise typer.BadParameter(f"only --target apple is supported (got {target})")
    plats = [p.strip() for p in platforms.split(",") if p.strip()]
    for p in plats:
        if p not in _PLATFORMS:
            raise typer.BadParameter(f"unknown platform {p!r}; known: {list(_PLATFORMS)}")

    resolved = resolve_model(model)
    if not isinstance(resolved, ResolvedGGUF):
        raise typer.BadParameter(f"alloy pack supports GGUF models only; {model} is {resolved.format}")
    meta = _gguf_meta(resolved.path)
    slug = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")
    pack_dir = out / f"{slug}.alloypack"

    typer.echo(f"packing {model}  ({meta['arch']} · {meta['quant']} · ctx {meta['native_ctx']})")
    cap, hf, roles, pf_chunk, golden, vocab, prompt, max_new = _capture_plans(model)

    # --- cross-plan shared buffer identity (state vs immutable const) ---
    pf_post, dc_post = cap["prefill"]["post"], cap["decode"]["post"]
    shared = set(pf_post.values()) & set(dc_post.values())
    written_ptrs: set[int] = set()
    for k in ("prefill", "decode"):
        plan, post = cap[k]["plan"], cap[k]["post"]
        wr = {si for d in plan.dispatches for si in d.write_slot_indices}
        written_ptrs |= {post[si] for si in wr if si in post}
        written_ptrs |= {post[si] for si in plan.mutation_input_slots.values() if si in post}
    state_ptrs = shared & written_ptrs

    shutil.rmtree(pack_dir, ignore_errors=True)
    for sub in ("shaders", "tokenizer"):
        (pack_dir / sub).mkdir(parents=True, exist_ok=True)

    shader_index: dict[str, int] = {}      # msl source -> idx
    shader_srcs: dict[int, str] = {}       # idx -> msl source
    weight_blob: dict[int, int] = {}       # weight base_ptr -> blob idx
    weight_bytes_by_idx: dict[int, bytes] = {}  # blob idx -> weight payload
    const_bytes_by_idx: dict[int, bytes] = {}   # blob idx -> const payload
    buffers: dict[int, dict] = {}          # post_ptr -> {id, kind, nbytes, [blob]}
    const_count = [0]

    def buffer_id(post_ptr, nbytes, snap_bytes, is_state) -> int:
        rec = buffers.get(post_ptr)
        if rec is not None:
            return rec["id"]
        bid = len(buffers)
        rec = {"id": bid, "nbytes": nbytes, "kind": "state" if is_state else "const"}
        if not is_state:
            ci = const_count[0]
            const_count[0] += 1
            const_bytes_by_idx[ci] = snap_bytes or b""
            rec["blob"] = ci
        buffers[post_ptr] = rec
        return bid

    def emit_plan(kind: str) -> dict:
        plan, post, snap = cap[kind]["plan"], cap[kind]["post"], cap[kind]["snap"]
        info = {e[0]: e for e in cap[kind]["info"]}
        pinned_role = {v: r for r, v in roles[kind].items() if v is not None}

        disp_meta = []
        for d in plan.dispatches:
            src = d.msl_source
            if src not in shader_index:
                idx = len(shader_index)
                shader_index[src] = idx
                shader_srcs[idx] = src
            sidx = shader_index[src]
            disp_meta.append({"shader": sidx, "function_name": _renamed_fn(sidx, d.function_name),
                              "grid": list(d.grid), "tg": list(d.tg),
                              "slot_indices": list(d.buf_slot_indices), "offsets": list(d.buf_offsets)})

        slot_meta = []
        for si, slot in enumerate(plan.slots):
            if isinstance(slot, WeightSlot):
                arr = plan.weight_bindings[si]
                bp = arr.base_ptr
                if bp not in weight_blob:
                    wi = len(weight_blob)
                    weight_blob[bp] = wi
                    weight_bytes_by_idx[wi] = _read_addr(arr.base_ptr, arr.metal_nbytes)
                slot_meta.append({"role": "weight", "blob": weight_blob[bp], "nbytes": arr.metal_nbytes})
            elif isinstance(slot, IntermediateSlot):
                slot_meta.append({"role": "intermediate", "phys": slot.physical_idx})
            elif si in pinned_role:
                slot_meta.append({"role": pinned_role[si], "nbytes": slot.nbytes})
            else:
                pp, pre_ptr, voff = post[si], info[si][2], info[si][3]
                snap_bytes = snap.get(pre_ptr)
                storage_nbytes = len(snap_bytes) if snap_bytes else info[si][4]
                bid = buffer_id(pp, storage_nbytes, snap_bytes, pp in state_ptrs)
                slot_meta.append({"role": "buffer", "buffer": bid, "view_offset": voff, "nbytes": info[si][4]})

        token_out = next(({"slot_idx": e.slot_idx, "byte_offset": e.byte_offset}
                          for e in reversed(plan.output_mapping)
                          if isinstance(e, OutputSlot) and tuple(e.shape) == (1, 1) and e.dtype.itemsize == 8), None)
        feedback = []
        for o_idx, dst in plan.mutation_input_slots.items():
            e = plan.output_mapping[o_idx]
            if isinstance(e, OutputSlot):
                nb = 1
                for s in e.shape:
                    nb *= s
                feedback.append({"src_slot": e.slot_idx, "byte_offset": e.byte_offset,
                                 "nbytes": nb * e.dtype.itemsize, "dst_slot": dst})
        written = sorted({si for d in plan.dispatches for si in d.write_slot_indices})
        return {"dispatches": disp_meta, "slots": slot_meta, "dep_groups": plan.dep_groups,
                "phys_sizes": [a.metal_nbytes for a in plan.phys_arrays],
                "token_out": token_out, "feedback": feedback, "written_slots": written}

    pf_plan, dc_plan = emit_plan("prefill"), emit_plan("decode")
    weights = [weight_bytes_by_idx[i] for i in range(len(weight_blob))]
    consts = [const_bytes_by_idx[i] for i in range(const_count[0])]

    # --- compile the metallib(s) ---
    typer.echo(f"compiling {len(shader_srcs)} shaders → metallib for {plats}")
    for p in plats:
        _build_metallib(shader_srcs, pack_dir / "shaders", _PLATFORMS[p], p)

    # --- tokenizer ---
    tok = load_gguf_tokenizer(Path(resolved.path))
    tok.save_pretrained(str(pack_dir / "tokenizer"))
    # swift-transformers' AutoTokenizer.from(modelFolder:) requires config.json
    # (model_type → tokenizer class). The chat template ships as chat_template.jinja
    # and is passed explicitly on-device.
    hf.config.to_json_file(str(pack_dir / "tokenizer" / "config.json"))

    # --- resident-set / min_ram accounting (full set the engine allocates) ---
    weight_bytes = sum(len(b) for b in weights)
    const_bytes = sum(len(b) for b in consts)
    state_bytes = sum(b["nbytes"] for b in buffers.values() if b["kind"] == "state")
    interm_bytes = sum(pf_plan["phys_sizes"]) + sum(dc_plan["phys_sizes"])
    min_ram = weight_bytes + const_bytes + state_bytes + interm_bytes

    # The full stop set the golden was generated with (config + generation_config
    # eos, plus gemma's <end_of_turn>=106) — a single config eos misses the chat
    # turn marker, so the model loops past it.
    eos_ids = list(resolve_eos_tokens(hf)) or [-1]
    manifest = {
        "format": "alloypack/1", "model": model, "arch": meta["arch"], "vocab": vocab,
        "eos": eos_ids,
        "native_ctx": meta["native_ctx"], "prefill_chunk": pf_chunk,
        "metallib": [f"default.{p}.metallib" for p in plats],
        "n_shaders": len(shader_srcs),
        "roles": {k: {r: s for r, s in v.items() if s is not None} for k, v in roles.items()},
        "prefill": pf_plan, "decode": dc_plan,
        "buffers": [buffers[p] for p in sorted(buffers, key=lambda x: buffers[x]["id"])],
        "golden": {"prompt": prompt, "tokens": golden[len(prompt):], "max_new": max_new},
        "resident": {"weights": weight_bytes, "consts": const_bytes, "kv_state": state_bytes,
                     "intermediates": interm_bytes, "min_ram": min_ram},
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # pack.bin (structure + weights + consts) written into pack_dir, then copied
    # out as the distributable <id>.alloypack.bin.
    structure = {
        f.relative_to(pack_dir).as_posix(): f.read_bytes()
        for f in sorted(pack_dir.rglob("*")) if f.is_file() and f.name != "catalog_entry.json"
    }
    pack_bin = pack_dir / "pack.bin"
    _write_pack(pack_bin, structure, weights, consts)
    archive_path = out / f"{slug}.alloypack.bin"
    shutil.copyfile(pack_bin, archive_path)
    sha = hashlib.sha256(pack_bin.read_bytes()).hexdigest()
    download_bytes = pack_bin.stat().st_size
    archive_path.with_suffix(".bin.sha256").write_text(sha + "\n")

    catalog_entry = {
        "id": slug, "family": meta["arch"], "display_name": meta["name"] or model,
        "params": meta["size_label"], "quant": meta["quant"], "native_ctx": meta["native_ctx"],
        "modality": "text", "download_bytes": download_bytes, "min_ram": min_ram,
        "license": "see-model-card", "sha256": sha, "url": None,
    }
    (pack_dir / "catalog_entry.json").write_text(json.dumps(catalog_entry, indent=2))

    typer.echo(f"  weights {weight_bytes/1e6:.0f}MB · consts {const_bytes/1e6:.1f}MB · "
               f"KV(native) {state_bytes/1e6:.0f}MB · interm {interm_bytes/1e6:.0f}MB")
    typer.echo(f"  min_ram {min_ram/1e9:.2f}GB · pack {download_bytes/1e6:.0f}MB · sha256 {sha[:12]}…")
    typer.echo(f"wrote {archive_path}")

    if validate:
        _validate_pack(pack_dir, pack_bin, plats)


def _validate_pack(pack_dir: Path, archive_path: Path, plats: list[str]) -> None:
    """Run the full prefill→decode generation loop from the pack using ONLY the
    low-level Metal runtime — loading the macOS metallib (the device artifact),
    reading weights + consts from the single-file `<id>.alloypack.bin`, and
    allocating KV zeroed on-device — and confirm it reproduces the golden tokens.
    This validates the metallib, plans, weights, slot roles, and decode feedback
    end-to-end. iOS metallib parity is confirmed on-device (M1)."""
    if "macos" not in plats:
        typer.echo("  validation skipped (no macOS metallib built on this host)")
        return
    typer.echo("validating: full generation replay from metallib pack")

    man = json.loads((pack_dir / "manifest.json").read_text())
    lib = str(pack_dir / "shaders" / "default.macos.metallib")
    SLOT_INPUT, SLOT_WEIGHT, SLOT_INTERMEDIATE = 0, 1, 2

    def alloc_with(data: bytes) -> int:
        h = _metal_ext.buf_alloc(len(data))
        ctypes.memmove(_metal_ext.buf_ptr(h), data, len(data))
        return h

    def alloc_zero(n: int) -> int:
        h = _metal_ext.buf_alloc(n)
        ctypes.memset(_metal_ext.buf_ptr(h), 0, n)
        return h

    def write_vals(h: int, vals, dtype) -> None:
        b = np.asarray(vals, dtype=dtype).tobytes()
        ctypes.memmove(_metal_ext.buf_ptr(h), b, len(b))

    def read_i64(h: int, off: int) -> int:
        return int(np.frombuffer((ctypes.c_char * 8).from_address(_metal_ext.buf_ptr(h) + off), dtype=np.int64)[0])

    pso_cache: dict[str, int] = {}

    def pso_for(fn: str) -> int:
        if fn not in pso_cache:
            pso_cache[fn] = _metal_ext.compile_metallib(lib, fn)
        return pso_cache[fn]

    _files, weight_idx, const_idx = _read_pack_index(archive_path)
    buf_handle: dict[int, int] = {}
    for b in man["buffers"]:
        buf_handle[b["id"]] = (alloc_zero(b["nbytes"]) if b["kind"] == "state"
                               else alloc_with(_read_pack_entry(archive_path, const_idx[b["blob"]])))
    weight_handle: dict[int, int] = {}

    def weight_h(blob: int) -> int:
        if blob not in weight_handle:
            weight_handle[blob] = alloc_with(_read_pack_entry(archive_path, weight_idx[blob]))
        return weight_handle[blob]

    def setup_plan(kind: str):
        p = man[kind]
        phys = [alloc_zero(n) for n in p["phys_sizes"]]
        pinned: dict[str, int] = {}
        slot_handle = [0] * len(p["slots"])
        slots_data, input_updates = [], []
        for si, s in enumerate(p["slots"]):
            role = s["role"]
            if role == "weight":
                h = weight_h(s["blob"])
                slot_handle[si] = h
                slots_data.append((SLOT_WEIGHT, -1, h, s["nbytes"]))
            elif role == "intermediate":
                h = phys[s["phys"]]
                slot_handle[si] = h
                slots_data.append((SLOT_INTERMEDIATE, -1, h, p["phys_sizes"][s["phys"]]))
            elif role in ("token", "cache_position", "last_real", "attention_mask"):
                h = alloc_zero(s["nbytes"])
                pinned[role] = h
                slot_handle[si] = h
                slots_data.append((SLOT_INPUT, si, 0, s["nbytes"]))
                input_updates.append((si, h, 0))
            else:
                h = buf_handle[s["buffer"]]
                slot_handle[si] = h
                slots_data.append((SLOT_INPUT, si, 0, s["nbytes"]))
                input_updates.append((si, h, s["view_offset"]))
        dispatches = [(pso_for(d["function_name"]), list(d["slot_indices"]), list(d["offsets"]),
                       tuple(d["grid"]), tuple(d["tg"])) for d in p["dispatches"]]
        ph = _metal_ext.register_plan(dispatches, slots_data, p["dep_groups"], p["written_slots"])
        return ph, pinned, slot_handle, input_updates

    def apply_feedback(kind: str, slot_handle) -> None:
        p = man[kind]
        for f in p["feedback"]:
            src = _metal_ext.buf_ptr(slot_handle[f["src_slot"]]) + f["byte_offset"]
            dst_voff = p["slots"][f["dst_slot"]].get("view_offset", 0)
            ctypes.memmove(_metal_ext.buf_ptr(slot_handle[f["dst_slot"]]) + dst_voff, src, f["nbytes"])

    prompt, golden, eos = man["golden"]["prompt"], man["golden"]["tokens"], man["eos"]
    max_new, chunk, N = man["golden"]["max_new"], man["prefill_chunk"], len(man["golden"]["prompt"])
    pf_h, pf_pin, pf_sh, pf_iu = setup_plan("prefill")
    dc_h, dc_pin, dc_sh, dc_iu = setup_plan("decode")

    def cp_dtype(plan, n_elem):  # cache_position element width from the slot's byte size
        idx = next(i for i, s in enumerate(plan["slots"]) if s["role"] == "cache_position")
        return np.int32 if plan["slots"][idx]["nbytes"] // n_elem == 4 else np.int64

    def decode_step(t: int, pos: int) -> int:
        write_vals(dc_pin["token"], [t], np.int64)
        write_vals(dc_pin["cache_position"], [pos], cp_dtype(man["decode"], 1))
        _metal_ext.dispatch_plan(dc_h, dc_iu)
        apply_feedback("decode", dc_sh)
        return read_i64(dc_sh[man["decode"]["token_out"]["slot_idx"]], man["decode"]["token_out"]["byte_offset"])

    # Cold-prefill the first chunk (right-padded; causal attention discards pad rows),
    # sample at last_real = coldN-1. The general engine path: a long prompt is
    # cold-prefilled one chunk, then the rest rides the warm decode-loop.
    coldN = min(chunk, N)
    write_vals(pf_pin["token"], list(prompt[:coldN]) + [0] * (chunk - coldN), np.int64)
    write_vals(pf_pin["cache_position"], list(range(chunk)), cp_dtype(man["prefill"], chunk))
    if "last_real" in pf_pin:
        write_vals(pf_pin["last_real"], [coldN - 1], np.int64)
    _metal_ext.dispatch_plan(pf_h, pf_iu)
    cl = man["prefill"]["feedback"][0]["dst_slot"]
    cl_voff = man["prefill"]["slots"][cl].get("view_offset", 0)
    ctypes.memmove(_metal_ext.buf_ptr(pf_sh[cl]) + cl_voff, np.array([coldN], np.int64).tobytes(), 8)
    pred = read_i64(pf_sh[man["prefill"]["token_out"]["slot_idx"]], man["prefill"]["token_out"]["byte_offset"])

    # Warm decode-loop the rest of the prompt (primed cache), then generate.
    pos = coldN
    for t in prompt[coldN:]:
        pred = decode_step(t, pos)
        pos += 1
    gen = [pred]
    for _ in range(max_new - 1):
        nxt = decode_step(gen[-1], pos)
        pos += 1
        gen.append(nxt)
        if nxt in eos:
            break

    ok = gen == golden[:len(gen)]
    typer.echo(f"  cold({coldN})+warm({N - coldN}) generated: {gen}")
    typer.echo(f"  golden:                  {golden[:len(gen)]}")
    if not ok:
        raise typer.Exit("validation FAILED: generated tokens differ from golden")
    typer.echo("  ✅ pack reproduces golden tokens (metallib end-to-end, KV zeroed on-device)")


# Required keys + types for a catalog entry; the catalog is a list of these.
_CATALOG_FIELDS = {
    "id": str, "family": str, "display_name": str, "params": str, "quant": str,
    "native_ctx": int, "modality": str, "download_bytes": int, "min_ram": int,
    "license": str, "sha256": str, "url": str,
}


def _validate_catalog(models: list[dict]) -> list[str]:
    """Return a list of schema problems (empty = valid)."""
    errs: list[str] = []
    seen: set[str] = set()
    for i, m in enumerate(models):
        for key, typ in _CATALOG_FIELDS.items():
            if key not in m:
                errs.append(f"[{i}] missing {key!r}")
            elif m[key] is None or not isinstance(m[key], typ):
                errs.append(f"[{i}] {key!r} must be {typ.__name__}, got {m[key]!r}")
        mid = m.get("id")
        if mid in seen:
            errs.append(f"[{i}] duplicate id {mid!r}")
        seen.add(mid)
    return errs


def pack_publish(
    dist: Path = typer.Argument(Path("dist"), help="Directory of built *.alloypack packs."),
    hf_org: str = typer.Option("alloy", "--hf-org", help="HuggingFace org hosting the packs + catalog."),
    upload: bool = typer.Option(False, "--upload", help="Push packs + catalog to HuggingFace (needs HF token)."),
) -> None:
    """Assemble `catalog.json` from built packs and (optionally) publish to HuggingFace.

    Each pack → its own HF model repo `{org}/{id}` holding `{id}.alloypack.bin`
    (the single-file pack); the catalog → `{org}/catalog/catalog.json`. The catalog
    is always written locally to `dist/catalog.json`; `--upload` pushes (gated on
    `huggingface_hub`)."""
    entries = sorted(dist.glob("*.alloypack/catalog_entry.json"))
    if not entries:
        raise typer.Exit(f"no *.alloypack/catalog_entry.json under {dist}")

    models: list[dict] = []
    for ep in entries:
        entry = json.loads(ep.read_text())
        slug = entry["id"]
        entry["url"] = f"https://huggingface.co/{hf_org}/{slug}/resolve/main/{slug}.alloypack.bin"
        models.append(entry)

    problems = _validate_catalog(models)
    if problems:
        for p in problems[:20]:
            typer.echo(f"  catalog ✗ {p}")
        raise typer.Exit(f"catalog failed schema validation ({len(problems)} problems)")

    catalog = {"schema_version": 1, "models": models}
    (dist / "catalog.json").write_text(json.dumps(catalog, indent=2))
    typer.echo(f"catalog.json: {len(models)} models, schema-valid ✅ → {dist / 'catalog.json'}")
    for m in models:
        typer.echo(f"  {m['id']:<16} {m['params']:>6} {m['quant']:<8} "
                   f"dl {m['download_bytes']/1e6:.0f}MB · min_ram {m['min_ram']/1e9:.2f}GB")

    if not upload:
        typer.echo("(local only — pass --upload to push to HuggingFace)")
        return

    api = HfApi()
    for ep in entries:
        slug = json.loads(ep.read_text())["id"]
        pack_bin = ep.parent.with_suffix(".alloypack.bin")
        repo = f"{hf_org}/{slug}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(pack_bin), path_in_repo=pack_bin.name, repo_id=repo, repo_type="model")
        typer.echo(f"  uploaded {pack_bin.name} → {repo}")
    api.create_repo(f"{hf_org}/catalog", repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(dist / "catalog.json"), path_in_repo="catalog.json",
                    repo_id=f"{hf_org}/catalog", repo_type="model")
    typer.echo(f"  uploaded catalog.json → {hf_org}/catalog")

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Grid3D: TypeAlias = tuple[int, int, int]
BufferBinding: TypeAlias = tuple[int, int, int]
DispatchEntry: TypeAlias = tuple[int, Sequence[BufferBinding], Grid3D, Grid3D]
DispatchGroup: TypeAlias = Sequence[DispatchEntry]
PlanDispatch: TypeAlias = tuple[int, Sequence[int], Sequence[int], Grid3D, Grid3D]
PlanSlot: TypeAlias = tuple[int, int, int, int]
InputUpdate: TypeAlias = tuple[int, int, int]
Timing: TypeAlias = dict[str, float]
ProfileRecord: TypeAlias = dict[str, int | float | str | Grid3D]

_training_mode_flag: bool


def alloc_typed(
    nbytes: int,
    shape: tuple[int, ...],
    dtype: str,
) -> tuple[NDArray[np.generic], int, int]: ...


def buf_alloc(nbytes: int) -> int: ...


def buf_handle_for_ptr(ptr: int) -> int: ...


def buf_nbytes(handle: int) -> int: ...


def buf_numpy(handle: int, shape: tuple[int, ...], dtype: str) -> NDArray[np.generic]: ...


def buf_ptr(handle: int) -> int: ...


def buf_release(handle: int) -> None: ...


def clear_buffer_cache() -> None: ...


def compile_metallib(path: str, function_name: str) -> int: ...


def compile_msl(source: str, function_name: str) -> int: ...


def device_info() -> dict[str, str | int | bool]: ...


def dispatch(groups: Sequence[DispatchGroup]) -> Timing: ...


def dispatch_plan(
    plan_handle: int,
    input_updates: Sequence[InputUpdate],
    serialized: bool = False,
    defer_wait: bool = False,
) -> Timing: ...


def dispatch_plan_greedy_loop(
    plan_handle: int,
    input_updates: Sequence[InputUpdate],
    token_input_slot_idx: int,
    cache_position_slot_idx: int,
    token_output_slot_idx: int,
    token_output_byte_offset: int,
    generated_handle: int,
    generated_count: int,
    initial_token: int,
    initial_cache_position: int,
    mutation_pairs: Sequence[tuple[int, int, int, int]] = (),
    eos_token_ids: Sequence[int] = (),
) -> Timing: ...


def dispatch_plan_profiled(
    plan_handle: int,
    input_updates: Sequence[InputUpdate],
) -> list[ProfileRecord]: ...


def gpu_sync() -> None: ...


def pipeline_max_threads(handle: int) -> int: ...


def pipeline_thread_width(handle: int) -> int: ...


def register_plan(
    dispatches: Sequence[PlanDispatch],
    slots: Sequence[PlanSlot],
    groups: Sequence[Sequence[int]],
    written_slots: Sequence[int] = (),
) -> int: ...


def set_training_mode(mode: bool) -> None: ...

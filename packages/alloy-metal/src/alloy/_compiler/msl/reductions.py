"""Reduction-related MSL emitter methods."""

from __future__ import annotations

from collections.abc import Callable

from alloy._compiler.tile_ir import Reduce, SimdReduce
from alloy._compiler.msl.context import PER_THREAD, ValLoc


class ReductionEmitterMixin:
    @staticmethod
    def _reduce_identity(op: str) -> str:
        return {
            "max": "-INFINITY",
            "min": "INFINITY",
            "sum": "0.0f",
        }.get(op, "0.0f")

    def _emit_reduce_partial(self, op: Reduce):
        """Accumulate a partial reduction over BLOCK_N columns into existing accumulator."""
        name = op.result.name
        inp = op.input
        inp_loc = self._val_loc.get(inp.name, PER_THREAD)

        if inp_loc.kind != "shared":
            return

        buf, stride = inp_loc.name, inp_loc.stride
        rows = inp.shape[0] if len(inp.shape) == 2 else 1
        cols = self._block_n

        identity = self._reduce_identity(op.op)
        combine = {"max": "max", "min": "min", "sum": "+"}.get(op.op, "+")

        self._emit("{")
        self._indent += 1
        self._emit(f"{self._acc_dtype} _chunk = {identity};")
        self._emit(f"if (_row < {rows}u) {{")
        self._indent += 1
        self._emit_col_loop_open("_n", cols)
        if combine in ("max", "min"):
            self._emit(f"_chunk = {combine}(_chunk, {buf}[_row * {stride}u + _n]);")
        else:
            self._emit(f"_chunk += {buf}[_row * {stride}u + _n];")
        self._indent -= 1
        self._emit("}")
        if self._tpr > 1:
            self._emit_simd_butterfly("_chunk", combine, self._tpr)
        self._indent -= 1
        self._emit("}")
        if combine in ("max", "min"):
            self._emit(f"{name} = {combine}({name}, _chunk);")
        else:
            self._emit(f"{name} += _chunk;")
        self._indent -= 1
        self._emit("}")

        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_2d_shared_loop(
        self,
        name: str,
        rows: int,
        cols: int,
        out_buf: str,
        out_stride: int,
        expr_fn: Callable[[str], str],
    ) -> None:
        """Emit a 2D row-guarded column loop that writes to shared memory.

        Shared by _emit_binop (shared path) and _emit_unaryop (shared path).
        expr_fn("_c") returns the MSL expression to store at each column.
        """
        self._emit(f"if (_row < {rows}u)")
        self._emit_col_loop_open("_c", cols)
        self._emit(
            f"{out_buf}[_row * {out_stride}u + _c] = {self._shmem_cast(expr_fn('_c'), out_buf)};"
        )
        self._indent -= 1
        self._emit("}")
        self._val_loc[name] = ValLoc("shared", out_buf, out_stride)
        # Mark shmem as touched by a row-guarded scalar op — simdgroups outside
        # the guard are idle and may race to the next cooperative op.
        self._scalar_shmem_dirty = True

    # --- Per-row reduction on shared memory tile ---

    def _emit_col_loop_open(self, var: str, cols: int) -> None:
        """Open a per-row column loop striped across the row's `tpr` lanes.

        The natural form `for (_c = _lane; _c < cols; _c += tpr)` starts at a
        runtime value, so the Metal compiler can't prove the trip count is
        uniform across the SIMD group and runs the WHOLE body under a predicate
        mask — the GPU profiler bills that as "Predication" (~6-7% per such loop
        on attention's softmax epilogue at depth). When `tpr` divides `cols`
        (the common power-of-two case) every lane runs exactly `cols/tpr`
        in-bounds iterations, so emit a CONSTANT-TRIP loop that the compiler
        unrolls with no per-iteration predicate. Same fix as the cooperative
        loads' `_emit_coop_loop_open`. Increments indent; the caller emits the
        body and closes one brace. Falls back to the data-dependent form for
        `tpr == 1` (scalar) or a non-dividing `cols`.
        """
        tpr = self._tpr
        if tpr > 1 and cols % tpr == 0:
            self._emit(f"for (ushort _ci = 0; _ci < {cols // tpr}u; _ci++) {{")
            self._indent += 1
            self._emit(f"uint {var} = _lane + _ci * {tpr}u;")
        elif tpr > 1:
            self._emit(f"for (uint {var} = _lane; {var} < {cols}u; {var} += {tpr}u) {{")
            self._indent += 1
        else:
            self._emit(f"for (uint {var} = 0; {var} < {cols}u; {var}++) {{")
            self._indent += 1

    def _emit_simd_butterfly(self, name: str, combine: str, width: int = 32):
        """Emit simd_shuffle_xor butterfly reduction for a scalar value.

        `width` is the lane span cooperating on one reduction (a power of
        two ≤ 32). Offsets descend from width/2 so the butterfly stays
        within each width-lane group — crucial when tpr < 32 packs several
        rows into one simdgroup (e.g. attention's softmax at tpr=16 puts two
        rows per simdgroup, so offset 16 would cross into the neighbour row).
        Defaults to 32 (a full simdgroup), preserving the cross-simdgroup
        and tpr=32 reduction paths bit-for-bit.
        """
        off = width // 2
        while off >= 1:
            if combine in ("max", "min"):
                self._emit(f"{name} = {combine}({name}, simd_shuffle_xor({name}, {off}u));")
            else:
                self._emit(f"{name} += simd_shuffle_xor({name}, {off}u);")
            off //= 2

    def _emit_2d_reduce_body(
        self,
        name: str,
        rows: int,
        combine: str,
        loop_var: str,
        loop_count: int,
        elem_expr: str,
        strided: bool,
    ):
        """Emit the inner loop + butterfly for a 2D per-row reduction."""
        self._emit(f"if (_row < {rows}u) {{")
        self._indent += 1
        if strided and self._tpr > 1:
            self._emit_col_loop_open(loop_var, loop_count)
        else:
            self._emit(f"for (uint {loop_var} = 0; {loop_var} < {loop_count}u; {loop_var}++) {{")
            self._indent += 1
        if combine in ("max", "min"):
            self._emit(f"{name} = {combine}({name}, {elem_expr});")
        else:
            self._emit(f"{name} += {elem_expr};")
        self._indent -= 1
        self._emit("}")
        if self._tpr > 1:
            self._emit_simd_butterfly(name, combine, self._tpr)
        self._indent -= 1
        self._emit("}")

    def _emit_reduce(self, op: Reduce):
        name = op.result.name
        inp = op.input
        inp_loc = self._val_loc.get(inp.name, PER_THREAD)

        # Pre-reduce hook (e.g. attention masking)
        pre_reduce = self._config.get("pre_reduce")
        if pre_reduce:
            pre_reduce(self, op)

        if inp_loc.kind == "local_array":
            D = self._local_arrays[inp.name]
            arr = self._exprs.get(inp.name, inp.name)
            rows = inp.shape[0] if len(inp.shape) == 2 else 1
            identity = self._reduce_identity(op.op)
            combine = {"max": "max", "min": "min", "sum": "+"}.get(op.op, "+")

            self._emit(f"{self._acc_dtype} {name} = {identity};")
            self._emit_2d_reduce_body(name, rows, combine, "_d", D, f"{arr}[_d]", strided=False)
            self._exprs[name] = name
            self._val_loc[name] = PER_THREAD
        elif inp_loc.kind == "shared":
            buf, stride = inp_loc.name, inp_loc.stride
            rows = inp.shape[0] if len(inp.shape) == 2 else 1
            cols = self._eff_cols(inp.shape[1] if len(inp.shape) == 2 else inp.shape[0])
            identity = self._reduce_identity(op.op)
            if op.axis == 0 and len(inp.shape) == 2 and rows > 1:
                # Reduce over ROWS → per-column result, shape (1, cols). Emitted
                # as a (cols,) local array indexed by `_c` (downstream
                # `_elem_access` reads `name[_c]`). Each thread computes the full
                # column-reduction (redundant across threads but correct). The
                # default `_emit_2d_reduce_body` path only reduces over columns
                # (axis=1, per-row scalar), so axis=0 needs its own loop.
                # Each thread reads ALL rows of the shared input, so the
                # producing write must be visible threadgroup-wide first.
                self._flush_tg_barrier()
                fn = {"max": "max", "min": "min"}.get(op.op)
                self._emit(f"{self._acc_dtype} {name}[{cols}];")
                self._emit(f"for (uint _c = 0; _c < {cols}u; _c++) {{")
                self._indent += 1
                self._emit(f"{self._acc_dtype} _acc = {identity};")
                self._emit(f"for (uint _r = 0; _r < {rows}u; _r++)")
                self._indent += 1
                elem = f"{self._acc_dtype}({buf}[_r * {stride}u + _c])"
                self._emit(f"_acc = {fn}(_acc, {elem});" if fn else f"_acc += {elem};")
                self._indent -= 1
                self._emit(f"{name}[_c] = _acc;")
                self._indent -= 1
                self._emit("}")
                self._exprs[name] = name
                self._val_loc[name] = ValLoc("local_array")
                self._local_arrays[name] = cols
            else:
                combine = {"max": "max", "min": "min", "sum": "+"}.get(op.op, "+")
                self._emit(f"{self._acc_dtype} {name} = {identity};")
                self._emit_2d_reduce_body(
                    name,
                    rows,
                    combine,
                    "_n",
                    cols,
                    f"{buf}[_row * {stride}u + _n]",
                    strided=True,
                )
                self._exprs[name] = name
                self._val_loc[name] = PER_THREAD
        else:
            # 1D butterfly reduction: simdgroup xor + cross-simdgroup via shared memory
            input_expr = self._get(inp)
            uid = self._reduce_counter
            self._reduce_counter += 1
            n_sg = self._threads // 32

            identity = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY"}.get(op.op, "0.0f")

            combine = {"max": "max", "min": "min", "sum": "+"}.get(op.op, "+")
            # Mask lanes beyond the tile's logical extent. A 1D (N,) tile is
            # loaded one-element-per-thread, but the cooperative load populates
            # EVERY threadgroup lane with `base[tid]` — threads N..threads-1
            # hold out-of-tile (adjacent-memory) values, not zeros. The butterfly
            # below sums all `self._threads` lanes, so without a mask those stray
            # lanes corrupt the result. (Chunked delta-rule's per-chunk g-sum hit
            # this: chunk 0's 8-lane sum pulled in chunk 1's g[8:16], scaling the
            # carried state by exp(Σg_chunk1).) Seed stray lanes with identity.
            n_elems = inp.shape[0] if inp.shape else self._threads
            if isinstance(n_elems, int) and n_elems < self._threads:
                seed_expr = f"(tid < {n_elems}u) ? ({input_expr}) : {identity}"
            else:
                seed_expr = input_expr
            self._emit(f"float {name} = {seed_expr};")
            self._emit_simd_butterfly(name, combine)
            # Cross-simdgroup via shared memory
            sg = f"_sg{uid}"
            ln = f"_ln{uid}"
            self._emit(f"uint {sg} = tid / 32, {ln} = tid % 32;")
            # Pre-write barrier. Skipping it on uid==0 is incorrect when
            # the reduction lives inside a loop: at runtime the "first"
            # emit re-executes for iteration N+1 after iteration N's last
            # reduction wrote to _red, so we still need the barrier.
            # Unconditional emit is the safe choice; the no-op cost on
            # uid==0 of a kernel-wide-first reduction is negligible.
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._emit(f"if ({ln} == 0) _red[{sg}] = {name};")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._emit(f"{name} = ({sg} == 0 && {ln} < {n_sg}u) ? _red[{ln}] : {identity};")
            self._emit(f"if ({sg} == 0) {{")
            self._indent += 1
            self._emit_simd_butterfly(name, combine)
            self._indent -= 1
            self._emit("}")
            # Broadcast final result to all threads
            self._emit(f"if ({sg} == 0 && {ln} == 0) _red[0] = {name};")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._emit(f"{name} = _red[0];")

            self._exprs[name] = name
            self._val_loc[name] = PER_THREAD

    def _emit_simd_reduce(self, op: SimdReduce):
        """Emit simd_sum/simd_max/simd_min — hardware cross-lane reduction."""
        name = op.result.name
        inp_expr = self._get(op.input)
        simd_fn = {"sum": "simd_sum", "max": "simd_max", "min": "simd_min"}.get(op.op, "simd_sum")
        self._emit(f"{self._acc_dtype} {name} = {simd_fn}({self._acc_dtype}({inp_expr}));")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

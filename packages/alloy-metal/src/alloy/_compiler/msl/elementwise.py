"""Elementwise MSL emitter methods."""

from __future__ import annotations

from alloy._compiler.dtypes import ALLOY_TO_MSL
from alloy._compiler.msl.context import ADDRESS, PER_THREAD, ValLoc, msl_dtype_for_value
from alloy._compiler.msl.math import (
    BINOP_FUNC_MSL,
    BINOP_MSL,
    CMPOP_MSL,
    UNARY_FUNC_MSL,
    UNARY_MSL,
    format_scalar_op,
)
from alloy._compiler.tile_ir import (
    BinOp,
    BoolOp,
    Compare,
    Dot4,
    ExpandDims,
    InterleaveVec4,
    Load4Vec,
    MakeRange,
    Select,
    TernaryOp,
    TileValue,
    UnaryOp,
)


class ElementwiseEmitterMixin:
    def _elem_access(self, val: TileValue, col_var: str) -> str:
        """Return MSL expression for one element of val at col_var.

        Handles all value locations: shared, local_array, per_thread, address.
        """
        # Deferred cast on a 2D shared/local input: recurse into the input's
        # element access and wrap with the cast helper. See _emit_cast.
        deferred = self._deferred_cast.get(val.name)
        if deferred is not None:
            input_val, msl_type = deferred
            return f"_alloy_cast_{msl_type}({self._elem_access(input_val, col_var)})"
        loc = self._val_loc.get(val.name, PER_THREAD)
        if loc.kind == "shared":
            row_var = "0u" if len(val.shape) == 2 and val.shape[0] == 1 else "_row"
            col_expr = "0u" if len(val.shape) == 2 and val.shape[1] == 1 else col_var
            return f"{loc.name}[{row_var} * {loc.stride}u + {col_expr}]"
        if loc.kind == "local_array":
            arr = self._exprs.get(val.name, val.name)
            return f"{arr}[{col_var}]"
        # ADDRESS values from ExpandDims of a local_array: the expression is
        # the array name, which needs indexing.  Check if the underlying
        # expression refers to a known local array.
        expr = self._exprs.get(val.name)
        if expr and expr in self._local_arrays:
            # (M, 1) = row vector: data varies by row, constant across cols
            # (1, N) = col vector: data varies by col, constant across rows
            if len(val.shape) == 2 and val.shape[1] == 1:
                return f"{expr}[_row]"
            return f"{expr}[{col_var}]"
        return self._get(val)

    def _cow_save(self, val, loc, rows: int, cols: int) -> None:
        """Copy-on-write: save a shared-memory value to a per-thread local
        array before the shared buffer gets overwritten by an in-place op.

        Emits a local array + copy loop, then redirects the value's location
        so future _elem_access reads from the local array.
        """
        arr = f"_cow{self._cow_counter}"
        self._cow_counter += 1
        buf, stride = loc.name, loc.stride
        self._emit(f"float {arr}[{cols}];")
        self._emit(f"if (_row < {rows}u)")
        self._emit(f"for (uint _c = 0; _c < {cols}u; _c++)")
        self._indent += 1
        self._emit(f"{arr}[_c] = {buf}[_row * {stride}u + _c];")
        self._indent -= 1
        # Redirect: future reads of this value use the local array
        self._exprs[val.name] = arr
        self._val_loc[val.name] = ValLoc("local_array")
        self._local_arrays[val.name] = cols
        self._use_counts[val.name] = self._use_counts.get(val.name, 1) - 1

    def _shmem_cast(self, expr: str, buf_name: str | None = None) -> str:
        """Wrap an RHS in an explicit `bfloat(...)` when the target shmem
        slot is bfloat. MSL's `bfloat` disallows implicit narrowing from
        `float`; `half` and integer types either accept implicit narrowing
        or never receive float expressions.

        Per-buffer dtype: when `buf_name` is supplied, use that buffer's
        actual dtype (set by the planner's per-buffer dtype assignment).
        Without a buf_name, falls back to the kernel-global `_shmem_dtype`
        (legacy callers).
        """
        target_dt = self._buf_shmem_dtype(buf_name) if buf_name else self._shmem_dtype
        if target_dt == "bfloat":
            return f"bfloat({expr})"
        if target_dt == "half":
            return f"half({expr})"
        return expr

    def _emit_load4_vec(self, op: Load4Vec):
        """Emit vectorized 4-element load. Uses the pointer's declared type."""
        name = op.result.name
        ptr = self._get(op.ptr)
        offs = self._get(op.offsets)
        # Resolve the buffer's actual dtype the same way scalar Load does —
        # `op.ptr.dtype` is unreliable because alloy declares all buffer
        # params as `device const float*` in the kernel signature regardless
        # of the real element type. `_buffer_dtypes` is the authoritative map.
        ptr_name = op.ptr.name if op.ptr else None
        buf_dt = self._buffer_dtypes.get(ptr_name, "float") if ptr_name else "float"
        if buf_dt in ("half", "bfloat", "f16", "bf16"):
            vec_type, elem_type = "half4", "half"
        elif buf_dt in ("u8", "uchar"):
            vec_type, elem_type = "uchar4", "uchar"
        elif buf_dt in ("i8", "char"):
            vec_type, elem_type = "char4", "char"
        elif buf_dt in ("u32", "uint"):
            # 16-byte load of 16 packed int8 codes; components reinterpret via as_char4.
            vec_type, elem_type = "uint4", "uint"
        else:
            vec_type, elem_type = "float4", "float"
        # Pointer-cast to the element type BEFORE adding `offs` so the
        # arithmetic uses the correct stride (1 byte per uchar, 4 per float
        # etc.). Without the inner cast, `B_q4 + offs` advances by
        # `offs * sizeof(float)` bytes when B_q4 is actually uchar — silently
        # reading from the wrong location.
        self._emit(
            f"{vec_type} {name} = *reinterpret_cast<device const {vec_type}*>("
            f"reinterpret_cast<device const {elem_type}*>({ptr}) + {offs});"
        )
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_load_wide(self, op):
        """Emit one wide reinterpret load: `<T> v = *(device const T*)((device
        const char*)ptr + byte_off);` — llama.cpp's `(uint16_t*)scales` read."""
        name = op.result.name
        ptr = self._get(op.ptr)
        offs = self._get(op.offsets)
        msl_t = ALLOY_TO_MSL.get(op.wide, op.wide)
        self._emit(
            f"{msl_t} {name} = *reinterpret_cast<device const {msl_t}*>("
            f"reinterpret_cast<device const char*>({ptr}) + {offs});"
        )
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_dot4(self, op: Dot4):
        """Emit dot(float4(a), float4(b)) → scalar f32."""
        name = op.result.name
        a = self._get(op.a)
        b = self._get(op.b)
        self._emit(f"float {name} = dot(float4({a}), float4({b}));")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_as_char4(self, op):
        """Emit `float4 v = float4(as_type<char4>(a.{x,y,z,w}));` — one uint
        component of a uint4 reinterpreted as 4 int8 codes, converted once."""
        name = op.result.name
        a = self._get(op.a)
        component = "xyzw"[op.lane]
        self._emit(f"float4 {name} = float4(as_type<char4>({a}.{component}));")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_unpack4(self, op):
        """Emit `float v = float(a.{x,y,z,w});` — scalar extract from vec4."""
        name = op.result.name
        a = self._get(op.a)
        component = "xyzw"[op.lane]
        # Cast to float so downstream scalar arithmetic matches dot4's f32
        # result regardless of whether `a` is half4/bfloat4/float4.
        self._emit(f"float {name} = float({a}.{component});")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _emit_interleave_vec4(self, op: InterleaveVec4):
        """Emit vec4(lo.x, hi.x, lo.y, hi.y) or vec4(lo.z, hi.z, lo.w, hi.w).

        The MSL constructor compiles to simdgroup-internal shuffles on M-series
        GPUs (no DRAM round-trip). Result type matches the input vec type.
        """
        name = op.result.name
        lo = self._get(op.lo)
        hi = self._get(op.hi)
        dtype = op.lo.dtype if op.lo else "u8"
        if dtype in ("f16", "bf16"):
            vec_type = "half4"
        elif dtype in ("u8", "uchar"):
            vec_type = "uchar4"
        elif dtype in ("i8", "char"):
            vec_type = "char4"
        else:
            vec_type = "float4"
        if op.half == 0:
            expr = f"{vec_type}({lo}.x, {hi}.x, {lo}.y, {hi}.y)"
        else:
            expr = f"{vec_type}({lo}.z, {hi}.z, {lo}.w, {hi}.w)"
        self._emit(f"{vec_type} {name} = {expr};")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    # --- BinOp — per-thread scalars, per-row shared, local arrays, address ---

    def _emit_binop(self, op: BinOp):
        name = op.result.name
        lhs = op.lhs
        rhs = op.rhs
        lhs_loc = self._val_loc.get(lhs.name, PER_THREAD)
        rhs_loc = self._val_loc.get(rhs.name, PER_THREAD)
        result_shape = op.result.shape if op.result else ()

        op_sym = BINOP_MSL.get(op.op)
        op_func = BINOP_FUNC_MSL.get(op.op)

        def _fmt(l_expr, r_expr):
            if op_func:
                return f"{op_func}({l_expr}, {r_expr})"
            return f"({l_expr} {op_sym} {r_expr})"

        # Address computations: just store expression, no emitted code.
        # Skip when the other operand is data (shared/local_array).
        lhs_data = lhs_loc.kind in ("shared", "local_array")
        rhs_data = rhs_loc.kind in ("shared", "local_array")
        if (
            (lhs_loc.kind == "address" or rhs_loc.kind == "address")
            and not lhs_data
            and not rhs_data
        ):
            if result_shape and len(result_shape) > 0:
                self._exprs[name] = format_scalar_op(op, self._get)
                self._val_loc[name] = ADDRESS
                return

        # Flush the deferred barrier only when this BinOp will actually read
        # shmem. Per-thread arithmetic doesn't need prior cooperative writes
        # visible; without this guard, mask computations between Load A and
        # Load B in a GEMM K-loop body materialize a barrier per K-tile that
        # the MMA's own _check_flush_for would have absorbed.
        if lhs_data or rhs_data or (
            result_shape and len(result_shape) == 2
            and self._shmem_plan.get(name) is not None
        ):
            self._flush_tg_barrier()

        # Scalar (both per-thread or address)
        if not result_shape or len(result_shape) == 0:
            expr = format_scalar_op(op, self._get)
            if hasattr(self, "_carried_finals") and name in self._carried_finals:
                self._emit(f"float {name} = {expr};")
                self._exprs[name] = name
            elif self._scalar_uses.get(name, 0) > 1 and name not in self._opaque_vec:
                # CSE: a scalar referenced more than once is materialized into a
                # named temp instead of re-inlining its expression tree at every
                # use. Gives the Metal compiler clean input for CSE / loop
                # induction-variable formation on addresses.
                self._emit(f"{msl_dtype_for_value(op.result, 'int')} {name} = {expr};")
                self._exprs[name] = name
            else:
                self._exprs[name] = expr
            self._val_loc[name] = PER_THREAD
            return

        # 2D data operation — unified via _elem_access
        if len(result_shape) == 2:
            rows, cols = result_shape
            cols = self._eff_cols(cols)
            res_plan = self._shmem_plan.get(name)
            has_local = lhs_loc.kind == "local_array" or rhs_loc.kind == "local_array"
            has_shared = lhs_loc.kind == "shared" or rhs_loc.kind == "shared"

            if has_local:
                # Output to local array — prefer lhs as mutation target
                if lhs_loc.kind == "local_array":
                    la_val = lhs
                else:
                    la_val = rhs
                D = self._local_arrays[la_val.name]
                arr = self._exprs.get(la_val.name, la_val.name)
                can_mutate = self._local_arr_can_mutate(la_val.name)
                self._consume_ref(lhs.name)
                self._consume_ref(rhs.name)
                if not can_mutate:
                    arr = self._alloc_local_copy(arr, D)
                # Temporarily point la_val's expr at arr for _elem_access
                old_expr = self._exprs.get(la_val.name)
                self._exprs[la_val.name] = arr
                need_row_guard = has_shared
                if need_row_guard:
                    self._emit(f"if (_row < {rows}u) {{")
                    self._indent += 1
                l_e = self._elem_access(lhs, "_d")
                r_e = self._elem_access(rhs, "_d")
                self._emit(f"for (uint _d = 0; _d < {D}u; _d++) {arr}[_d] = {_fmt(l_e, r_e)};")
                if need_row_guard:
                    self._indent -= 1
                    self._emit("}")
                # Restore la_val's expr. The redirect above is per-op scratch;
                # leaving it set would permanently alias la_val to this op's
                # result array (breaks a reused parent like `zeros` across loop
                # iterations).
                if old_expr is not None:
                    self._exprs[la_val.name] = old_expr
                else:
                    self._exprs.pop(la_val.name, None)
                self._val_loc[name] = ValLoc("local_array")
                self._local_arrays[name] = D
                self._exprs[name] = arr
                if has_shared:
                    self._scalar_shmem_dirty = True
            elif has_shared:
                # Output to shared memory
                for loc in (lhs_loc, rhs_loc):
                    if loc.kind == "shared":
                        default_buf, default_stride = loc.name, loc.stride
                        break
                out_buf = res_plan[0] if res_plan else default_buf
                out_stride = res_plan[3] if res_plan else default_stride
                # Copy-on-write: if an operand lives in the output buffer and
                # has remaining consumers after this op, save it to a register
                # before the in-place overwrite clobbers it.
                for operand, loc in [(lhs, lhs_loc), (rhs, rhs_loc)]:
                    if (
                        loc.kind == "shared"
                        and loc.name == out_buf
                        and self._has_future_use(operand.name, op)
                    ):
                        self._cow_save(operand, loc, rows, cols)
                self._emit_2d_shared_loop(
                    name,
                    rows,
                    cols,
                    out_buf,
                    out_stride,
                    lambda c: _fmt(self._elem_access(lhs, c), self._elem_access(rhs, c)),
                )
            else:
                # Both per_thread scalars with 2D shape.
                # If the shmem plan has a buffer for this value (i.e. a downstream
                # Dot needs it), write to shmem so the Dot can load it via
                # simdgroup_load.  Otherwise keep as a per-thread expression.
                if res_plan:
                    out_buf, _, _, out_stride = res_plan
                    expr_str = _fmt(self._get(lhs), self._get(rhs))
                    self._emit_2d_shared_loop(
                        name,
                        rows,
                        cols,
                        out_buf,
                        out_stride,
                        lambda c, e=expr_str: e,
                    )
                else:
                    self._exprs[name] = _fmt(self._get(lhs), self._get(rhs))
                    self._val_loc[name] = PER_THREAD
            return

        # 1D result
        l = self._get(op.lhs)  # noqa: E741
        r = self._get(op.rhs)
        if op_func:
            self._exprs[name] = f"{op_func}({l}, {r})"
        elif op_sym:
            self._exprs[name] = f"({l} {op_sym} {r})"
        self._val_loc[name] = PER_THREAD

    def _emit_unaryop(self, op: UnaryOp):
        name = op.result.name
        inp = op.input
        inp_loc = self._val_loc.get(inp.name, PER_THREAD)
        result_shape = op.result.shape if op.result else ()

        # 2D paths below emit row loops that read shmem; flush deferred barrier
        # so prior writes are visible. Per-thread / address-only fallthrough
        # (the trailing `else`) doesn't access shmem, so it skips the flush.
        if len(result_shape) == 2 and (
            inp_loc.kind in ("shared", "local_array") or self._shmem_plan.get(name)
        ):
            self._flush_tg_barrier()

        func_name = UNARY_FUNC_MSL.get(op.op)
        prefix = UNARY_MSL.get(op.op)

        def _fmt(e):
            if op.op == "bitcast":
                target_msl = ALLOY_TO_MSL.get(op.result.dtype, "float")
                if target_msl == "float":
                    # Width-polymorphic (scalar + vec4) reinterpret; a scalar
                    # as_type<float> from a vec<int,4> is a size-mismatch error.
                    return f"_alloy_bitcast_float({e})"
                return f"as_type<{target_msl}>({e})"
            if func_name:
                return f"{func_name}({e})"
            if prefix:
                return f"({prefix}{e})"
            return e

        if inp_loc.kind == "shared" and len(result_shape) == 2:
            rows = result_shape[0]
            cols = self._eff_cols(result_shape[1])
            res_plan = self._shmem_plan.get(name)
            out_buf = res_plan[0] if res_plan else inp_loc.name
            out_stride = res_plan[3] if res_plan else inp_loc.stride
            # Copy-on-write: if the input shares the output buffer (in-place
            # mutation) and has live consumers downstream, save it to a local
            # array first so subsequent reads see the pre-mutation value (a SiLU
            # epilogue's `neg(t57)` would otherwise clobber the tile before
            # `mul(t57, sigmoid(t57))` reads it).
            if (
                inp_loc.name == out_buf
                and self._has_future_use(inp.name, op)
            ):
                self._cow_save(inp, inp_loc, rows, cols)
            self._emit_2d_shared_loop(
                name,
                rows,
                cols,
                out_buf,
                out_stride,
                lambda c: _fmt(self._elem_access(inp, c)),
            )
        elif inp_loc.kind == "local_array" and len(result_shape) == 2:
            D = self._local_arrays[inp.name]
            arr_name = self._exprs.get(inp.name, inp.name)
            can_mutate = self._local_arr_can_mutate(inp.name)
            self._consume_ref(inp.name)
            if not can_mutate:
                arr_name = self._alloc_local_copy(arr_name, D)
            self._emit(
                f"for (uint _d = 0; _d < {D}u; _d++) {arr_name}[_d] = {_fmt(f'{arr_name}[_d]')};"
            )
            self._val_loc[name] = ValLoc("local_array")
            self._local_arrays[name] = D
            self._exprs[name] = arr_name
        elif len(result_shape) == 2 and self._shmem_plan.get(name):
            # Per-thread scalar with 2D shape that a downstream Dot needs in shmem
            rows = result_shape[0]
            cols = self._eff_cols(result_shape[1])
            res_plan = self._shmem_plan[name]
            out_buf, _, _, out_stride = res_plan
            expr_str = format_scalar_op(op, self._get) or _fmt(self._get(inp))
            self._emit_2d_shared_loop(
                name,
                rows,
                cols,
                out_buf,
                out_stride,
                lambda c, e=expr_str: e,
            )
        else:
            self._exprs[name] = format_scalar_op(op, self._get)
            self._val_loc[name] = PER_THREAD

    def _emit_ternaryop(self, op: TernaryOp):
        name = op.result.name
        self._exprs[name] = format_scalar_op(op, self._get)
        self._val_loc[name] = PER_THREAD

    def _emit_select(self, op: Select):
        """Emit select(cond, true_val, false_val) — ternary conditional."""
        name = op.result.name
        result_shape = op.result.shape if op.result else ()

        true_loc = self._val_loc.get(op.true_val.name, PER_THREAD)
        false_loc = self._val_loc.get(op.false_val.name, PER_THREAD)

        # 2D: iterate per-element in shared memory
        if len(result_shape) == 2:
            self._flush_tg_barrier()
            rows, cols = result_shape
            cols = self._eff_cols(cols)
            res_plan = self._shmem_plan.get(name)

            # Find a shared source to determine output buffer
            for val, loc in [(op.true_val, true_loc), (op.false_val, false_loc)]:
                if loc.kind == "shared":
                    src_buf, src_stride = loc.name, loc.stride
                    out_buf = res_plan[0] if res_plan else src_buf
                    out_stride = res_plan[3] if res_plan else src_stride
                    break
            else:
                out_buf = res_plan[0] if res_plan else f"_s{name}"
                out_stride = res_plan[3] if res_plan else cols + 4

            # Build per-element condition: resolve 2D address comparisons
            # For (row_expr)[:, None] >= (col_expr)[None, :], the condition
            # uses _row and _c inside the element loop.
            cond_elem = self._resolve_2d_cond(op.cond, "_row", "_c")

            # Hoist a column-invariant condition operand out of the per-element
            # loop. A positional/causal mask `Compare(row_expr[:,None],
            # col_expr[None,:])` has a left side constant across columns; inlined
            # into the ternary the compiler recomputes the whole predicate every
            # column iteration (~6-7% "Predication" at depth on the causal mask).
            # Precompute it once per row into `_selL`. Only fires for the
            # row-vs-col Compare shape; compound BoolOp masks fall through.
            cond_op = self._op_map.get(op.cond.name)
            row_hoist = None
            if isinstance(cond_op, Compare):
                lhs_op = self._op_map.get(cond_op.lhs.name)
                rhs_op = self._op_map.get(cond_op.rhs.name)
                if (
                    isinstance(lhs_op, ExpandDims)
                    and lhs_op.axis == 1
                    and isinstance(rhs_op, ExpandDims)
                    and rhs_op.axis == 0
                ):
                    l_expr = self._resolve_1d_addr(cond_op.lhs, "_row", "_c")
                    r_expr = self._resolve_1d_addr(cond_op.rhs, "_row", "_c")
                    row_hoist = (l_expr, f"(_selL {CMPOP_MSL[cond_op.op]} {r_expr})")
                    cond_elem = row_hoist[1]

            self._emit(f"if (_row < {rows}u) {{")
            self._indent += 1
            if row_hoist is not None:
                self._emit(f"auto _selL = {row_hoist[0]};")
            # Stride columns across the row's tpr lanes so the masked `where`
            # parallelizes like the binop/unaryop/reduce row paths, in
            # constant-trip form (see _emit_col_loop_open). Without striping the
            # mask is the lone serial pass in a lane-parallel softmax epilogue.
            self._emit_col_loop_open("_c", cols)
            # Resolve both branches through the unified element accessor so
            # local_array (per-thread constant-row) and address-of-local_array
            # operands index correctly (`_la0[_c]`) instead of emitting a raw
            # `float(_la0)` cast of the array pointer.
            t_expr = self._elem_access(op.true_val, "_c")
            f_expr = self._elem_access(op.false_val, "_c")
            ternary = f"({cond_elem}) ? float({t_expr}) : float({f_expr})"
            self._emit(
                f"{out_buf}[_row * {out_stride}u + _c] = {self._shmem_cast(ternary, out_buf)};"
            )
            self._indent -= 1
            self._emit("}")
            self._indent -= 1
            self._emit("}")
            self._val_loc[name] = ValLoc("shared", out_buf, out_stride)
            return

        # Scalar — consuming a comparison as select condition, not a store mask
        self._mask_expr = None
        expr = format_scalar_op(op, self._get)
        # Materialize to a named variable — the operands may reference
        # variables from inner scopes (IfElse bodies) that are out of scope
        # at the point where this select result is consumed.
        out_dtype = msl_dtype_for_value(op.result, self._acc_dtype)
        # MSL's `bfloat` disallows implicit narrowing from a `float`-typed
        # ternary; `half` permits it. Wrap explicitly for bf16.
        if out_dtype == "bfloat":
            self._emit(f"{out_dtype} {name} = bfloat({expr});")
        else:
            self._emit(f"{out_dtype} {name} = {expr};")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

    def _resolve_2d_elem(self, val, loc, row_var, col_var):
        """Get per-element expression for a 2D value."""
        if loc.kind == "shared":
            row_expr = "0u" if len(val.shape) == 2 and val.shape[0] == 1 else row_var
            col_expr = "0u" if len(val.shape) == 2 and val.shape[1] == 1 else col_var
            return f"{loc.name}[{row_expr} * {loc.stride}u + {col_expr}]"
        return self._get(val)

    def _resolve_2d_cond(self, cond_val, row_var, col_var):
        """Resolve a 2D condition to a per-element expression.

        Handles Compare(expand_dims(row, 1), expand_dims(col, 0)) by
        substituting _row for the row range and _c for the col range.
        Also handles ExpandDims(cond_1d, axis): a 1D bool broadcast to 2D
        (e.g. `kv_mask = (j+rn) < N; kv_mask[None, :]`), which must be
        re-evaluated at col_var (axis=0) or row_var (axis=1).
        """
        op = self._op_map.get(cond_val.name)
        if isinstance(op, ExpandDims):
            var = col_var if op.axis == 0 else row_var
            return self._resolve_1d_cond_at(op.input, var)
        if isinstance(op, Compare):
            l_expr = self._resolve_1d_addr(op.lhs, row_var, col_var)
            r_expr = self._resolve_1d_addr(op.rhs, row_var, col_var)
            return f"({l_expr} {CMPOP_MSL[op.op]} {r_expr})"
        if isinstance(op, (BoolOp, BinOp)) and op.op in ("and", "bitand", "or", "bitor"):
            l = self._resolve_2d_cond(op.lhs, row_var, col_var)  # noqa: E741
            r = self._resolve_2d_cond(op.rhs, row_var, col_var)
            sym = "&&" if op.op in ("and", "bitand") else "||"
            return f"({l} {sym} {r})"
        return self._get(cond_val)

    def _resolve_1d_cond_at(self, cond_val, idx_var):
        """Evaluate a 1D Compare/BoolOp at a specific index var.

        Substitutes idx_var for any 1D MakeRange inside Compare operands, so
        that a cached scalar expression emitted with `tid` is not reused in
        a 2D element loop where the range must vary with `_c` (or `_row`).
        """
        op = self._op_map.get(cond_val.name)
        if isinstance(op, Compare):
            l_expr = self._resolve_range_at(op.lhs, idx_var)
            r_expr = self._resolve_range_at(op.rhs, idx_var)
            return f"({l_expr} {CMPOP_MSL[op.op]} {r_expr})"
        if isinstance(op, (BoolOp, BinOp)) and op.op in ("and", "bitand", "or", "bitor"):
            l = self._resolve_1d_cond_at(op.lhs, idx_var)  # noqa: E741
            r = self._resolve_1d_cond_at(op.rhs, idx_var)
            sym = "&&" if op.op in ("and", "bitand") else "||"
            return f"({l} {sym} {r})"
        return self._get(cond_val)

    def _resolve_1d_addr(self, val, row_var, col_var):
        """Resolve a 1D or expanded value to a scalar expression using row/col vars.

        expand_dims(x, 1) → x evaluated at row_var
        expand_dims(x, 0) → x evaluated at col_var
        scalar → as-is
        """
        op = self._op_map.get(val.name)
        if isinstance(op, ExpandDims):
            # axis=1 means row dimension, axis=0 means col dimension
            inner = op.input
            if op.axis == 1:
                return self._resolve_range_at(inner, row_var)
            else:
                return self._resolve_range_at(inner, col_var)
        if val.shape == ():
            return self._get(val)
        return self._get(val)

    def _resolve_range_at(self, val, idx_var):
        """Resolve a 1D range value (base + arange) at a specific index."""
        op = self._op_map.get(val.name)
        if isinstance(op, MakeRange):
            return f"(int){idx_var}"
        if isinstance(op, BinOp) and op.op == "add":
            l_op = self._op_map.get(op.lhs.name)
            r_op = self._op_map.get(op.rhs.name)
            if isinstance(r_op, MakeRange):
                return f"({self._get(op.lhs)} + (int){idx_var})"
            if isinstance(l_op, MakeRange):
                return f"((int){idx_var} + {self._get(op.rhs)})"
        return f"{self._get(val)}"

    def _emit_compare(self, op: Compare):
        expr = format_scalar_op(op, self._get)
        self._exprs[op.result.name] = expr
        self._val_loc[op.result.name] = PER_THREAD
        # Track for 1D mask guard on stores
        self._mask_expr = expr

    # --- Loop helpers ---

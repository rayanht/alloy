"""Memory load/store MSL emitter methods."""

from __future__ import annotations

from alloy._compiler.msl.context import PER_THREAD, ValLoc
from alloy._compiler.msl.math import format_scalar_op
from alloy._compiler.tile_ir import (
    BinOp,
    BoolOp,
    Compare,
    Constant,
    Load,
    MakeRange,
    Store,
    TileValue,
)


class MemoryEmitterMixin:
    # --- Address tracing for cooperative loads ---

    def _resolve_2d_addr(self, op) -> tuple[str, int, str, str]:
        """Resolve 2D addressing from semantic fields on Load/Store.

        Returns (row_start_expr, row_stride, col_start_expr, base_offset_expr).
        """
        assert (
            op.row_indices is not None and op.col_indices is not None and op.row_stride is not None
        ), (
            f"Missing semantic 2D addressing on {type(op).__name__} (result={op.result.name if op.result else '?'})"
        )
        row_start = self._extract_range_base(op.row_indices)
        col_start = self._extract_range_base(op.col_indices)
        row_stride = op.row_stride
        base_offset = self._get(op.base_offset) if op.base_offset is not None else "0"
        return row_start, row_stride, col_start, base_offset

    def _contains_make_range(self, val: TileValue) -> bool:
        """Whether `val`'s expression subtree contains a MakeRange (the per-row
        sweep the cooperative loader renders as `_r`)."""
        op = self._op_map.get(val.name)
        if isinstance(op, MakeRange):
            return True
        if isinstance(op, BinOp):
            return self._contains_make_range(op.lhs) or (
                op.rhs is not None and self._contains_make_range(op.rhs)
            )
        return False

    def _extract_range_base(self, val: TileValue) -> str:
        """Extract base expression from val = base + make_range(...)."""
        op = self._op_map.get(val.name)
        if isinstance(op, MakeRange):
            return str(op.start)
        if isinstance(op, BinOp) and op.op == "add":
            # Check if one operand is a MakeRange
            l_op = self._op_map.get(op.lhs.name)
            r_op = self._op_map.get(op.rhs.name)
            if isinstance(r_op, MakeRange):
                return self._get(op.lhs)
            if isinstance(l_op, MakeRange):
                return self._get(op.rhs)
            # Nested: a runtime scalar added onto a `program_base + make_range`
            # row, e.g. grouped-GEMM's `e*2I + (pn*BN + arange)`. The make_range
            # sits one level down; peel it from the range-bearing side (which the
            # loader turns into `_r`) and keep the other side as a scalar base.
            # Without this the whole expr falls through to `_get`, dragging the
            # make_range in as `tid` and corrupting every cooperatively-loaded row.
            if self._contains_make_range(op.rhs):
                return f"({self._get(op.lhs)} + {self._extract_range_base(op.rhs)})"
            if self._contains_make_range(op.lhs):
                return f"({self._extract_range_base(op.lhs)} + {self._get(op.rhs)})"
        if isinstance(op, BinOp) and op.op == "mod":
            # Pattern: (base + make_range) % modulus. The cooperative-load
            # emitter checks `_extract_row_modulus` separately to apply the
            # modulus per-thread; the bare base returned here is only used
            # when no modulus path is taken (e.g. for the cached-MSL hash).
            inner = self._extract_range_base(op.lhs)
            return inner
        return self._get(val)

    def _gather_index_load(self, row_indices: TileValue) -> Load | None:
        """If the row index is itself a buffer Load (a GATHER — e.g. MoE grouped GEMM
        gathers token rows via `ROW_TOKEN[rm]`), return that Load so the cooperative
        loader can read each row's source index per-row (`_gr = IDX[base + _r]`) instead
        of the affine `row_start + _r`. Peels single-input ops (Cast/Copy) so an int-cast
        index Load still matches. Affine row indices are ProgramId/MakeRange arithmetic —
        never a Load — so this is gated strictly to true gathers (affine fast path
        untouched). Requires the Load's offset to be affine (1D row sweep)."""
        val = row_indices
        seen: set[str] = set()
        while val is not None and val.name not in seen:
            seen.add(val.name)
            op = self._op_map.get(val.name)
            if op is None:
                return None
            if isinstance(op, Load):
                if op.ptr is None or op.offsets is None or op.row_indices is not None:
                    return None  # only a plain 1D index Load is a valid gather source
                return op
            operands = op.operand_values()
            if len(operands) != 1:  # only peel single-input ops (Cast/Copy/UnaryOp)
                return None
            val = operands[0]
        return None

    def _extract_row_modulus(self, val: TileValue) -> str | None:
        """Detect `(base + make_range) % modulus` row index.

        Returns the modulus expression as a string if matched, else None.
        Used by the cooperative-load emitter to wrap the per-thread row
        index after `_r` is added — `(base + _r) % modulus` is what we
        want, NOT `((base + tid) % modulus) + _r` which is what the
        naive bare-`row_start` path produces (the modulo would wrap in
        the wrong granularity since `tid` runs over the whole TG, not
        just the row dim).
        """
        op = self._op_map.get(val.name)
        if not (isinstance(op, BinOp) and op.op == "mod"):
            return None
        inner_op = self._op_map.get(op.lhs.name)
        if not (isinstance(inner_op, BinOp) and inner_op.op == "add"):
            return None
        l_op = self._op_map.get(inner_op.lhs.name)
        r_op = self._op_map.get(inner_op.rhs.name)
        if not (isinstance(l_op, MakeRange) or isinstance(r_op, MakeRange)):
            return None
        return self._get(op.rhs)

    def _extract_load_bounds(self, op: Load) -> tuple[str | None, str | None]:
        """Extract row and column bound variables from a Load's mask.

        Returns (row_bound_var, col_bound_var), e.g., ("M", "K") for A
        or ("K", "N") for B.  Returns (None, None) if mask can't be parsed.

        Expected mask structure:
            BoolOp(and,
                Compare(expand_dims(row_vals, 1) < ROW_BOUND),   ← shape (M, 1)
                Compare(expand_dims(col_vals, 0) < COL_BOUND))   ← shape (1, N)
        """
        if not op.mask:
            return None, None

        mask_op = self._op_map.get(op.mask.name)
        # Mask is BoolOp(and, ...) or BinOp(and/bitand, ...) depending on source
        is_and = (isinstance(mask_op, BoolOp) or isinstance(mask_op, BinOp)) and mask_op.op in (
            "and",
            "bitand",
        )
        if not is_and:
            if isinstance(mask_op, Compare) and mask_op.op == "lt":
                return self._bound_from_compare(mask_op)
            return None, None

        row_bound, col_bound = None, None

        def visit(node) -> None:
            nonlocal row_bound, col_bound
            if isinstance(node, Compare):
                if node.op == "lt":
                    r, c = self._bound_from_compare(node)
                    if r:
                        row_bound = r
                    if c:
                        col_bound = c
                return
            nested_and = isinstance(node, (BoolOp, BinOp)) and node.op in ("and", "bitand")
            if not nested_and:
                return
            visit(self._op_map.get(node.lhs.name))
            visit(self._op_map.get(node.rhs.name))

        visit(mask_op)

        return row_bound, col_bound

    def _bound_from_compare(self, cmp_op: "Compare") -> tuple[str | None, str | None]:
        """Extract bound variable from Compare(vals < BOUND).

        Returns (row_bound, None) if shape is (M, 1) or
        (None, col_bound) if shape is (1, N).
        """
        if not cmp_op.result or not cmp_op.rhs:
            return None, None
        shape = cmp_op.result.shape
        # Use the expression for the bound (resolves to "M", "K", "73", etc.)
        bound_expr = self._get(cmp_op.rhs)
        if len(shape) == 2 and shape[1] == 1:
            return bound_expr, None  # row bound
        elif len(shape) == 2 and shape[0] == 1:
            return None, bound_expr  # col bound
        return None, None

    # --- Cooperative 2D load: global → shared memory ---

    def _emit_coop_loop_open(self, n_total: int) -> None:
        """Open a threadgroup-striped cooperative loop over `n_total` work items.

        Defines `_idx` (the per-thread work item) in scope and opens exactly one
        brace (caller closes it). When `n_total` is an exact multiple of the
        threadgroup size, every thread runs the SAME number of iterations and
        every `_idx` is in-bounds, so we emit a **constant-trip** loop
        (`for (_it = 0; _it < K; _it++)`) that the compiler fully unrolls with no
        data-dependent predicate. The data-dependent form
        (`for (_idx = tid; _idx < n_total; _idx += threads)`) otherwise guards
        every global load with an `_idx < n_total` predicate — the compiler can't
        prove `tid < threads` — a predication cost ~26% of the kernel at deep
        cache offsets, where the loads dominate.
        """
        threads = self._threads
        if n_total % threads == 0:
            n_iters = n_total // threads
            self._coop_constant_trip = True
            if n_iters == 1:
                # Exactly one work item per thread (n_total == threads): emit a
                # plain scope, not a 1-trip `for`, so there's no loop counter /
                # compare / increment / branch and `_it` folds to 0. Common for
                # the small per-tile dequant B loads.
                self._emit("{")
                self._indent += 1
                self._emit("const ushort _it = 0u;")
                self._emit("uint _idx = tid;")
            else:
                self._emit(f"for (ushort _it = 0; _it < {n_iters}u; _it++) {{")
                self._indent += 1
                self._emit(f"uint _idx = tid + _it * {threads}u;")
        else:
            self._coop_constant_trip = False
            self._emit(f"for (uint _idx = tid; _idx < {n_total}u; _idx += {threads}u) {{")
            self._indent += 1

    def _emit_coop_decode(self, c: int) -> None:
        """Emit the per-thread (row `_r`, vec-col `_cv`) decode for a coop loop.

        In the constant-trip loop `_idx = tid + _it*threads`; when `threads % c == 0`
        the split is exact and DECOUPLES `_it` from the divide:
            _cv = _idx % c = tid % c                    (loop-invariant → hoists)
            _r  = _idx / c = tid/c + _it*(threads/c)    (only the +_it term varies)
        The Metal compiler does NOT strength-reduce the coupled `(tid+_it*threads)/c`
        form, so it keeps the divide/modulo on the load-address dependency chain every
        iteration (~5% of a deep attention kernel, since a low-occupancy kernel can't
        hide that latency). Emitting the decoupled form lets it hoist `tid/c`, `tid%c`.
        Indices are tile-local (< 2^16) so they live in `ushort` — 16-bit register deps
        are ~1.56cyc vs i32's ~1.84cyc, which matters in this low-occupancy regime
        (philipturner/metal-benchmarks). The device address (`_gr = global_row + _r`,
        then `*row_stride + base`) stays uint — global rows and byte offsets exceed 16
        bits.
        """
        if self._coop_constant_trip and self._threads % c == 0:
            self._emit(f"ushort _r = ushort(tid / {c}u + _it * {self._threads // c}u);")
            self._emit(f"ushort _cv = ushort(tid % {c}u);")
        else:
            self._emit(f"uint _r = _idx / {c}u;")
            self._emit(f"uint _cv = _idx % {c}u;")

    def _emit_coop_load(self, op: Load):
        """Generate cooperative vectorized load from global to shared memory."""
        name = op.result.name
        rows, cols = op.result.shape
        # Device-direct: this Load streams straight into the MMA (its only
        # consumer, reuse scope = 1 simdgroup). Emit nothing here — record the
        # device addressing for _emit_mma's simdgroup_load and mark the value.
        if name in self._device_direct_loads:
            ptr_name = op.ptr.name if op.ptr else "???"
            row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
            self._device_operands[name] = (ptr_name, base_offset, row_start, row_stride, col_start)
            self._val_loc[name] = ValLoc(
                "device", ptr_name, row_stride if isinstance(row_stride, int) else 0
            )
            return
        # If a scalar op has touched shmem since the last barrier, threads may
        # still be reading/writing the slot we're about to overwrite.  Emit a
        # barrier to sync all simdgroups before the cooperative writes begin.
        if self._scalar_shmem_dirty:
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._scalar_shmem_dirty = False
            self._pending_tg_barrier = False
            self._buffers_since_barrier.clear()
            self._buffers_read.clear()

        if self._register_resident:
            self._emit_register_load(op)
            return

        plan = self._shmem_plan.get(name)
        if not plan:
            raise ValueError(f"No shmem_plan entry for 2D load result '{name}'")
        buf_name, _, _, stride = plan

        # We're about to WRITE this buffer. Flush if a prior op wrote it
        # (WAW) or read it (WAR — the prior reads must finish before we
        # overwrite). Distinct-buffer writes (Q→_s2 then dO→_s3) skip this
        # and let the consumer absorb both deferred barriers into one.
        self._check_flush_for(writes={buf_name})

        ptr_name = op.ptr.name if op.ptr else "???"

        # Per-buffer shmem dtype overrides the kernel-global default for
        # this load's storage. Lets bf16 inputs stay in bf16 shmem under
        # HIGH_PRECISION=1 if the planner determined the MMA pair partner is
        # also bf16-compatible (saves 50% per such buffer).
        sdt = self._buf_shmem_dtype(buf_name)

        # Use address callback if provided (for internal attention path)
        addr_cb = self._config.get(f"addr_{name}")
        if addr_cb:
            vec_width = self._vec_width
            while (
                vec_width > 2
                and rows * (cols // vec_width) < self._threads
                and cols % (vec_width // 2) == 0
            ):
                vec_width //= 2
            vec_type = f"{sdt}{vec_width}"
            n_vecs = rows * (cols // vec_width)
            self._emit(f"// Cooperative load {name} [{rows}x{cols}] → {buf_name}")
            self._emit_coop_loop_open(n_vecs)
            self._emit(f"uint _r = _idx / {cols // vec_width}u;")
            self._emit(f"uint _cv = _idx % {cols // vec_width}u;")
            addr_cb(self, "_r", "_cv")
            self._indent -= 1
            self._emit("}")
            self._val_loc[name] = ValLoc("shared", buf_name, stride)
            self._pending_tg_barrier = True
            self._buffers_since_barrier.add(buf_name)
            return

        row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
        addr_base = f"({base_offset}) + " if base_offset != "0" else ""

        # Inside a ForLoop: if offsets are a carried variable, add loop variable
        # to the K dimension. Disambiguate by checking which dimension starts
        # at 0 (rk = arange(0, BLOCK_K) → base 0). When both dims have the
        # same size (BLOCK_K == BLOCK_N), the "starts at 0" check is decisive.
        if self._loop_var and op.offsets and op.offsets.name in self._carried_inits:
            carried_inc = self._carried_increments.get(op.offsets.name)
            if carried_inc is not None and carried_inc != self._loop_step:
                # Packed layout: carried increment (e.g. BN*BK) differs from loop
                # step (BK). Express tile advancement as a row offset.
                rows_per_iter = carried_inc // row_stride
                row_start = f"({self._loop_var} / {self._loop_step}u * {rows_per_iter}u)"
            elif col_start == "0" and row_start != "0":
                col_start = f"(0 + {self._loop_var})"
            elif row_start == "0" and col_start != "0":
                row_start = f"(0 + {self._loop_var})"
            elif cols == self._loop_step:
                col_start = f"({col_start} + {self._loop_var})"
            elif rows == self._loop_step:
                row_start = f"({row_start} + {self._loop_var})"

        # Use plan cols (BLOCK_N when column tiling, N otherwise)
        load_cols = self._eff_cols(cols)
        # Shrink vec_width when the tile has fewer vec-loads than threads, so
        # every thread participates in the load. Keep >= 2 — scalar vec=1
        # breaks the transposed-store path which uses _val[j] indexing.
        vec_width = self._vec_width
        while (
            vec_width > 2
            and rows * (load_cols // vec_width) < self._threads
            and load_cols % (vec_width // 2) == 0
        ):
            vec_width //= 2
        # Use the per-buffer shmem dtype for the cooperative-load vec type
        vec_type = f"{sdt}{vec_width}"
        n_vecs = rows * (load_cols // vec_width)

        # Column offset for tiled loads
        col_off = f" + {self._col_offset}" if self._col_offset else ""

        self._emit(
            f"// {'Async' if self._async_copy else 'Cooperative'} load {name} [{rows}x{load_cols}] → {buf_name}"
        )

        if self._async_copy:
            # Element size for async copy: sizeof/alignof the shmem element type
            _elem_sz = "2" if self._shmem_dtype == "half" else "4"
            # Per-simdgroup DMA: split the tile across simdgroups so each
            # initiates its own DMA transfer in parallel.
            db_off = ""
            if self._db_shmem_offsets:
                db_off_expr = self._db_shmem_offsets.get(buf_name, "")
                if db_off_expr:
                    db_off = f"{db_off_expr} + "

            # Single SIMD group does the async copy — faster than distributing
            # across multiple groups (integer instruction overhead for address
            # computation eclipses the parallel DMA benefit).
            self._emit(f"thread _simdgroup_event_t* _ev_{name} = nullptr;")
            src_offset = (
                f"{addr_base}uint({row_start}) * {row_stride}u + uint({col_start}){col_off}"
            )
            self._emit("if (simd_gid == 0) {")
            self._indent += 1
            self._emit(
                f"_ev_{name} = _alloy_async_copy_2d("
                f"{_elem_sz}, {_elem_sz}, "
                f"(threadgroup void*)(&{buf_name}[{db_off}0]), {stride}u, 1, "
                f"ulong2({load_cols}u, {rows}u), "
                f"(const device void*)({ptr_name} + {src_offset}), "
                f"{row_stride}u, 1, "
                f"ulong2({load_cols}u, {rows}u), long2(0, 0), 0);"
            )
            self._indent -= 1
            self._emit("}")
            if not self._db_shmem_offsets:
                self._pending_async_events.append(name)
            self._val_loc[name] = ValLoc("shared", buf_name, stride)

            # Apply prologue transform after async copy completes.
            # Wait for the copy, barrier, then run a cooperative elementwise
            # pass over shmem to apply the transform in-place.
            if op.transform:
                self._flush_async_events()
                self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                n_elems = rows * load_cols
                db_off_expr = (
                    self._db_shmem_offsets.get(buf_name, "") if self._db_shmem_offsets else ""
                )
                db_off = f"{db_off_expr} + " if db_off_expr else ""
                self._emit(f"for (uint _ti = tid; _ti < {n_elems}u; _ti += {self._threads}u) {{")
                self._indent += 1
                self._emit(f"uint _tr = _ti / {load_cols}u;")
                self._emit(f"uint _tc = _ti % {load_cols}u;")
                self._emit(f"uint _tg_r = uint({row_start}) + _tr;")
                self._emit(f"uint _tg_c = uint({col_start}){col_off} + _tc;")
                store_offs = f"_tg_r * {row_stride}u + _tg_c"
                xf_expr = self._eval_transform(
                    op.transform,
                    f"{buf_name}[{db_off}_tr * {stride}u + _tc]",
                    store_offs=store_offs,
                    extra_transforms=op.transform_extras,
                    chain_source_name=op.transform_source_name,
                )
                self._emit(f"{buf_name}[{db_off}_tr * {stride}u + _tc] = {xf_expr};")
                self._indent -= 1
                self._emit("}")
                self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

            return

        # Q4_K: vec4/vec8 decode from 144-byte super-blocks (2B d + 2B dmin +
        # 12B packed 6-bit scales/mins + 128B interleaved nibbles). Address
        # contract from the kernel side is `B + row * (N_GROUPS * 144) + k`
        # where k is the LOGICAL column. `vw` (<=8) contiguous K positions from
        # a vw-aligned start share one super-block, sub-block, scale/min and
        # nibble half — one block-header lookup + vw contiguous nibble bytes
        # cover them. weight = d*scale*nibble - dmin*min (get_scale_min_k4).
        if op.dequant_format == "q4_k":
            packed_col_start = f"(0 + {self._loop_var})" if self._loop_var else col_start
            row_bound_var, col_bound_var = self._extract_load_bounds(op)
            ce = self.func.constexpr_values

            def _resolve_bound_q4k(bvar):
                if bvar is None:
                    return None
                v = ce.get(bvar)
                if isinstance(v, int):
                    return v
                try:
                    return int(bvar)
                except (ValueError, TypeError):
                    return None

            rb_val = _resolve_bound_q4k(row_bound_var)
            if rb_val is not None and rb_val % rows == 0:
                row_bound_var = None
            cb_val = _resolve_bound_q4k(col_bound_var)
            cb_vec4_safe = cb_val is not None and cb_val % 4 == 0
            cb_vec8_safe = cb_val is not None and cb_val % 8 == 0
            if cb_val is not None and cb_val % load_cols == 0:
                col_bound_var = None

            db_off = ""
            if self._db_shmem_offsets:
                db_off_expr = self._db_shmem_offsets.get(buf_name, "")
                if db_off_expr:
                    db_off = f"{db_off_expr} + "

            def _emit_q4k_header():
                self._emit("uint _g_q4k = _gc / 256u;")
                self._emit("uint _ib_q4k = _gc % 256u;")
                self._emit("uint _sg_q4k = _ib_q4k / 32u;")
                self._emit("uint _pos_q4k = _ib_q4k % 32u;")
                self._emit(
                    f"device const uchar* _blk_q4k = (device const uchar*){ptr_name}"
                    f" + {addr_base}_gr * {row_stride}u + _g_q4k * 144u;"
                )
                self._emit("ushort _d_bits_q4k = ushort(uint(_blk_q4k[0]) | (uint(_blk_q4k[1]) << 8));")
                self._emit("ushort _dm_bits_q4k = ushort(uint(_blk_q4k[2]) | (uint(_blk_q4k[3]) << 8));")
                self._emit("float _d_q4k = float(as_type<half>(_d_bits_q4k));")
                self._emit("float _dmin_q4k = float(as_type<half>(_dm_bits_q4k));")
                self._emit("device const uchar* _scb_q4k = _blk_q4k + 4u;")
                self._emit("uint _jm4_q4k = (_sg_q4k >= 4u) ? (_sg_q4k - 4u) : 0u;")
                self._emit("int _qj_q4k = int(_scb_q4k[_sg_q4k]);")
                self._emit("int _qj4_q4k = int(_scb_q4k[_sg_q4k + 4u]);")
                self._emit("int _qjm4_q4k = int(_scb_q4k[_jm4_q4k]);")
                self._emit(
                    "int _sc_q4k = (_sg_q4k < 4u) ? (_qj_q4k & 63) : "
                    "((_qj4_q4k & 0xF) | ((_qjm4_q4k >> 6) << 4));"
                )
                self._emit(
                    "int _mn_q4k = (_sg_q4k < 4u) ? (_qj4_q4k & 63) : "
                    "((_qj4_q4k >> 4) | ((_qj_q4k >> 6) << 4));"
                )

            sdt_q4k = self._buf_shmem_dtype(buf_name)
            # vw=16 decodes the Q4_K header once per 16 values, halving the
            # per-K-iter decode latency. Only when 16 values/thread still fill
            # the threadgroup.
            use_vec16 = (
                load_cols % 16 == 0
                and sdt_q4k == "half"
                and col_bound_var is None
                and rows * (load_cols // 16) >= self._threads
            )
            if use_vec16:
                fill = float(op.other)
                n_vec = rows * (load_cols // 16)
                self._emit_coop_loop_open(n_vec)
                self._emit(f"uint _r = _idx / {load_cols // 16}u;")
                self._emit(f"uint _cv = _idx % {load_cols // 16}u;")
                self._emit("uint _c = _cv * 16u;")
                self._emit(f"uint _gr = uint({row_start}) + _r;")
                self._emit(f"uint _gc = uint({packed_col_start}){col_off} + _c;")
                for g in range(4):
                    self._emit(f"half4 _val{g}_q4k = half4({fill}h);")
                guard = f"_gr < {row_bound_var}" if row_bound_var else None
                if guard:
                    self._emit(f"if ({guard}) {{")
                    self._indent += 1
                _emit_q4k_header()
                self._emit("uint _qoff_q4k = 16u + (_sg_q4k / 2u) * 32u + _pos_q4k;")
                self._emit("uchar _sh_q4k = uchar((_sg_q4k & 1u) * 4u);")
                self._emit("half _dsc_q4k = as_type<half>(_d_bits_q4k) * half(_sc_q4k);")
                self._emit("half _dm_q4k = as_type<half>(_dm_bits_q4k) * half(_mn_q4k);")
                for g in range(4):
                    self._emit(
                        f"uchar4 _qb{g}_q4k = *(device const uchar4*)(_blk_q4k + _qoff_q4k + {g * 4}u);"
                    )
                    self._emit(
                        f"ushort4 _hb{g}_q4k = ushort4(0x6400) | "
                        f"ushort4((_qb{g}_q4k >> _sh_q4k) & uchar4(0x0F));"
                    )
                    self._emit(
                        f"_val{g}_q4k = (as_type<half4>(_hb{g}_q4k) - 1024.0h) * _dsc_q4k - _dm_q4k;"
                    )
                if guard:
                    self._indent -= 1
                    self._emit("}")
                for g in range(4):
                    self._emit(
                        f"*(threadgroup half4*)(&{buf_name}[{db_off}_r * {stride}u + _c + {g * 4}u]) = _val{g}_q4k;"
                    )
                self._indent -= 1
                self._emit("}")
                self._val_loc[name] = ValLoc("shared", buf_name, stride)
                self._pending_tg_barrier = True
                self._buffers_since_barrier.add(buf_name)
                return

            use_vec8 = load_cols % 8 == 0 and (col_bound_var is None or cb_vec8_safe)
            use_vec4 = load_cols % 4 == 0 and (col_bound_var is None or cb_vec4_safe)
            if use_vec8 or use_vec4:
                vw = 8 if use_vec8 else 4
                sdt = self._buf_shmem_dtype(buf_name)
                vec_t = f"{sdt}4"
                n_vec = rows * (load_cols // vw)
                self._emit_coop_loop_open(n_vec)
                self._emit(f"uint _r = _idx / {load_cols // vw}u;")
                self._emit(f"uint _cv = _idx % {load_cols // vw}u;")
                self._emit(f"uint _c = _cv * {vw}u;")
                self._emit(f"uint _gr = uint({row_start}) + _r;")
                self._emit(f"uint _gc = uint({packed_col_start}){col_off} + _c;")
                fill = float(op.other)
                lit = f"{fill}h" if sdt == "half" else f"{fill}f"
                self._emit(f"{vec_t} _val_lo = {vec_t}({lit});")
                if vw == 8:
                    self._emit(f"{vec_t} _val_hi = {vec_t}({lit});")
                guards = []
                if row_bound_var:
                    guards.append(f"_gr < {row_bound_var}")
                if col_bound_var:
                    guards.append(f"_gc + {vw - 1}u < {col_bound_var}")
                if guards:
                    self._emit(f"if ({' && '.join(guards)}) {{")
                    self._indent += 1
                _emit_q4k_header()
                self._emit("uint _qoff_q4k = 16u + (_sg_q4k / 2u) * 32u + _pos_q4k;")
                self._emit("uchar _sh_q4k = uchar((_sg_q4k & 1u) * 4u);")
                self._emit("uchar4 _qb_lo_q4k = *(device const uchar4*)(_blk_q4k + _qoff_q4k);")
                if vw == 8:
                    self._emit("uchar4 _qb_hi_q4k = *(device const uchar4*)(_blk_q4k + _qoff_q4k + 4u);")
                if sdt == "half":
                    # Bit-trick int4->half: 0x6400|nib == 1024.0h+nib (exact, nib<16),
                    # subtract 1024 before scaling (f16-exact). Kills the int->float
                    # AND float->half CONVERTs (the q4_k matmul's F32-limiter).
                    self._emit("half _dsc_q4k = as_type<half>(_d_bits_q4k) * half(_sc_q4k);")
                    self._emit("half _dm_q4k = as_type<half>(_dm_bits_q4k) * half(_mn_q4k);")
                    self._emit(
                        "ushort4 _hb_lo_q4k = ushort4(0x6400) | "
                        "ushort4((_qb_lo_q4k >> _sh_q4k) & uchar4(0x0F));"
                    )
                    self._emit("_val_lo = (as_type<half4>(_hb_lo_q4k) - 1024.0h) * _dsc_q4k - _dm_q4k;")
                    if vw == 8:
                        self._emit(
                            "ushort4 _hb_hi_q4k = ushort4(0x6400) | "
                            "ushort4((_qb_hi_q4k >> _sh_q4k) & uchar4(0x0F));"
                        )
                        self._emit("_val_hi = (as_type<half4>(_hb_hi_q4k) - 1024.0h) * _dsc_q4k - _dm_q4k;")
                else:
                    self._emit("float _dsc_q4k = _d_q4k * float(_sc_q4k);")
                    self._emit("float _dm_q4k = _dmin_q4k * float(_mn_q4k);")
                    self._emit("int4 _nib_lo_q4k = int4((_qb_lo_q4k >> _sh_q4k) & uchar4(0x0F));")
                    self._emit("_val_lo = float4(_nib_lo_q4k) * _dsc_q4k - _dm_q4k;")
                    if vw == 8:
                        self._emit("int4 _nib_hi_q4k = int4((_qb_hi_q4k >> _sh_q4k) & uchar4(0x0F));")
                        self._emit("_val_hi = float4(_nib_hi_q4k) * _dsc_q4k - _dm_q4k;")
                if guards:
                    self._indent -= 1
                    self._emit("}")
                self._emit(
                    f"*(threadgroup {vec_t}*)(&{buf_name}[{db_off}_r * {stride}u + _c]) = _val_lo;"
                )
                if vw == 8:
                    self._emit(
                        f"*(threadgroup {vec_t}*)(&{buf_name}[{db_off}_r * {stride}u + _c + 4u]) = _val_hi;"
                    )
                self._indent -= 1
                self._emit("}")
                self._val_loc[name] = ValLoc("shared", buf_name, stride)
                self._pending_tg_barrier = True
                self._buffers_since_barrier.add(buf_name)
                return

            # Scalar fallback for unaligned tile widths.
            n_elems = rows * load_cols
            self._emit_coop_loop_open(n_elems)
            self._emit(f"uint _r = _idx / {load_cols}u;")
            self._emit(f"uint _c = _idx % {load_cols}u;")
            self._emit(f"uint _gr = uint({row_start}) + _r;")
            self._emit(f"uint _gc = uint({packed_col_start}){col_off} + _c;")
            guards = []
            if row_bound_var:
                guards.append(f"_gr < {row_bound_var}")
            if col_bound_var:
                guards.append(f"_gc < {col_bound_var}")
            self._emit(f"float _val = {float(op.other)}f;")
            if guards:
                self._emit(f"if ({' && '.join(guards)}) {{")
                self._indent += 1
            _emit_q4k_header()
            self._emit("uint _qoff_q4k = 16u + (_sg_q4k / 2u) * 32u + _pos_q4k;")
            self._emit("uint _sh_q4k = (_sg_q4k & 1u) * 4u;")
            self._emit("int _nib_q4k = (int(_blk_q4k[_qoff_q4k]) >> _sh_q4k) & 0x0F;")
            self._emit("_val = _d_q4k * float(_sc_q4k) * float(_nib_q4k) - _dmin_q4k * float(_mn_q4k);")
            if guards:
                self._indent -= 1
                self._emit("}")
            self._emit(
                f"{buf_name}[{db_off}_r * {stride}u + _c] = {self._shmem_cast('_val', buf_name)};"
            )
            self._indent -= 1
            self._emit("}")
            self._val_loc[name] = ValLoc("shared", buf_name, stride)
            self._pending_tg_barrier = True
            self._buffers_since_barrier.add(buf_name)
            return

        # Q6_K: vec4 decode from 210-byte super-blocks (128 QL + 64 QH +
        # 16 sub-scales + fp16 d). Address contract from the kernel side is
        # `B + row * (N_GROUPS * 210) + k` where k is the LOGICAL column.
        # 4 contiguous K positions within a subgroup share QL/QH byte
        # access patterns (4 contiguous bytes each), sub-scale (1 per 16
        # K), and d (1 per 256 K) — mirrors dot_q6_k_v2's matvec
        # strategy.
        if op.dequant_format == "q6_k":
            packed_col_start = f"(0 + {self._loop_var})" if self._loop_var else col_start
            row_bound_var, col_bound_var = self._extract_load_bounds(op)
            ce = self.func.constexpr_values

            def _resolve_bound(bvar):
                if bvar is None:
                    return None
                v = ce.get(bvar)
                if isinstance(v, int):
                    return v
                try:
                    return int(bvar)
                except (ValueError, TypeError):
                    return None

            rb_val = _resolve_bound(row_bound_var)
            if rb_val is not None and rb_val % rows == 0:
                row_bound_var = None
            cb_val = _resolve_bound(col_bound_var)
            cb_vec4_safe = cb_val is not None and cb_val % 4 == 0
            cb_vec8_safe = cb_val is not None and cb_val % 8 == 0
            if cb_val is not None and cb_val % load_cols == 0:
                col_bound_var = None

            db_off = ""
            if self._db_shmem_offsets:
                db_off_expr = self._db_shmem_offsets.get(buf_name, "")
                if db_off_expr:
                    db_off = f"{db_off_expr} + "

            # Vec8 path: 8 contiguous K positions per thread, expressed as
            # 2× vec4 (Metal reserves the `float8`/`uchar8`/`int8` names).
            # Within an 8-aligned start, all 8 positions share the same Q6_K
            # subgroup (32-wide), nibble-shift class, AND sub-scale group
            # (16-wide), so one (scale, d) lookup + 2 (QL,QH) byte fetches
            # of 4 bytes each cover 8 outputs. Matches the amortization
            # dot_q6_k_v2's matvec gets.
            use_vec8 = load_cols % 8 == 0 and (col_bound_var is None or cb_vec8_safe)
            use_vec4 = load_cols % 4 == 0 and (col_bound_var is None or cb_vec4_safe)
            if use_vec8 or use_vec4:
                vw = 8 if use_vec8 else 4
                n_vec = rows * (load_cols // vw)
                self._emit_coop_loop_open(n_vec)
                self._emit(f"uint _r = _idx / {load_cols // vw}u;")
                self._emit(f"uint _cv = _idx % {load_cols // vw}u;")
                self._emit(f"uint _c = _cv * {vw}u;")
                self._emit(f"uint _gr = uint({row_start}) + _r;")
                self._emit(f"uint _gc = uint({packed_col_start}){col_off} + _c;")
                fill = float(op.other)
                sdt = self._buf_shmem_dtype(buf_name)
                vec_t = f"{sdt}4"
                self._emit(f"{vec_t} _val_lo = {vec_t}({fill}f);")
                if vw == 8:
                    self._emit(f"{vec_t} _val_hi = {vec_t}({fill}f);")
                guards = []
                if row_bound_var:
                    guards.append(f"_gr < {row_bound_var}")
                if col_bound_var:
                    guards.append(f"_gc + {vw - 1}u < {col_bound_var}")
                if guards:
                    self._emit(f"if ({' && '.join(guards)}) {{")
                    self._indent += 1
                self._emit("uint _g_q6k = _gc / 256u;")
                self._emit("uint _ib_q6k = _gc % 256u;")
                self._emit("uint _sg_q6k = _ib_q6k / 32u;")
                self._emit("uint _pos_q6k = _ib_q6k % 32u;")
                self._emit(
                    f"device const uchar* _blk_q6k = (device const uchar*){ptr_name}"
                    f" + {addr_base}_gr * {row_stride}u + _g_q6k * 210u;"
                )
                # QL bytes — `vw` contiguous, one per element, same nibble half.
                self._emit(
                    "uint _ql_off = (_sg_q6k / 4u) * 64u + (_sg_q6k & 1u) * 32u + _pos_q6k;"
                )
                self._emit("uchar _ql_shift = ((_sg_q6k & 3u) / 2u) * 4u;")
                self._emit(
                    "uchar4 _ql_v_lo = *(device const uchar4*)(_blk_q6k + _ql_off);"
                )
                self._emit(
                    "int4 _ql_n_lo = int4((_ql_v_lo >> _ql_shift) & uchar4(0x0F));"
                )
                if vw == 8:
                    self._emit(
                        "uchar4 _ql_v_hi = *(device const uchar4*)(_blk_q6k + _ql_off + 4u);"
                    )
                    self._emit(
                        "int4 _ql_n_hi = int4((_ql_v_hi >> _ql_shift) & uchar4(0x0F));"
                    )
                # QH bytes — `vw` contiguous, same shift extracts each element's 2 bits.
                self._emit(
                    "uint _qh_off = 128u + (_sg_q6k / 4u) * 32u + _pos_q6k;"
                )
                self._emit("uchar _qh_shift = (_sg_q6k & 3u) * 2u;")
                self._emit(
                    "uchar4 _qh_v_lo = *(device const uchar4*)(_blk_q6k + _qh_off);"
                )
                self._emit(
                    "int4 _qh_n_lo = int4((_qh_v_lo >> _qh_shift) & uchar4(0x03));"
                )
                if vw == 8:
                    self._emit(
                        "uchar4 _qh_v_hi = *(device const uchar4*)(_blk_q6k + _qh_off + 4u);"
                    )
                    self._emit(
                        "int4 _qh_n_hi = int4((_qh_v_hi >> _qh_shift) & uchar4(0x03));"
                    )
                # Sub-scale: 1 byte per 16 K; `vw` ≤ 8 from a vw-aligned
                # start stays within one scale group.
                self._emit("uint _sc_off = 192u + _ib_q6k / 16u;")
                self._emit("int _sc_raw = int(_blk_q6k[_sc_off]);")
                # signed int8 scale -> float bit-trick (raw^0x80 == scale+128).
                self._emit("float _sc_f = as_type<float>(((_sc_raw ^ 0x80) | 0x4B000000)) - 8388736.0;")
                self._emit(
                    "ushort _d_bits = ushort(uint(_blk_q6k[208]) | (uint(_blk_q6k[209]) << 8));"
                )
                if sdt == "half":
                    # u6->half bit-trick: 0x6400|u == 1024.0h+u (exact, u<64); the
                    # -1056 (= 1024 + 32) recovers q = u - 32. Kills int->float AND
                    # float->half CONVERTs (matches the q4_k path).
                    self._emit("half _d_sc = as_type<half>(_d_bits) * half(_sc_f);")
                    self._emit(
                        "_val_lo = (as_type<half4>(ushort4(0x6400) | "
                        "ushort4(_ql_n_lo | (_qh_n_lo << 4))) - 1056.0h) * _d_sc;"
                    )
                    if vw == 8:
                        self._emit(
                            "_val_hi = (as_type<half4>(ushort4(0x6400) | "
                            "ushort4(_ql_n_hi | (_qh_n_hi << 4))) - 1056.0h) * _d_sc;"
                        )
                else:
                    # int->float bit-trick: 0x4B000000|u == 8388608.0f+u; the
                    # -8388640 (= 8388608 + 32) recovers q = u - 32.
                    self._emit("int4 _q_lo = (_ql_n_lo | (_qh_n_lo << 4)) | int4(0x4B000000);")
                    if vw == 8:
                        self._emit("int4 _q_hi = (_ql_n_hi | (_qh_n_hi << 4)) | int4(0x4B000000);")
                    self._emit("float _d_q6k = float(as_type<half>(_d_bits));")
                    self._emit("float _d_sc = _d_q6k * _sc_f;")
                    self._emit("_val_lo = (as_type<float4>(_q_lo) - 8388640.0) * _d_sc;")
                    if vw == 8:
                        self._emit("_val_hi = (as_type<float4>(_q_hi) - 8388640.0) * _d_sc;")
                if guards:
                    self._indent -= 1
                    self._emit("}")
                self._emit(
                    f"*(threadgroup {vec_t}*)(&{buf_name}[{db_off}_r * {stride}u + _c]) = _val_lo;"
                )
                if vw == 8:
                    self._emit(
                        f"*(threadgroup {vec_t}*)(&{buf_name}[{db_off}_r * {stride}u + _c + 4u]) = _val_hi;"
                    )
                self._indent -= 1
                self._emit("}")
                self._val_loc[name] = ValLoc("shared", buf_name, stride)
                self._pending_tg_barrier = True
                self._buffers_since_barrier.add(buf_name)
                return

            # Scalar fallback for unaligned tile widths.
            n_elems = rows * load_cols
            self._emit_coop_loop_open(n_elems)
            self._emit(f"uint _r = _idx / {load_cols}u;")
            self._emit(f"uint _c = _idx % {load_cols}u;")
            self._emit(f"uint _gr = uint({row_start}) + _r;")
            self._emit(f"uint _gc = uint({packed_col_start}){col_off} + _c;")
            guards = []
            if row_bound_var:
                guards.append(f"_gr < {row_bound_var}")
            if col_bound_var:
                guards.append(f"_gc < {col_bound_var}")
            self._emit(f"float _val = {float(op.other)}f;")
            if guards:
                self._emit(f"if ({' && '.join(guards)}) {{")
                self._indent += 1
            self._emit("uint _g_q6k = _gc / 256u;")
            self._emit("uint _ib_q6k = _gc % 256u;")
            self._emit("uint _sg_q6k = _ib_q6k / 32u;")
            self._emit("uint _pos_q6k = _ib_q6k % 32u;")
            self._emit(
                f"device const uchar* _blk_q6k = (device const uchar*){ptr_name}"
                f" + {addr_base}_gr * {row_stride}u + _g_q6k * 210u;"
            )
            self._emit(
                "uint _ql_off = (_sg_q6k / 4u) * 64u + (_sg_q6k & 1u) * 32u + _pos_q6k;"
            )
            self._emit("uint _ql_shift = ((_sg_q6k & 3u) / 2u) * 4u;")
            self._emit("int _ql_q6k = (int(_blk_q6k[_ql_off]) >> _ql_shift) & 0x0F;")
            self._emit(
                "uint _qh_off = 128u + (_sg_q6k / 4u) * 32u + _pos_q6k;"
            )
            self._emit("uint _qh_shift = (_sg_q6k & 3u) * 2u;")
            self._emit("int _qh_q6k = (int(_blk_q6k[_qh_off]) >> _qh_shift) & 0x03;")
            self._emit("int _q_q6k = (_ql_q6k | (_qh_q6k << 4)) | 0x4B000000;")  # int->float bit-trick
            self._emit("uint _sc_off = 192u + _ib_q6k / 16u;")
            self._emit("int _sc_raw = int(_blk_q6k[_sc_off]);")
            self._emit("float _sc_f = as_type<float>(((_sc_raw ^ 0x80) | 0x4B000000)) - 8388736.0;")
            self._emit(
                "ushort _d_bits = ushort(uint(_blk_q6k[208]) | (uint(_blk_q6k[209]) << 8));"
            )
            self._emit("float _d_q6k = float(as_type<half>(_d_bits));")
            self._emit("_val = _d_q6k * _sc_f * (as_type<float>(_q_q6k) - 8388640.0);")
            if guards:
                self._indent -= 1
                self._emit("}")
            self._emit(
                f"{buf_name}[{db_off}_r * {stride}u + _c] = {self._shmem_cast('_val', buf_name)};"
            )
            self._indent -= 1
            self._emit("}")
            self._val_loc[name] = ValLoc("shared", buf_name, stride)
            self._pending_tg_barrier = True
            self._buffers_since_barrier.add(buf_name)
            return

        # Packed format: vectorized uint load + nibble extraction
        # For INT4 (pack_factor=2, bits=4): load 4 bytes (uint) = 8 nibbles per thread.
        if op.pack_factor > 0:
            pf = op.pack_factor
            bits = op.pack_bits
            mask_val = (1 << bits) - 1
            elems_per_uint = 4 * pf  # 4 bytes × pf elements/byte (INT4: 8 elems/uint)
            packed_col_start = f"(0 + {self._loop_var})" if self._loop_var else "0"
            # Number of uint loads needed to fill the tile
            n_packed_bytes = rows * (load_cols // pf)  # total packed bytes
            n_uint_loads = (n_packed_bytes + 3) // 4  # round up to uint granularity
            packed_stride = row_stride
            packed_cols = load_cols // pf  # packed bytes per row

            row_bound_var, col_bound_var = self._extract_load_bounds(op)
            # Elide bounds checks that the tile shape makes provably-true, exactly
            # as the q6_k path does: when N % rows == 0 every `_gr` is in range,
            # and when K % load_cols == 0 the last K-tile never overhangs so the
            # per-nibble `if (_gc >= K)` is dead. Both otherwise emit a predicate
            # the compiler can't prove uniform (predication) AND the live col
            # bound blocks the vectorized dequant below.
            _q4_ce = self.func.constexpr_values

            def _resolve_q4_bound(bvar):
                if bvar is None:
                    return None
                v = _q4_ce.get(bvar)
                if isinstance(v, int):
                    return v
                try:
                    return int(bvar)
                except (ValueError, TypeError):
                    return None

            _q4_rb = _resolve_q4_bound(row_bound_var)
            if _q4_rb is not None and _q4_rb % rows == 0:
                row_bound_var = None
            _q4_cb = _resolve_q4_bound(col_bound_var)
            if _q4_cb is not None and _q4_cb % load_cols == 0:
                col_bound_var = None
            fill = f"{float(op.other)}f"

            # Dequant params (resolved once, used per-element)
            has_dequant = op.dequant_scale_ptr is not None and op.dequant_n_groups > 0
            hoist_dequant = False
            if has_dequant:
                scale_name = op.dequant_scale_ptr.name
                bias_name = op.dequant_bias_ptr.name if op.dequant_bias_ptr is not None else ""
                zero_point = op.dequant_zero_point
                n_groups = op.dequant_n_groups
                ce = self.func.constexpr_values
                group_size = ce.get("GROUP_SIZE", 1)
                block_k = ce.get("BLOCK_K", 0)
                # All `elems_per_uint` nibbles of one packed uint land in a
                # SINGLE quant group when the group tiles the uint's column span
                # evenly (group_size a multiple of elems_per_uint) and that span
                # is group-aligned — k is a BK multiple and `_pb*pf` is
                # elems-aligned from the 4-byte-per-thread tiling (`packed_cols %
                # 4 == 0`). Then `_group`, `_scale`, `_bias` are invariant across
                # the nibble loop: load them ONCE per uint instead of per nibble.
                # Otherwise each of the 8 nibbles re-loads the same scale+bias
                # from device, and that dependent load latency lands ~10% on the
                # dequant `*scale` / `+bias` lines at depth.
                hoist_dequant = (
                    elems_per_uint <= group_size
                    and group_size % elems_per_uint == 0
                    and packed_cols % 4 == 0
                    and block_k % elems_per_uint == 0
                )

            db_off = ""
            if self._db_shmem_offsets:
                db_off_expr = self._db_shmem_offsets.get(buf_name, "")
                if db_off_expr:
                    db_off = f"{db_off_expr} + "

            # End-to-end-f16 vectorized dequant decision (needed before the hoist
            # declaration so scale/bias get typed `half`). On Apple GPU `float4`
            # vectorization is a no-op for ALU throughput — the ALU is scalar
            # regardless of SIMD width (metal-benchmarks). The lever is f16: half
            # FMA is ~28% faster than f32, scale/bias are already f16 in device
            # memory (no f16->f32 load convert), and storing half directly removes
            # the float->half convert — one of two 4-cycle converts per weight.
            # Same aligned / full-tile / unsigned-nibble gate as the predication
            # elision; ~1 f16-ULP/weight vs float.
            _signed_byte = self._buffer_dtypes.get(ptr_name, self._shmem_dtype) == "char" and bits == 8
            out_dt = self._buf_shmem_dtype(buf_name)
            can_vectorize = (
                has_dequant
                and hoist_dequant
                and op.dequant_high_ptr is None
                and not op.transform
                and col_bound_var is None
                and out_dt == "half"
                and not db_off
                and elems_per_uint % 4 == 0
                and load_cols % elems_per_uint == 0
                and stride % 4 == 0
            )
            dequant_reg_dt = "half" if can_vectorize else "float"
            # Bit-trick int4->half: build the half DIRECTLY from the 4-bit value
            # as the bit pattern 0x6400|v == 1024.0h + v (exact for v < 1024, the
            # mantissa ULP at exponent 2^10 is 1.0), instead of `half4(uint4)`
            # which Apple lowers through the F32 CONVERT pipe (~4cyc/weight). On
            # the q4_k matmuls that convert is the bottleneck — F32 limiter ~97%
            # while the matmul matrix-pipe sits at ~3% (starved on dequant). The
            # +1024 offset folds into the per-group bias (`bias - 1024*scale`),
            # so the FMA is unchanged. q4_k only (bits==4, zero-point 0, biased).
            use_bittrick = (
                can_vectorize
                and bits == 4
                and op.dequant_bias_ptr is not None
                and float(zero_point) == 0.0
            )

            self._emit_coop_loop_open(n_uint_loads)
            # Map linear index to (row, packed_byte_col) within tile. These are
            # tile-local — `_flat_byte < rows*packed_cols`, `_r < rows`, `_pb <
            # packed_cols` — so they live in ushort when the tile fits 16 bits
            # (16-bit register deps are cheaper; the device row/col `_gr`/`_gc_base`
            # stay uint since they scale with N/K).
            _ix = "ushort" if (rows * packed_cols) <= 0xFFFF else "uint"
            self._emit(f"{_ix} _flat_byte = {_ix}(_idx * 4u);")
            self._emit(f"{_ix} _r = {_ix}(_flat_byte / {packed_cols}u);")
            self._emit(f"{_ix} _pb = {_ix}(_flat_byte % {packed_cols}u);")  # packed byte col
            self._emit(f"uint _gr = uint({row_start}) + _r;")
            self._emit(
                f"uint _gc_base = uint({packed_col_start}){col_off} + _pb * {pf}u;"
            )  # unpacked col

            # Bounds check on row
            bounds_row = f"_gr < {row_bound_var}" if row_bound_var else None

            # Load 4 packed bytes as uint
            self._emit("uint _packed = 0u;")
            if hoist_dequant:
                self._emit(f"{dequant_reg_dt} _hscale = 0.0f;")
                if op.dequant_bias_ptr is not None:
                    self._emit(f"{dequant_reg_dt} _hbias = 0.0f;")
            row_guard = f"_r < {rows}u" + (f" && {bounds_row}" if bounds_row else "")
            self._emit(f"if ({row_guard}) {{")
            self._indent += 1
            # _pb is byte offset within tile; actual device byte offset includes K-loop offset
            self._emit(
                f"uint _dev_pb = uint({packed_col_start}) / {pf}u + _pb;"
            )  # absolute packed byte col
            self._emit(f"uint _bytes_left = min(4u, {packed_cols}u - _pb);")
            self._emit(
                f"device const uchar* _src = (device const uchar*)({ptr_name} + {addr_base}_gr * {packed_stride}u + _dev_pb);"
            )
            # Load up to 4 bytes; use scalar loads to avoid alignment issues on edge
            self._emit("if (_bytes_left >= 4u) _packed = *(device const uint*)_src;")
            self._emit(
                "else { for (uint _b = 0; _b < _bytes_left; _b++) _packed |= uint(_src[_b]) << (_b * 8u); }"
            )
            # Per-uint quant params, loaded once (all nibbles share one group).
            if hoist_dequant:
                self._emit(f"uint _hgroup = _gc_base / {group_size}u;")
                self._emit(f"_hscale = {scale_name}[_gr * {n_groups}u + _hgroup];")
                if op.dequant_bias_ptr is not None:
                    self._emit(f"_hbias = {bias_name}[_gr * {n_groups}u + _hgroup];")
            self._indent -= 1
            self._emit("}")

            # Q5_0: load the matching high-bit bytes (same packed layout) so the
            # per-element extract can rebuild `nibble | (high_bit << bits)`.
            high_name = op.dequant_high_ptr.name if op.dequant_high_ptr is not None else ""
            if high_name:
                self._emit("uint _packed_high = 0u;")
                self._emit(f"if ({row_guard}) {{")
                self._indent += 1
                self._emit(f"uint _dev_pb_h = uint({packed_col_start}) / {pf}u + _pb;")
                self._emit(f"uint _bytes_left_h = min(4u, {packed_cols}u - _pb);")
                self._emit(
                    f"device const uchar* _src_h = (device const uchar*)({high_name} + {addr_base}_gr * {packed_stride}u + _dev_pb_h);"
                )
                self._emit("if (_bytes_left_h >= 4u) _packed_high = *(device const uint*)_src_h;")
                self._emit(
                    "else { for (uint _b = 0; _b < _bytes_left_h; _b++) _packed_high |= uint(_src_h[_b]) << (_b * 8u); }"
                )
                self._indent -= 1
                self._emit("}")

            # Extract nibbles and write to shared memory
            src_dt = self._buffer_dtypes.get(ptr_name, self._shmem_dtype)
            signed_byte_values = src_dt == "char" and bits == 8
            if can_vectorize:
                # End-to-end f16: extract 4 nibbles → half4, dequant with the
                # f16 scale/bias (half FMA, ~28% faster), store half4 directly.
                # No float intermediate → no `float->half` convert, and the
                # scale/bias loads above are native f16. ~1 f16-ULP/weight vs the
                # scalar float path.
                self._emit(f"if (_r < {rows}u) {{")
                self._indent += 1
                # zero_point == 0 (q4_k, q8_0): the subtract is a no-op — elide it.
                # Otherwise subtract a native half literal (`Nh`), not `half(Nf)`.
                _zp_sub = "" if float(zero_point) == 0.0 else f" - {float(zero_point)}h"
                for _ck in range(elems_per_uint // 4):
                    if _signed_byte and float(zero_point) == 0.0:
                        # q8_0 signed-byte bit-trick — the 8-bit analog of the
                        # 0x6400 nibble trick below, avoiding the char4→half4
                        # CONVERT (F32 convert pipe, ~4cyc/weight). Two's
                        # complement biases via XOR: u' = u ^ 0x80 == v + 128
                        # for all v ∈ [-128,127], so the half BIT PATTERN
                        # 0x6400|u' == 1024.0 + u' and (that − 1152) == v —
                        # every intermediate ≤ 1279 where the f16 ULP is 1.0, so
                        # the path is exact. Byte extract + XOR run as I16 vector
                        # ops.
                        # bits==8 → elems_per_uint==4, so _ck is 0.
                        self._emit("ushort2 _ps8 = as_type<ushort2>(_packed);")
                        self._emit(
                            "ushort4 _hb8 = ushort4(0x6400) | "
                            "(((ushort4(_ps8.x, _ps8.x, _ps8.y, _ps8.y) >> ushort4(0, 8, 0, 8)) "
                            "& ushort4(0xFF)) ^ ushort4(0x80));"
                        )
                        _expr = "(as_type<half4>(_hb8) - 1152.0h) * _hscale"
                    elif _signed_byte:
                        # signed bytes with a non-zero zero_point: keep the
                        # sign-extending vector convert (still one convert per
                        # 4 weights vs 4 scalar chains).
                        _expr = f"(half4(as_type<char4>(_packed)){_zp_sub}) * _hscale"
                    elif use_bittrick:
                        # Build the half BIT PATTERN with integer ops only — no
                        # convert anywhere on the path. Each 4-bit value v maps to
                        # the half bits 0x6400|v == 1024.0 + v; build the 4 half
                        # lanes of this group then bitcast to half4.
                        #
                        # Do the bit-twiddle in ushort (I16), not uint (I32). It's
                        # a dependent chain (shift -> AND -> OR), and at this
                        # kernel's low occupancy (~28%, ~2-4 simds/core) the 32-bit
                        # register-dependency penalty is large: per philip turner,
                        # dependent IADD is ~3.4-6.6cyc in I32 vs ~2.2-3.9 in I16
                        # (~1.7-2x). Bitcast the packed uint to ushort2 (free; each
                        # ushort holds this group's 4 nibbles at bits [0,4,8,12]),
                        # then build all 4 lanes with one vectorized ushort4 op
                        # set. (vector width is a no-op for ALU throughput on Apple
                        # — these are 4 scalar I16 ops — the win is I16 vs I32.)
                        if _ck == 0:
                            self._emit("ushort2 _psu = as_type<ushort2>(_packed);")
                        self._emit(
                            f"ushort4 _hb{_ck} = ushort4(0x6400) | "
                            f"((ushort4(_psu[{_ck}]) >> ushort4(0, 4, 8, 12)) & ushort4(0xF));"
                        )
                        # Subtract 1024 BEFORE scaling — exact in half (1024+n and
                        # 1024 both representable, diff == n), avoiding the fp16
                        # cancellation that folding -1024*scale into the bias would
                        # cause. The subtract runs on the otherwise-idle F16 pipe.
                        _expr = f"(as_type<half4>(_hb{_ck}) - 1024.0h) * _hscale"
                    else:
                        _shifts = ", ".join(f"{(_ck * 4 + j) * bits}u" for j in range(4))
                        self._emit(
                            f"uint4 _nib{_ck} = (uint4(_packed) >> uint4({_shifts})) & {mask_val}u;"
                        )
                        _base = f"half4(_nib{_ck}){_zp_sub}"
                        _expr = f"({_base}) * _hscale" if _zp_sub else f"{_base} * _hscale"
                    if op.dequant_bias_ptr is not None:
                        _expr += " + _hbias"
                    self._emit(
                        f"*(threadgroup half4*)(&{buf_name}[_r * {stride}u "
                        f"+ _pb * {pf}u + {_ck * 4}u]) = {_expr};"
                    )
                self._indent -= 1
                self._emit("}")
                self._indent -= 1
                self._emit("}")  # for _idx
                self._val_loc[name] = ValLoc("shared", buf_name, stride)
                self._pending_tg_barrier = True
                self._buffers_since_barrier.add(buf_name)
                return
            self._emit("#pragma unroll")
            self._emit(f"for (uint _e = 0; _e < {elems_per_uint}u; _e++) {{")
            self._indent += 1
            self._emit(f"uint _c = _pb * {pf}u + _e;")
            self._emit(f"if (_r < {rows}u && _c < {load_cols}u) {{")
            self._indent += 1
            self._emit("uint _gc = _gc_base + _e;")
            if signed_byte_values:
                self._emit("int _ival = int((_packed >> (_e * 8u)) & 255u);")
                self._emit("if (_ival >= 128) _ival -= 256;")
                self._emit("float _val = float(_ival);")
            elif high_name:
                # Q5_0: nibble (low `bits`) | high bit (bit 0 of the same nibble
                # position in the high buffer) shifted into bit `bits`.
                self._emit(f"uint _nib = (_packed >> (_e * {bits}u)) & {mask_val}u;")
                self._emit(f"uint _hb = (_packed_high >> (_e * {bits}u)) & 1u;")
                self._emit(f"float _val = float(_nib | (_hb << {bits}u));")
            else:
                self._emit(f"float _val = float((_packed >> (_e * {bits}u)) & {mask_val}u);")

            # Bounds check on column for edge tiles
            if col_bound_var:
                self._emit(f"if (_gc >= {col_bound_var}) _val = {fill};")

            # Inline dequant: (val - zero_point) * scale. When every nibble of
            # the uint shares one group, scale/bias were hoisted (loaded once
            # above) — use the registers; otherwise load per nibble.
            if has_dequant and hoist_dequant:
                self._emit(f"_val = (_val - {float(zero_point)}f) * _hscale;")
                if op.dequant_bias_ptr is not None:
                    self._emit("_val = _val + _hbias;")
            elif has_dequant:
                self._emit(f"uint _group = _gc / {group_size}u;")
                self._emit(f"float _scale = {scale_name}[_gr * {n_groups}u + _group];")
                self._emit(f"_val = (_val - {float(zero_point)}f) * _scale;")
                if op.dequant_bias_ptr is not None:
                    self._emit(f"float _bias = {bias_name}[_gr * {n_groups}u + _group];")
                    self._emit("_val = _val + _bias;")

            # Apply transform chain if present (prologue fusion)
            if op.transform:
                self._emit("// prologue transform")
                for xf_op in op.transform:
                    expr = format_scalar_op(
                        xf_op,
                        lambda v: "_val" if v.name == op.transform_source_name else self._get(v),
                    )
                    if expr:
                        self._emit(f"_val = {expr};")

            self._emit(
                f"{buf_name}[{db_off}_r * {stride}u + _c] = {self._shmem_cast('_val', buf_name)};"
            )
            self._indent -= 1
            self._emit("}")  # if in bounds
            self._indent -= 1
            self._emit("}")  # for _e
            self._indent -= 1
            self._emit("}")  # for _idx
            self._val_loc[name] = ValLoc("shared", buf_name, stride)
            self._pending_tg_barrier = True
            self._buffers_since_barrier.add(buf_name)
            return

        row_modulus = self._extract_row_modulus(op.row_indices)
        # GATHERED row index: the row index is itself a Load (an index buffer) — e.g. the
        # MoE grouped GEMM gathers token rows via ROW_TOKEN[rm]. Affine row indices are
        # NEVER a Load (they're ProgramId/MakeRange arithmetic), so this branch is gated to
        # true gathers and the affine fast path (every other GEMM) is untouched. Each
        # loaded row `_r` reads its source row from the index buffer; the `_gr < row_bound`
        # check emitted below guards garbage pad-row indices (skips the load → fill).
        _gather = self._gather_index_load(op.row_indices)
        self._emit_coop_loop_open(n_vecs)
        self._emit_coop_decode(load_cols // vec_width)
        if _gather is not None:
            _idx_base = self._extract_range_base(_gather.offsets)
            self._emit(f"uint _gr = uint(int({_gather.ptr.name}[uint({_idx_base}) + _r]));")
        elif row_modulus is not None:
            self._emit(f"uint _gr = (uint({row_start}) + _r) % uint({row_modulus});")
        else:
            self._emit(f"uint _gr = uint({row_start}) + _r;")
        self._emit(f"uint _gc = uint({col_start}){col_off} + _cv * {vec_width}u;")
        # Extract per-load bounds from mask (e.g., row < M && col < K)
        row_bound_var, col_bound_var = self._extract_load_bounds(op)
        # mask=True (Constant(True)): unconditional load, skip blanket bounds
        _mask_unconditional = False
        if op.mask:
            _mop = self._op_map.get(op.mask.name)
            if isinstance(_mop, Constant) and _mop.value is True:
                _mask_unconditional = True
        # Fallback to blanket M if mask doesn't provide row bound
        if not _mask_unconditional and not row_bound_var and self._row_bound:
            row_bound_var = self._row_bound

        # Elide bounds checks when tile dimensions evenly divide problem dimensions.
        # When M % BM == 0, every threadgroup's rows are guaranteed in-bounds.
        # When K % BK == 0, every K-iteration's columns are guaranteed in-bounds.
        ce = self.func.constexpr_values

        def _resolve_bound(bvar):
            """Resolve bound variable to integer value (constexpr name or literal)."""
            if bvar is None:
                return None
            v = ce.get(bvar)
            if isinstance(v, int):
                return v
            try:
                return int(bvar)
            except (ValueError, TypeError):
                return None

        rb_val = _resolve_bound(row_bound_var)
        if rb_val is not None and rb_val % rows == 0:
            row_bound_var = None
        cb_val = _resolve_bound(col_bound_var)
        if cb_val is not None and cb_val % cols == 0:
            col_bound_var = None

        bounds = []
        if row_bound_var:
            bounds.append(f"_gr < {row_bound_var}")
        if col_bound_var:
            bounds.append(f"_gc + {vec_width - 1}u < {col_bound_var}")
        elif self._col_offset:
            bounds.append(f"_gc + {vec_width}u <= N")

        # Type conversion: when buffer dtype differs from the per-buffer
        # shmem dtype (e.g., char→float for int8 promotion, OR float→half
        # for downcast on Q tiles paired with half K in mixed-precision
        # attention), load `_raw` at the buffer dtype and convert to
        # `_val` at the shmem dtype. Compare against `sdt` (per-buffer)
        # not `self._shmem_dtype` (kernel-global) — per-value dtype
        # overrides via `_compute_per_value_shmem_dtype` mean a buffer
        # can be narrower than the kernel-global shmem dtype.
        buf_dt = self._buffer_dtypes.get(ptr_name, sdt)
        needs_promote = buf_dt != sdt
        buf_vec_type = f"{buf_dt}{vec_width}" if needs_promote else vec_type
        # Sub-word vec4 requires aligned row stride (e.g., char4 needs stride % 4 == 0)
        can_vec = not needs_promote or row_stride % vec_width == 0
        if needs_promote:
            fill_scalar = f"{float(op.other)}f" if buf_dt == "float" else str(int(op.other))
            fill_target = "_raw"
            fill_type = buf_vec_type
        else:
            fill_scalar = (
                f"{float(op.other)}f" if self._shmem_dtype == "float" else str(int(op.other))
            )
            fill_target = "_val"
            fill_type = vec_type

        if bounds:
            if needs_promote:
                self._emit(f"{buf_vec_type} _raw;")
            self._emit(f"{vec_type} _val;")
            if can_vec:
                self._emit(f"if ({' && '.join(bounds)}) {{")
                self._indent += 1
                if needs_promote:
                    self._emit(
                        f"_raw = *(device const {buf_vec_type}*)({ptr_name} + {addr_base}_gr * {row_stride}u + _gc);"
                    )
                else:
                    self._emit(
                        f"_val = *(device const {vec_type}*)({ptr_name} + {addr_base}_gr * {row_stride}u + _gc);"
                    )
                self._indent -= 1
                self._emit("} else {")
                self._indent += 1
            else:
                self._emit("{")
                self._indent += 1
            self._emit(f"{fill_target} = {fill_type}({fill_scalar});")
            # Per-element fallback for partial vec at column boundaries only.
            # When there is no col bound, the row is either fully valid (vec4
            # path above) or fully out-of-bounds (zero-fill is correct). The
            # scalar fallback inside the else branch with only a row guard
            # would be dead code: we're already in else(_gr >= M), so _gr < M
            # is always false here.
            row_guard = f"_gr < {row_bound_var}" if row_bound_var else ""
            col_guard_var = col_bound_var if col_bound_var else ("N" if self._col_offset else "")
            if col_guard_var:
                guard = row_guard if row_guard else "true"
                scalar_target = "_raw" if needs_promote else "_val"
                self._emit(f"if ({guard}) for (uint _vi = 0; _vi < {vec_width}u; _vi++)")
                self._emit(
                    f"    if (_gc + _vi < {col_guard_var}) {scalar_target}[_vi] = {ptr_name}[{addr_base}_gr * {row_stride}u + _gc + _vi];"
                )
            self._indent -= 1
            self._emit("}")
            if needs_promote:
                self._emit(f"_val = {vec_type}(_raw);")
            if op.transform:
                self._emit_load_transform(op.transform, vec_type)
            db_off_expr = self._db_shmem_offsets.get(buf_name, "") if self._db_shmem_offsets else ""
            db_off = f"{db_off_expr} + " if db_off_expr else ""
            if op.addr_transposed:
                sdt = self._buf_shmem_dtype(buf_name)
                for j in range(vec_width):
                    self._emit(
                        f"{buf_name}[{db_off}(_cv * {vec_width}u + {j}u) * {stride}u + _r] = ({sdt})_val[{j}];"
                    )
            else:
                self._emit(
                    f"*(threadgroup {vec_type}*)(&{buf_name}[{db_off}_r * {stride}u + _cv * {vec_width}u]) = _val;"
                )
        else:
            if needs_promote:
                self._emit(
                    f"{buf_vec_type} _raw = *(device const {buf_vec_type}*)({ptr_name} + {addr_base}_gr * {row_stride}u + _gc);"
                )
                self._emit(f"{vec_type} _val = {vec_type}(_raw);")
            else:
                self._emit(
                    f"{vec_type} _val = *(device const {vec_type}*)({ptr_name} + {addr_base}_gr * {row_stride}u + _gc);"
                )
            if op.transform:
                self._emit_load_transform(op.transform, vec_type)
            db_off_expr = self._db_shmem_offsets.get(buf_name, "") if self._db_shmem_offsets else ""
            db_off = f"{db_off_expr} + " if db_off_expr else ""
            if op.addr_transposed:
                sdt = self._buf_shmem_dtype(buf_name)
                for j in range(vec_width):
                    self._emit(
                        f"{buf_name}[{db_off}(_cv * {vec_width}u + {j}u) * {stride}u + _r] = ({sdt})_val[{j}];"
                    )
            else:
                self._emit(
                    f"*(threadgroup {vec_type}*)(&{buf_name}[{db_off}_r * {stride}u + _cv * {vec_width}u]) = _val;"
                )
        self._indent -= 1
        self._emit("}")
        if not self._db_shmem_offsets:
            # Defer barrier — will be flushed before the first consumer (Dot, next loop iter)
            self._pending_tg_barrier = True
            self._buffers_since_barrier.add(buf_name)

        self._val_loc[name] = ValLoc("shared", buf_name, stride)

    def _emit_register_load(self, op: Load):
        """Load 2D data into per-thread local arrays (register-resident mode).

        Each lane loads D = cols/tpr elements with interleaved addressing for
        coalesced global memory access.  No shared memory or barriers needed.
        """
        name = op.result.name
        rows, cols = op.result.shape
        D = cols // self._tpr

        ptr_name = op.ptr.name if op.ptr else "???"
        row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
        addr_base = f"({base_offset}) + " if base_offset != "0" else ""

        self._emit(f"// Register load {name} [{rows}x{cols}] → local[{D}]")
        self._emit(f"{self._acc_dtype} {name}[{D}];")
        self._emit(f"if (_row < {rows}u) {{")
        self._indent += 1
        self._emit(f"uint _gr = uint({row_start}) + _row;")
        if self._row_bound:
            self._emit(f"if (_gr < {self._row_bound}) {{")
            self._indent += 1
            self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
            self._emit(
                f"    {name}[_d] = {self._acc_dtype}({ptr_name}[{addr_base}_gr * {row_stride}u + uint({col_start}) + _d * {self._tpr}u + _lane]);"
            )
            self._indent -= 1
            self._emit("} else {")
            self._indent += 1
            self._emit(f"for (uint _d = 0; _d < {D}u; _d++) {name}[_d] = 0.0f;")
            self._indent -= 1
            self._emit("}")
        else:
            self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
            self._emit(
                f"    {name}[_d] = {self._acc_dtype}({ptr_name}[{addr_base}(uint({row_start}) + _row) * {row_stride}u + uint({col_start}) + _d * {self._tpr}u + _lane]);"
            )
        self._indent -= 1
        self._emit("} else {")
        self._indent += 1
        self._emit(f"for (uint _d = 0; _d < {D}u; _d++) {name}[_d] = 0.0f;")
        self._indent -= 1
        self._emit("}")

        self._local_arrays[name] = D
        self._exprs[name] = name
        self._val_loc[name] = ValLoc("local_array")
        # No barrier — each thread loaded its own data

    # --- Cooperative 2D store: shared/local → global memory ---

    def _emit_coop_store(self, op: Store):
        """Store a 2D value (shared or local array) back to global memory."""
        ptr_name = op.ptr.name if op.ptr else "out"
        val = op.value
        val_loc = self._val_loc.get(val.name, PER_THREAD) if val else PER_THREAD

        # Address callback: custom store addressing (e.g. attention output)
        store_cb = self._config.get(f"addr_store_{ptr_name}")
        if store_cb and val_loc.kind == "local_array":
            D = self._local_arrays[val.name]
            n_rows = val.shape[0] if val and len(val.shape) == 2 else 1
            self._emit(f"if (_row < {n_rows}u) {{")
            self._indent += 1
            self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
            store_cb(self, "_row", "_d")
            self._indent -= 1
            self._emit("}")
            return

        # Use semantic 2D addressing if available, else trace
        if op.row_indices is not None and op.col_indices is not None and op.row_stride is not None:
            n_rows, n_cols = (
                op.offsets.shape if op.offsets and len(op.offsets.shape) == 2 else val.shape
            )
            row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
        elif op.offsets and len(op.offsets.shape) == 2:
            n_rows, n_cols = op.offsets.shape
            row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
        elif val and len(val.shape) == 2:
            n_rows, n_cols = val.shape
            row_start, row_stride, col_start, base_offset = "0", n_cols, "0", "0"
        else:
            self._emit_store(op)
            return
        addr_base = f"({base_offset}) + " if base_offset != "0" else ""
        store_row_bound, store_col_bound = (None, None)
        if op.mask:
            store_row_bound, store_col_bound = self._extract_load_bounds(op)

        if val_loc.kind == "shared":
            buf, stride = val_loc.name, val_loc.stride
            store_cols = self._eff_cols(n_cols)
            col_off = f" + {self._col_offset}" if self._col_offset else ""
            self._emit(f"// Cooperative store [{n_rows}x{store_cols}] → {ptr_name}")
            if self._row_bound and op.base_offset is None:
                # Both local (prevent OOB shared reads) and global bounds
                guard = f"_row < {n_rows}u && uint({row_start}) + _row < {self._row_bound}"
            elif store_row_bound:
                guard = f"_row < {n_rows}u && uint({row_start}) + _row < {store_row_bound}"
            else:
                guard = f"_row < {n_rows}u"
            self._emit(f"if ({guard}) {{")
            self._indent += 1
            self._emit(f"uint _gr = uint({row_start}) + _row;")
            if self._tpr > 1:
                self._emit(f"for (uint _c = _lane; _c < {store_cols}u; _c += {self._tpr}u)")
            else:
                self._emit(f"for (uint _c = 0; _c < {store_cols}u; _c++)")
            # Column bounds check for partial last tile
            col_guard = ""
            if self._col_offset:
                col_guard = f"if (uint({col_start}){col_off} + _c < N) "
            out_dt = self._buffer_dtypes.get(ptr_name, self._shmem_dtype)
            if op.transform:
                if op.col_slice is not None:
                    # Column-slice epilogue: guard + remapped addressing
                    cs = op.col_slice
                    global_col = f"(uint({col_start}){col_off} + _c)"
                    self._emit(
                        f"    if ({global_col} >= {cs.col_start}u && {global_col} < {cs.col_end}u) {{"
                    )
                    self._indent += 1
                    self._emit(f"    {self._shmem_dtype} _ev = {buf}[_row * {stride}u + _c];")
                    self._emit_store_transform(
                        op.transform,
                        self._shmem_dtype,
                        "_ev",
                        row_stride=op.row_stride,
                        chain_source_name=op.transform_source_name,
                        extra_transforms=op.transform_extras,
                    )
                    self._emit(
                        f"    {ptr_name}[_gr * {cs.out_stride}u + ({global_col} - {cs.col_start}u)] = {out_dt}(_ev);"
                    )
                    self._indent -= 1
                    self._emit("    }")
                else:
                    self._emit(f"    {col_guard}{{")
                    self._indent += 1
                    self._emit(f"    {self._shmem_dtype} _ev = {buf}[_row * {stride}u + _c];")
                    self._emit_store_transform(
                        op.transform,
                        self._shmem_dtype,
                        "_ev",
                        row_stride=op.row_stride,
                        chain_source_name=op.transform_source_name,
                        extra_transforms=op.transform_extras,
                    )
                    self._emit(
                        f"    {ptr_name}[{addr_base}_gr * {row_stride}u + uint({col_start}){col_off} + _c] = {out_dt}(_ev);"
                    )
                    self._indent -= 1
                    self._emit("    }")
            else:
                self._emit(
                    f"    {col_guard}{ptr_name}[{addr_base}_gr * {row_stride}u + uint({col_start}){col_off} + _c] = {out_dt}({buf}[_row * {stride}u + _c]);"
                )
            self._indent -= 1
            self._emit("}")
        elif val_loc.kind == "local_array":
            D = self._local_arrays.get(val.name, n_cols)
            arr_name = self._exprs.get(val.name, val.name)
            self._emit(f"// Store local array [{n_rows}x{D}] → {ptr_name}")
            if self._row_bound and op.base_offset is None:
                guard = f"_row < {n_rows}u && uint({row_start}) + _row < {self._row_bound}"
            elif store_row_bound:
                guard = f"_row < {n_rows}u && uint({row_start}) + _row < {store_row_bound}"
            else:
                guard = f"_row < {n_rows}u"
            self._emit(f"if ({guard}) {{")
            self._indent += 1
            self._emit(f"uint _gr = uint({row_start}) + _row;")
            if self._register_resident:
                # Interleaved: lane l wrote element _d at column _d*tpr+l
                out_dt = self._buffer_dtypes.get(ptr_name, self._shmem_dtype)
                self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
                self._emit(
                    f"    {ptr_name}[{addr_base}_gr * {row_stride}u + uint({col_start}) + _d * {self._tpr}u + _lane] = {out_dt}({arr_name}[_d]);"
                )
            else:
                out_dt = self._buffer_dtypes.get(ptr_name, self._shmem_dtype)
                self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
                self._emit(
                    f"    {ptr_name}[{addr_base}_gr * {row_stride}u + uint({col_start}) + _d] = {out_dt}({arr_name}[_d]);"
                )
            self._indent -= 1
            self._emit("}")
        else:
            # Callback or error
            store_cb = self._config.get("store_callback")
            if store_cb:
                store_cb(self, op)
            else:
                raise ValueError(f"Unsupported store val_loc: {val_loc}")

    # --- Scalar load/store ---

    def _emit_store(self, op: Store):
        store_cb = self._config.get("store_callback")
        if store_cb:
            store_cb(self, op)
        else:
            ptr = self._get(op.ptr) if op.ptr else "out"
            offs = self._get(op.offsets) if op.offsets else "0"
            val = self._get(op.value) if op.value else "0.0f"

            # Apply Store.transform (epilogue fusion)
            store_offs_remapped = None
            if op.transform:
                # Non-source transform operands that are NOT device buffers are
                # local/register values (e.g. a loop-carried per-thread acc); the
                # store-transform path would otherwise index them as
                # `name[store_offs]` (correct only for a fused residual buffer).
                # Pre-resolve them to their MSL register expression.
                produced = {t.result.name for t in op.transform if t.result}
                src_name = op.transform_source_name
                param_names = {p.name for p in self.func.params}
                operand_exprs: dict[str, str] = {}
                for t in op.transform:
                    for v in t.operand_values():
                        if v.name in produced or v.name == src_name or v.name in operand_exprs:
                            continue
                        if v.name not in param_names:
                            operand_exprs[v.name] = self._get(v)
                val = self._eval_transform(
                    op.transform,
                    val,
                    store_offs=offs,
                    extra_transforms=op.transform_extras,
                    chain_source_name=op.transform_source_name,
                    operand_exprs=operand_exprs,
                )
                # Scatter-store remapping: only if the OUTPUT param
                # itself has nd_strides (view+permute epilogue).
                # Never use an extra's nd_strides for the store address.
                cv = self.func.constexpr_values
                _out_name = op.ptr.name if op.ptr else None
                if _out_name:
                    nd_s = cv.get(f"_{_out_name}_nd_shape")
                    nd_st = cv.get(f"_{_out_name}_nd_strides")
                    if nd_s is not None and nd_st is not None and len(nd_s) > 2:
                        parts = []
                        for d in range(len(nd_s) - 1):
                            inner_size = 1
                            for dd in range(d + 1, len(nd_s)):
                                inner_size *= nd_s[dd]
                            idx = f"(({offs}) / {inner_size}u) % {nd_s[d]}u"
                            parts.append(f"({idx}) * {nd_st[d]}u")
                        parts.append(f"(({offs}) % {nd_s[-1]}u)")
                        store_offs_remapped = f"({' + '.join(parts)})"

            # Store address — may differ from flat offs when epilogue crosses view+permute
            store_addr = store_offs_remapped if store_offs_remapped else offs

            # Determine store dtype for cast — compare against actual MSL type
            # (sub-word types like bf16/i8/i16 are promoted to acc_dtype for compute)
            ptr_dtype = self._buffer_dtypes.get(op.ptr.name, self._dtype) if op.ptr else self._dtype
            cast = f"{ptr_dtype}({val})" if ptr_dtype != self._acc_dtype else val

            # reduce="add" scatters overlap (e.g. the embedding backward, where a
            # repeated token maps several rows to the same destination), so the
            # write is an atomic_fetch_add into the (atomic<float>*-declared)
            # destination rather than a plain store.
            def _store_stmt(addr: str, value: str) -> str:
                if op.reduce == "add":
                    return f"atomic_fetch_add_explicit(&{ptr}[{addr}], float({value}), memory_order_relaxed);"
                return f"{ptr}[{addr}] = {value};"

            # Guard logic for 1D stores
            # 1. Post-reduction scalar store: only tid 0 writes
            if self._has_1d_reduce and not op.mask and not self._mask_expr:
                self._emit(f"if (tid == 0) {_store_stmt(store_addr, cast)}")
                return
            # 2. Mask guard from Compare
            if op.mask:
                mask = self._get(op.mask)
                self._emit(f"if ({mask}) {_store_stmt(store_addr, cast)}")
                return
            if self._mask_expr:
                self._emit(f"if ({self._mask_expr}) {_store_stmt(store_addr, cast)}")
                return

            # In dot composable, more threads than rows — guard 1D stores
            block_m = self.func.constexpr_values.get("BLOCK_M", 0)
            needs_guard = block_m > 0 and self._threads > block_m and self._tpr == 1
            if needs_guard:
                self._emit(f"if (_row < {block_m}u)")
            self._emit(
                f"    {_store_stmt(store_addr, f'{self._dtype}({val})')}"
                if needs_guard
                else _store_stmt(store_addr, cast)
            )

    def _emit_scalar_load(self, op: Load):
        name = op.result.name
        ptr = self._get(op.ptr)

        # In dot composable: 1D load from a range.
        # Two cases:
        #   (a) Broadcast context: load N elements into local array for
        #       broadcast-add to 2D MMA result (e.g., bias vector).
        #   (b) Distributed context: each thread loads 1 element at tid offset
        #       for parallel reduction (e.g., RMSNorm sq_sum accumulation).
        # Heuristic: if the load is inside a per-row for loop (ForLoop body
        # where we're iterating rows), it's distributed. Otherwise broadcast.
        result_shape = op.result.shape if op.result else ()
        if (
            not self._per_thread
            and len(result_shape) == 1
            and result_shape[0] > 1
            and self._tpr == 1
            and self._threads > 1
            and op.offsets
        ):
            D = result_shape[0]
            base = self._extract_range_base(op.offsets)
            # Check if this load is used in a 2D broadcast context
            # (operand of a BinOp whose result is 2D). If not, use
            # per-thread scalar distribution instead of local array.
            is_broadcast = False
            for user_op in self._find_users(name):
                if hasattr(user_op, "result") and user_op.result and len(user_op.result.shape) == 2:
                    is_broadcast = True
                    break
            if is_broadcast:
                # If every 2D consumer of this load is inside a flat-threaded
                # FusedElementwise chain whose `_c` matches our `D`, skip the
                # per-thread broadcast array and inline a single scalar load
                # at the chain's column index (256 threads × 1 read instead
                # of 256 threads × D reads on a per-thread float[D] array).
                if self._all_consumers_flat_threaded(name, D):
                    self._flat_loads_inline[name] = (
                        f"{self._acc_dtype}({ptr}[uint({base}) + {{col}}])"
                    )
                    # Register a placeholder location so `_resolve` knows to
                    # consult `_flat_loads_inline` rather than treat this as
                    # a per-thread scalar.
                    self._exprs[name] = name
                    self._val_loc[name] = ValLoc("flat_load_inline")
                    return
                self._emit(f"// 1D load {name} [{D}] → local array (broadcast)")
                self._emit(f"{self._acc_dtype} {name}[{D}];")
                self._emit(f"for (uint _d = 0; _d < {D}u; _d++)")
                self._emit(f"    {name}[_d] = {self._acc_dtype}({ptr}[uint({base}) + _d]);")
                self._local_arrays[name] = D
                self._exprs[name] = name
                self._val_loc[name] = ValLoc("local_array")
                return
            # Non-broadcast: fall through to scalar load path (uses tid via MakeRange→"tid")

        offs = self._get(op.offsets)
        # Determine load dtype from buffer
        buf_dt = self._buffer_dtypes.get(op.ptr.name, self._dtype) if op.ptr else self._dtype
        load_type = self._acc_dtype if buf_dt in ("char", "short", "bfloat") else buf_dt

        if op.mask:
            mask = self._get(op.mask)
            other = f"{op.other}f" if load_type == "float" else str(int(op.other))
            expr = f"({mask} ? {load_type}({ptr}[{offs}]) : {other})"
            self._emit(f"{load_type} {name} = {expr};")
        else:
            expr = f"{load_type}({ptr}[{offs}])"
            self._emit(f"{load_type} {name} = {expr};")
        # Apply Load.transform (prologue fusion) — applies to all paths.
        # Must be guarded by the same mask as the load to avoid OOB access
        # on extra buffers (e.g., bias in add+layernorm) for masked-out threads.
        if op.transform:
            xf_expr = self._eval_transform(
                op.transform,
                name,
                store_offs=offs,
                extra_transforms=op.transform_extras,
                chain_source_name=op.transform_source_name,
            )
            if op.mask:
                mask = self._get(op.mask)
                self._emit(f"if ({mask}) {name} = {xf_expr};")
            else:
                self._emit(f"{name} = {xf_expr};")
        self._exprs[name] = name
        self._val_loc[name] = PER_THREAD

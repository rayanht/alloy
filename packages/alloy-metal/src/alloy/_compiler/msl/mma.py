"""MMA MSL emitter methods."""

from __future__ import annotations

from alloy._compiler.fusion_transforms import (
    ColumnBroadcastTransform,
    IdentityTransform,
    RowBroadcastTransform,
    ScalarBroadcastTransform,
    ScatterTransform,
    StridedTransform,
)
from alloy._compiler.msl.context import MMA, PER_THREAD, PERSISTENT_MMA, ValLoc
from alloy._compiler.tile_ir import Dot, Store, TileValue
from alloy._compiler.tile_plan import pick_dot_reg


class MmaEmitterMixin:
    def _acc_pre_scale_buf_name(self, dot_op):
        return f"_acc_pre_scale_{dot_op.acc.name}"

    def _acc_post_scale_buf_name(self, dot_op):
        return f"_acc_post_scale_{dot_op.acc.name}"

    # --- Scalar dot (BLOCK_M < 8, no MMA) ---

    def _emit_scalar_dot(self, op: Dot):
        """Emit per-thread scalar dot product for small BLOCK_M (< 8).

        Each thread computes one (m, n) output element by iterating over K.
        Reads A and B from shared memory (already loaded cooperatively).
        """
        self._flush_tg_barrier()

        result = op.result
        lhs = op.lhs
        rhs = op.rhs

        lhs_loc = self._val_loc.get(lhs.name, PER_THREAD)
        rhs_loc = self._val_loc.get(rhs.name, PER_THREAD)
        lhs_buf = lhs_loc.name if lhs_loc.kind == "shared" else "???"
        lhs_stride = lhs_loc.stride if lhs_loc.kind == "shared" else 0
        rhs_buf = rhs_loc.name if rhs_loc.kind == "shared" else "???"
        rhs_stride = rhs_loc.stride if rhs_loc.kind == "shared" else 0

        if op.transpose_lhs:
            K_dim, M_dim = lhs.shape
        else:
            M_dim, K_dim = lhs.shape
        if op.transpose_rhs:
            N_dim = rhs.shape[0]
        else:
            _, N_dim = rhs.shape

        is_persistent = op.acc is not None
        acc_name = f"_sacc_{op.acc.name if is_persistent else result.name}"
        acc_stride = self._eff_cols(N_dim)

        if not is_persistent:
            self._emit(f"// Scalar dot: {M_dim}×{K_dim} @ {K_dim}×{N_dim}")
            self._scalar_dot_result = (acc_name, M_dim, N_dim)
        elif op.acc_pre_scale is not None:
            alpha_expr = self._get(op.acc_pre_scale)
            self._emit(f"if (_row < {M_dim}u) {{")
            self._indent += 1
            self._emit(f"float _alpha = {alpha_expr};")
            self._emit(f"for (uint _c = 0; _c < {N_dim}u; _c++)")
            self._emit(f"    {acc_name}[_row * {acc_stride}u + _c] *= _alpha;")
            self._indent -= 1
            self._emit("}")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        # else: persistent accumulator was declared and zeroed at Zeros site

        # SIMD-reduce dot: each simdgroup handles output columns strided by n_sg.
        # 32 lanes parallelize K-reduction via simd_sum.
        n_sg = "NUM_THREADS / 32u"
        self._emit(f"{{ // simd-reduce dot {M_dim}×{K_dim} @ {K_dim}×{N_dim}")
        self._indent += 1
        self._emit(f"for (uint _col = simd_gid; _col < {N_dim}u; _col += {n_sg}) {{")
        self._indent += 1
        for m in range(M_dim):
            self._emit(f"{self._acc_dtype} _sum{m} = 0;")
        self._emit(f"for (uint _kb = 0; _kb < {K_dim}u; _kb += 32u) {{")
        self._indent += 1
        self._emit("uint _k = _kb + simd_lane;")
        if op.transpose_rhs:
            self._emit(
                f"{self._acc_dtype} _bv = (_k < {K_dim}u) ? {self._acc_dtype}({rhs_buf}[_col * {rhs_stride}u + _k]) : 0;"
            )
        else:
            self._emit(
                f"{self._acc_dtype} _bv = (_k < {K_dim}u) ? {self._acc_dtype}({rhs_buf}[_k * {rhs_stride}u + _col]) : 0;"
            )
        for m in range(M_dim):
            if op.transpose_lhs:
                # lhs storage is (K, M) with row stride M
                av_expr = (
                    f"(_k < {K_dim}u) ? {self._acc_dtype}({lhs_buf}[_k * {lhs_stride}u + {m}u]) : 0"
                )
            else:
                av_expr = (
                    f"(_k < {K_dim}u) ? {self._acc_dtype}({lhs_buf}[{m}u * {lhs_stride}u + _k]) : 0"
                )
            self._emit(f"{{ {self._acc_dtype} _av = {av_expr};")
            self._emit(f"  _sum{m} += simd_sum(_av * _bv); }}")
        self._indent -= 1
        self._emit("}")
        # All lanes have the same sum after simd_sum — all write to accumulator
        for m in range(M_dim):
            update_op = "+=" if is_persistent else "="
            self._emit(
                f"if (simd_lane == 0) {acc_name}[{m}u * {acc_stride}u + _col] {update_op} _sum{m};"
            )
        self._indent -= 1
        self._emit("}")
        self._indent -= 1
        self._emit("}")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Mark result location
        self._val_loc[result.name] = ValLoc("shared", acc_name, acc_stride)
        self._exprs[result.name] = acc_name
        self._scalar_dot_result = (acc_name, M_dim, N_dim)

    # --- Simdgroup MMA ---

    def _emit_mma(self, op: Dot):
        # Conditional barrier flush: only emit if our reads/writes conflict
        # with pending state. RAR stays silent.
        result = op.result
        lhs = op.lhs
        rhs = op.rhs
        _pre_lhs_loc = self._val_loc.get(lhs.name, PER_THREAD)
        _pre_rhs_loc = self._val_loc.get(rhs.name, PER_THREAD)
        _pre_reads: set[str] = set()
        if _pre_lhs_loc.kind == "shared":
            _pre_reads.add(_pre_lhs_loc.name)
        if _pre_rhs_loc.kind == "shared":
            _pre_reads.add(_pre_rhs_loc.name)
        # Result buf only matters if it's stored to shmem (non-persistent MMA
        # with a shmem_plan entry). Persistent MMA result stays in registers.
        _is_persistent_pre = op.acc is not None
        _pre_writes: set[str] = set()
        if not _is_persistent_pre:
            _res_plan = self._shmem_plan.get(result.name)
            if _res_plan:
                _pre_writes.add(_res_plan[0])
        self._check_flush_for(reads=_pre_reads, writes=_pre_writes)
        # MMA reads shmem across all simdgroups; if a row-guarded scalar op
        # just touched shmem, idle simdgroups may race ahead without sync.
        if self._scalar_shmem_dirty:
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._scalar_shmem_dirty = False
            self._buffers_since_barrier.clear()
            self._buffers_read.clear()
        # Per-dot reg pick: each Dot's tile factor must independently divide
        # its (M, N). A kernel-global reg wastes simdgroups for big dots or
        # makes `_sg_cols = N // (reg*8) = 0` and skips small dots entirely.
        if op.transpose_lhs:
            _M_for_reg = lhs.shape[1]
        else:
            _M_for_reg = lhs.shape[0]
        _N_for_reg = rhs.shape[0] if op.transpose_rhs else rhs.shape[1]
        reg = pick_dot_reg(
            _M_for_reg, _N_for_reg, override=self._reg_override, target_n_sg=self._n_sg_target
        )
        TM = reg * 8
        TN = reg * 8

        lhs_loc = self._val_loc.get(lhs.name, PER_THREAD)
        rhs_loc = self._val_loc.get(rhs.name, PER_THREAD)

        # Materialize persistent MMA values to shared memory if needed
        # (chained GEMM: inner loop's MMA result feeds outer loop's Dot)
        if lhs_loc == PERSISTENT_MMA:
            self._materialize_mma_to_shmem(lhs)
            lhs_loc = self._val_loc.get(lhs.name, PER_THREAD)
        if rhs_loc == PERSISTENT_MMA:
            self._materialize_mma_to_shmem(rhs)
            rhs_loc = self._val_loc.get(rhs.name, PER_THREAD)

        # Materialize local_array values to shared memory for simdgroup_load.
        # This happens in attention backward where a 2D Load uses register-resident
        # mode (local arrays) but a downstream Dot needs it in threadgroup memory.
        for operand, loc, label in [(lhs, lhs_loc, "lhs"), (rhs, rhs_loc, "rhs")]:
            if loc.kind == "local_array" and len(operand.shape) == 2:
                plan_entry = self._shmem_plan.get(operand.name)
                if plan_entry:
                    buf_name, _, _, stride = plan_entry
                    rows, cols = operand.shape
                    arr_name = self._exprs.get(operand.name, operand.name)
                    D = self._local_arrays.get(operand.name, cols)
                    self._emit(f"// Spill {operand.name} from local to shmem for Dot")
                    self._emit(f"if (_row < {rows}u) {{")
                    self._indent += 1
                    self._emit(f"for (uint _c = 0; _c < {D}u; _c++)")
                    self._emit(
                        f"    {buf_name}[_row * {stride}u + _c] = {self._shmem_cast(f'{arr_name}[_c]', buf_name)};"
                    )
                    self._indent -= 1
                    self._emit("}")
                    self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    self._val_loc[operand.name] = ValLoc("shared", buf_name, stride)
                    if operand is lhs:
                        lhs_loc = self._val_loc[operand.name]
                    else:
                        rhs_loc = self._val_loc[operand.name]

        lhs_buf = lhs_loc.name if lhs_loc.kind == "shared" else "???"
        lhs_stride = lhs_loc.stride if lhs_loc.kind == "shared" else 0
        rhs_buf = rhs_loc.name if rhs_loc.kind == "shared" else "???"
        rhs_stride = rhs_loc.stride if rhs_loc.kind == "shared" else 0

        if op.transpose_lhs:
            K_dim, M_dim = lhs.shape
        else:
            M_dim, K_dim = lhs.shape
        if op.transpose_rhs:
            N_dim = rhs.shape[0]
        else:
            _, N_dim = rhs.shape

        sg_rows = M_dim // TM
        sg_cols = N_dim // TN
        n_sg = sg_rows * sg_cols

        is_persistent = op.acc is not None
        if is_persistent:
            acc_pfx = f"_acc_{op.acc.name}"
            sg_m = f"_sg_m_{op.acc.name}"
            sg_n = f"_sg_n_{op.acc.name}"
        else:
            acc_pfx = f"_acc_{result.name}"
            for i in range(reg):
                for j in range(reg):
                    self._emit(f"simdgroup_matrix<{self._acc_dtype}, 8, 8> {acc_pfx}_{i}_{j}(0);")
            sg_cols_var = f"_sg_cols_{result.name}"
            self._emit(f"uint {sg_cols_var} = {sg_cols}u;")
            sg_m = f"(simd_gid / {sg_cols_var})"
            sg_n = f"(simd_gid % {sg_cols_var})"

        # COW: before the Dot writes to shmem, save any live value in the
        # target buffer. Must happen outside the simd_gid guard so the local
        # array is visible to all threads for subsequent elementwise ops.
        did_cow = False
        if not is_persistent:
            res_plan = self._shmem_plan.get(result.name)
            if res_plan:
                cow_buf = res_plan[0]
                for vname, vloc in list(self._val_loc.items()):
                    if (
                        vloc.kind == "shared"
                        and vloc.name == cow_buf
                        and vname != result.name
                        and self._has_future_use(vname, op)
                    ):

                        class CowProxy:
                            pass

                        CowProxy.name = vname
                        self._cow_save(CowProxy(), vloc, M_dim, vloc.stride)
                        did_cow = True

        # Fence the cross-thread race: all threads read the shmem buffer into
        # per-thread _cow arrays above, then only simd 0 issues simdgroup_store
        # to the same buffer below. Without this barrier, simd 0 can overwrite
        # the buffer before other simds finish the COW read.
        if did_cow:
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        # FA-2 forward `o = (o * alpha) + tile_dot(p, V)` rescale: per-row
        # alpha materialise to small shmem buffer, then per-lane rescale
        # of the persistent acc via thread_elements() (in-register, no spill).
        if is_persistent and op.acc_pre_scale is not None:
            alpha_buf = self._acc_pre_scale_buf_name(op)
            alpha_rows = op.acc.shape[0]
            alpha_expr = self._get(op.acc_pre_scale)
            self._emit(f"if (_row < {alpha_rows}u) {{ {alpha_buf}[_row] = {alpha_expr}; }}")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        self._emit(f"if (simd_gid < {n_sg}u) {{")
        self._indent += 1

        # FA-2 forward rescale `o = (o * alpha) + tile_dot(p, V)`: multiply
        # the persistent accumulator's per-lane thread_elements() by the
        # per-row alpha in-register. thread_elements() writes propagate into
        # the next simdgroup_multiply_accumulate on M4 Max / Metal 3.x, so the
        # accumulator need not spill to shmem to be rescaled. Lane→element
        # mapping: lane holds thread_elements()[0..1] at row `_lr` of an 8x8
        # simdgroup_matrix.
        if is_persistent and op.acc_pre_scale is not None:
            alpha_buf = self._acc_pre_scale_buf_name(op)
            self._emit("uint _lr_ps = (simd_lane / 16u) * 4u + (simd_lane % 8u) / 2u;")
            for i in range(reg):
                self._emit(
                    f"float _alpha_ps_{i} = {alpha_buf}[{sg_m} * {TM}u + {i * 8}u + _lr_ps];"
                )
                for j in range(reg):
                    self._emit(
                        f"{acc_pfx}_{i}_{j}.thread_elements()[0] *= _alpha_ps_{i};"
                    )
                    self._emit(
                        f"{acc_pfx}_{i}_{j}.thread_elements()[1] *= _alpha_ps_{i};"
                    )

        # Constant-trip contraction loop. At small K (attention QK = 128 → 16
        # iterations; GEMM inner tiles) the Metal compiler keeps the loop and
        # pays the `_dk < K` comparison every iteration (~10% of the QK MMA at
        # depth, per the GPU profiler). Force a full unroll when the trip count
        # is small; large-K dots keep the loop to avoid code-size blowup.
        if K_dim <= 128:
            self._emit("#pragma clang loop unroll(full)")
        self._emit(f"for (uint _dk = 0; _dk < {K_dim}u; _dk += 8) {{")
        self._indent += 1

        # A/B operands match shared memory dtype; accumulators may differ (mixed precision).
        # Per-buffer dtype: each operand's matrix element type comes from its
        # OWN shmem buffer's dtype (not the kernel-global). Apple Silicon's
        # `simdgroup_multiply_accumulate(acc, a, b, acc)` requires both A
        # and B inputs to share the same matrix element type, so the planner
        # gates per-buffer dtype assignment to enforce that — this code only
        # consumes that decision.
        lhs_op_dtype = self._buf_shmem_dtype(lhs_buf) if lhs_buf != "???" else self._shmem_dtype
        rhs_op_dtype = self._buf_shmem_dtype(rhs_buf) if rhs_buf != "???" else self._shmem_dtype
        lhs_db = self._db_shmem_offsets.get(lhs_buf, "") if self._db_shmem_offsets else ""
        rhs_db = self._db_shmem_offsets.get(rhs_buf, "") if self._db_shmem_offsets else ""
        lhs_off = f"{lhs_db} + " if lhs_db else ""
        rhs_off = f"{rhs_db} + " if rhs_db else ""
        for i in range(reg):
            self._emit(f"simdgroup_matrix<{lhs_op_dtype}, 8, 8> _a{i};")
            if op.transpose_lhs:
                # lhs storage is (K, M); load as (M, K) via transpose flag.
                # shmem tile shape is (K, M) with stride=M, so the read
                # origin is _dk rows × (sg_m * TM + i*8) cols.
                self._emit(
                    f"simdgroup_load(_a{i}, &{lhs_buf}[{lhs_off}_dk * {lhs_stride}u + {sg_m} * {TM}u + {i * 8}u], {lhs_stride}u, ulong2(0,0), true);"
                )
            else:
                self._emit(
                    f"simdgroup_load(_a{i}, &{lhs_buf}[{lhs_off}({sg_m} * {TM}u + {i * 8}u) * {lhs_stride}u + _dk], {lhs_stride}u);"
                )

        # Device-direct RHS (Flash-Attention K/V): stream 8x8 blocks straight
        # from device memory into the MMA — no shmem tile, no barrier. The
        # buffer is its own dtype (K/V are half*), addressed in element units:
        #   addr = base + (row_start + block_row)*row_stride + col_start + col
        # transpose (K): block_row = N-tile, col = _dk (contraction).
        # plain    (V): block_row = _dk (contraction), col = N-tile (output).
        rhs_dev = self._device_operands.get(rhs.name) if rhs_loc.kind == "device" else None
        if rhs_dev is not None:
            d_ptr, d_base, d_rs, d_stride, d_cs = rhs_dev
            d_dt = self._buffer_dtypes.get(d_ptr, self._shmem_dtype)
            d_base_pfx = f"({d_base}) + " if d_base not in ("0", 0) else ""
            d_cs_pfx = f"{d_cs} + " if d_cs not in ("0", 0) else ""
        for j in range(reg):
            if rhs_dev is not None:
                self._emit(f"simdgroup_matrix<{d_dt}, 8, 8> _b{j};")
                if op.transpose_rhs:
                    self._emit(
                        f"simdgroup_load(_b{j}, &{d_ptr}[{d_base_pfx}"
                        f"(uint({d_rs}) + {sg_n} * {TN}u + {j * 8}u) * {d_stride}u "
                        f"+ {d_cs_pfx}_dk], {d_stride}u, ulong2(0,0), true);"
                    )
                else:
                    self._emit(
                        f"simdgroup_load(_b{j}, &{d_ptr}[{d_base_pfx}"
                        f"(uint({d_rs}) + _dk) * {d_stride}u "
                        f"+ {d_cs_pfx}{sg_n} * {TN}u + {j * 8}u], {d_stride}u);"
                    )
                continue
            self._emit(f"simdgroup_matrix<{rhs_op_dtype}, 8, 8> _b{j};")
            if op.transpose_rhs:
                self._emit(
                    f"simdgroup_load(_b{j}, &{rhs_buf}[{rhs_off}({sg_n} * {TN}u + {j * 8}u) * {rhs_stride}u + _dk], {rhs_stride}u, ulong2(0,0), true);"
                )
            else:
                self._emit(
                    f"simdgroup_load(_b{j}, &{rhs_buf}[{rhs_off}_dk * {rhs_stride}u + {sg_n} * {TN}u + {j * 8}u], {rhs_stride}u);"
                )

        for i in range(reg):
            for j in range(reg):
                self._emit(
                    f"simdgroup_multiply_accumulate({acc_pfx}_{i}_{j}, _a{i}, _b{j}, {acc_pfx}_{i}_{j});"
                )

        self._indent -= 1
        self._emit("}")

        if is_persistent:
            # Accumulators stay in registers — no shared store, no barrier
            self._val_loc[result.name] = MMA
        else:
            res_plan = self._shmem_plan.get(result.name)
            if res_plan:
                res_buf, _, _, res_stride = res_plan
                # Mixed precision: acc may be float while shmem is half —
                # simdgroup_store requires matching types, so spill per-lane
                # via thread_elements() with an explicit cast.
                if self._acc_dtype == self._shmem_dtype:
                    for i in range(reg):
                        for j in range(reg):
                            self._emit(
                                f"simdgroup_store({acc_pfx}_{i}_{j}, &{res_buf}[({sg_m} * {TM}u + {i * 8}u) * {res_stride}u + {sg_n} * {TN}u + {j * 8}u], {res_stride}u);"
                            )
                else:
                    # Still inside `if (simd_gid < n_sg)` guard from _emit_mma.
                    self._emit("uint _lr = (simd_lane / 16u) * 4u + (simd_lane % 8u) / 2u;")
                    self._emit("uint _lc = ((simd_lane / 8u) % 2u) * 4u + (simd_lane % 2u) * 2u;")
                    for i in range(reg):
                        for j in range(reg):
                            tag = f"_mm_{result.name}_{i}_{j}"
                            self._emit(
                                f"thread auto & {tag} = {acc_pfx}_{i}_{j}.thread_elements();"
                            )
                            self._emit(
                                f"{res_buf}[(({sg_m}) * {TM}u + {i * 8}u + _lr) * {res_stride}u + "
                                f"({sg_n}) * {TN}u + {j * 8}u + _lc + 0u] = "
                                f"{self._shmem_dtype}({tag}[0]);"
                            )
                            self._emit(
                                f"{res_buf}[(({sg_m}) * {TM}u + {i * 8}u + _lr) * {res_stride}u + "
                                f"({sg_n}) * {TN}u + {j * 8}u + _lc + 1u] = "
                                f"{self._shmem_dtype}({tag}[1]);"
                            )
                self._val_loc[result.name] = ValLoc("shared", res_buf, res_stride)
            else:
                self._val_loc[result.name] = MMA

        self._indent -= 1
        self._emit("}")
        # Defer the post-MMA barrier: lhs/rhs reads + result write all need
        # visibility before the next op that touches the same buffers (next-
        # iter coop load reusing _s2, downstream Dot reading _s4). Tracking
        # those buffers in `_buffers_since_barrier` lets unrelated ops in
        # between (a coop load to a fresh buffer like the mask → _s5) skip
        # the barrier and let the eventual reader absorb it. Suppressed in
        # double-buffer mode where the loop manages synchronization.
        if not self._db_shmem_offsets:
            self._pending_tg_barrier = True
            # Reads of lhs/rhs go to `_buffers_read` so a later WRITE to
            # the same buffer (next-iter coop load reusing _s2) flushes via
            # WAR — but a later READ (chain consuming the same input as a
            # prior persistent MMA) does not, since RAR is silent.
            for buf in (lhs_buf, rhs_buf):
                if buf and buf != "???":
                    self._buffers_read.add(buf)
            # Result write goes to `_buffers_since_barrier` so a later READ
            # (chain reading the dot output) flushes via RAW. Persistent MMA
            # writes only registers — nothing to track for shmem visibility.
            res_buf_name = self._val_loc.get(result.name)
            if res_buf_name and res_buf_name.kind == "shared":
                self._buffers_since_barrier.add(res_buf_name.name)

    def _materialize_mma_to_shmem(self, value: "TileValue"):
        """Store persistent MMA accumulators to shared memory.

        Used when a persistent MMA result (from an inner ForLoop) is consumed
        as input to another Dot (e.g., chained GEMM intermediate).
        """
        # Find the Dot that produced this value
        dot_op = self._op_map.get(value.name)
        if not isinstance(dot_op, Dot) or dot_op.acc is None:
            return

        M_dim = dot_op.acc.shape[0]
        N_dim = dot_op.acc.shape[1]
        reg = pick_dot_reg(M_dim, N_dim, override=self._reg_override, target_n_sg=self._n_sg_target)
        TM = reg * 8
        TN = reg * 8
        sg_rows = M_dim // TM
        sg_cols = N_dim // TN
        n_sg = sg_rows * sg_cols
        acc_pfx = f"_acc_{dot_op.acc.name}"
        sg_m = f"_sg_m_{dot_op.acc.name}"
        sg_n = f"_sg_n_{dot_op.acc.name}"

        # Find shmem buffer (auto-allocated by _auto_shmem_plan for Dot results)
        shmem_info = self._shmem_plan.get(value.name)
        if not shmem_info:
            return
        buf_name, _, _, stride = shmem_info

        self._emit("// Materialize persistent MMA → shared memory")
        # simdgroup_store requires matching dtype; mixed-precision plans have
        # acc=float but shmem=half, so fall back to per-lane writes through
        # thread_elements() with an explicit cast when types differ.
        if self._acc_dtype == self._shmem_dtype:
            for i in range(reg):
                for j in range(reg):
                    self._emit(
                        f"simdgroup_store({acc_pfx}_{i}_{j}, "
                        f"&{buf_name}[({sg_m} * {TM}u + {i * 8}u) * {stride}u + "
                        f"{sg_n} * {TN}u + {j * 8}u], {stride}u);"
                    )
        else:
            self._emit(f"if (simd_gid < {n_sg}u) {{")
            self._emit("    uint _lr = (simd_lane / 16u) * 4u + (simd_lane % 8u) / 2u;")
            self._emit("    uint _lc = ((simd_lane / 8u) % 2u) * 4u + (simd_lane % 2u) * 2u;")
            for i in range(reg):
                for j in range(reg):
                    tag = f"_mm_{dot_op.acc.name}_{i}_{j}"
                    self._emit(f"    thread auto & {tag} = {acc_pfx}_{i}_{j}.thread_elements();")
                    self._emit(
                        f"    {buf_name}[(({sg_m}) * {TM}u + {i * 8}u + _lr) * {stride}u + "
                        f"({sg_n}) * {TN}u + {j * 8}u + _lc + 0u] = "
                        f"{self._shmem_dtype}({tag}[0]);"
                    )
                    self._emit(
                        f"    {buf_name}[(({sg_m}) * {TM}u + {i * 8}u + _lr) * {stride}u + "
                        f"({sg_n}) * {TN}u + {j * 8}u + _lc + 1u] = "
                        f"{self._shmem_dtype}({tag}[1]);"
                    )
            self._emit("}")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        self._val_loc[value.name] = ValLoc("shared", buf_name, stride)

    def _emit_persistent_mma_store(self, op: Store):
        """Store persistent MMA accumulators directly to device memory.

        Uses simdgroup_store to threadgroup staging per 8x8 tile, then each lane
        writes its 2 elements to device. Tiles are fully unrolled (no runtime branching).
        """
        # Find the Dot op that produced this value (via op_map)
        dot_op = self._op_map.get(op.value.name) if op.value else None
        if not isinstance(dot_op, Dot) or dot_op.acc is None:
            return

        M_dim = dot_op.acc.shape[0]
        N_dim = dot_op.acc.shape[1]
        reg = pick_dot_reg(M_dim, N_dim, override=self._reg_override, target_n_sg=self._n_sg_target)
        TM = reg * 8
        TN = reg * 8
        sg_rows = M_dim // TM
        sg_cols = N_dim // TN
        n_sg = sg_rows * sg_cols
        acc_pfx = f"_acc_{dot_op.acc.name}"
        ptr_name = op.ptr.name if op.ptr else "C"
        store_idx = self._pmma_store_counter
        self._pmma_store_counter += 1
        # Reuse a cooperative load buffer as staging (it's dead after the K-loop).
        # Only when accumulator dtype matches shmem dtype to avoid type conflicts.

        row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
        addr_base = f"({base_offset}) + " if base_offset != "0" else ""

        # Extract bounds from Store mask
        store_row_bound, store_col_bound = None, None
        if op.mask:
            store_row_bound, store_col_bound = self._extract_load_bounds(op)

        # Scatter-accumulate epilogue (reduce="add"): atomic_fetch_add each element into
        # the destination instead of overwriting (MoE grouped-down fused combine).
        reduce_add = op.reduce == "add"
        # Gathered store row: the store's row index is itself a buffer Load (e.g. the MoE
        # down GEMM scatters tile rows to Y[TOK_ST[rm]]). Same detection as the gathered
        # cooperative load; affine row indices are never a Load.
        store_gather = (
            self._gather_index_load(op.row_indices) if op.row_indices is not None else None
        )

        # Determine bounds expressions (elide when tile divides problem size)
        ce = self.func.constexpr_values

        def _resolve_store_bound(bvar):
            if bvar is None:
                return None
            v = ce.get(bvar)
            if isinstance(v, int):
                return v
            try:
                return int(bvar)
            except (ValueError, TypeError):
                return None

        if self._row_bound:
            rb_val = _resolve_store_bound(self._row_bound)
            rb = None if rb_val is not None and rb_val % M_dim == 0 else str(self._row_bound)
            cb_val = _resolve_store_bound(store_col_bound or "N")
            cb = None if cb_val is not None and cb_val % N_dim == 0 else str(store_col_bound or "N")
        elif store_row_bound or store_col_bound:
            rb_val = _resolve_store_bound(store_row_bound)
            rb = (
                None
                if rb_val is not None and rb_val % M_dim == 0
                else str(store_row_bound or M_dim)
            )
            cb_val = _resolve_store_bound(store_col_bound)
            cb = (
                None
                if cb_val is not None and cb_val % N_dim == 0
                else str(store_col_bound or N_dim)
            )
        else:
            rb = None
            cb = None

        # A gathered store row is bounded by the MASK's row compare (e.g. TOK_ST sentinel
        # < T_ROWS), NOT by tile divisibility — never elide it, or pad rows scatter into
        # (or past) live Y rows.
        if store_gather is not None and store_row_bound:
            rb = str(store_row_bound)

        sg_m = f"_sg_m_{dot_op.acc.name}"
        sg_n = f"_sg_n_{dot_op.acc.name}"
        out_dt = self._buffer_dtypes.get(ptr_name, self._acc_dtype)

        # FA-2 forward post-loop scale: populate per-row (1/l) buffer before
        # the simd_gid guard so all threads (incl. tid<M ones) participate.
        if op.acc_post_scale is not None:
            post_buf = self._acc_post_scale_buf_name(dot_op)
            post_expr = self._get(op.acc_post_scale)
            self._emit(f"if (_row < {M_dim}u) {{ {post_buf}[_row] = {post_expr}; }}")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Direct register → global store using simdgroup_matrix::thread_elements().
        # Each lane owns 2 elements of the 8x8 tile (Apple Silicon layout):
        #   r = (lane / 16) * 4 + (lane % 8) / 2
        #   c_base = ((lane / 8) % 2) * 4 + (lane % 2) * 2
        #   el[0] is at (r, c_base), el[1] is at (r, c_base + 1)
        # Avoids a simdgroup_store → shmem → per-lane re-read round-trip
        # and the transient staging allocation it requires.
        guard_parts = []
        if rb is not None:
            guard_parts.append(f"_gm < {rb}")
        if cb is not None:
            guard_parts.append(f"_gn < {cb}")
        guard_expr = " && ".join(guard_parts) if guard_parts else None

        # Wrap in a block scope so the _lr/_lc helpers don't collide when
        # multiple MMA stores (e.g., branchy epilogues with a main output
        # plus extra_branches) share a kernel.
        # Guard by simd_gid < n_sg: idle simdgroups (NUM_THREADS exceeds
        # 32*n_sg when sized for cooperative-load parallelism) would
        # otherwise stream zero-initialised accumulators into global memory,
        # clobbering cells owned by neighbouring threadgroups.
        self._emit(f"if (simd_gid < {n_sg}u) {{")
        self._emit("    uint _lr = (simd_lane / 16u) * 4u + (simd_lane % 8u) / 2u;")
        self._emit("    uint _lc = ((simd_lane / 8u) % 2u) * 4u + (simd_lane % 2u) * 2u;")

        # Classify transform extras for vectorized preloading.
        # The two thread_elements per lane sit at adjacent columns (_gn,
        # _gn+1) at the same row (_gm), so extras whose index stride in the
        # column dim is 1 can be loaded as a single float2; extras that
        # depend only on the row (or are scalar) can be loaded once as a
        # float.  Modular/scatter extras fall back to per-element scalar.
        vec_xform_types = (
            IdentityTransform,
            RowBroadcastTransform,
            StridedTransform,
        )
        scalar_xform_types = (
            ColumnBroadcastTransform,
            ScalarBroadcastTransform,
        )
        extras_map = op.transform_extras or {}
        vec_extras: list[str] = []
        scalar_extras: list[str] = []
        for ename, xf in extras_map.items():
            if isinstance(xf, vec_xform_types):
                vec_extras.append(ename)
            elif isinstance(xf, scalar_xform_types):
                scalar_extras.append(ename)

        # Check once whether the output uses a scatter (view+permute) layout —
        # non-contiguous addresses for adjacent columns rule out a vec store.
        cv_outer = self.func.constexpr_values
        _nd_s_out = cv_outer.get(f"_{ptr_name}_nd_shape")
        _nd_st_out = cv_outer.get(f"_{ptr_name}_nd_strides")
        _has_scatter_out = _nd_s_out is not None and _nd_st_out is not None and len(_nd_s_out) > 2
        # Vec-store fast path: aligned tile, no column slice, contiguous
        # output layout. Dtype mismatch (acc=float, out=bfloat) is fine —
        # Metal's vec ctor auto-narrows: `bfloat2(float, float)` truncates
        # each element. The conversion lands in one packed device store
        # instead of two scalar stores.
        _can_vec_store = (
            not guard_expr
            and op.col_slice is None
            and not _has_scatter_out
            and not reduce_add          # atomics are per-element
            and store_gather is None    # gathered rows need the per-element guard path
        )

        for i in range(reg):
            for j in range(reg):
                el_name = f"{acc_pfx}_{i}_{j}_el_{store_idx}"
                self._emit(f"    thread auto & {el_name} = {acc_pfx}_{i}_{j}.thread_elements();")
                # FA-2 post-scale: each lane multiplies its 2 elements by
                # (1/l)[row] in-place. Both el[0] and el[1] are at the same
                # row (different cols), so a single lookup per (i, j) tile
                # suffices. Modifies the simdgroup_matrix register state in
                # place — safe because there's only one persistent-acc store
                # in the FA-2 forward.
                if op.acc_post_scale is not None:
                    post_buf = self._acc_post_scale_buf_name(dot_op)
                    self._emit(
                        f"    {{ float _ps_{i}_{j} = {post_buf}[{sg_m} * {TM}u + {i * 8}u + _lr];"
                    )
                    self._emit(
                        f"      {el_name}[0] = {self._acc_dtype}({el_name}[0] * _ps_{i}_{j});"
                    )
                    self._emit(
                        f"      {el_name}[1] = {self._acc_dtype}({el_name}[1] * _ps_{i}_{j}); }}"
                    )
                # Per-tile (i,j) scope block so _gm/_gn_base and preload
                # variables don't collide across tiles.
                self._emit("    {")
                if store_gather is not None:
                    # Gathered store row: the destination row comes from the index
                    # buffer at the TILE row (idx_base + tile-local row), not from an
                    # affine row_start. Pad rows hold a sentinel that fails the
                    # `_gm < rb` guard below.
                    _sg_idx_base = self._extract_range_base(store_gather.offsets)
                    self._emit(
                        f"        uint _srm = uint({_sg_idx_base}) + {sg_m} * {TM}u + {i * 8}u + _lr;"
                    )
                    self._emit(
                        f"        uint _gm = uint(int({store_gather.ptr.name}[_srm]));"
                    )
                else:
                    self._emit(
                        f"        uint _gm = uint({row_start}) + {sg_m} * {TM}u + {i * 8}u + _lr;"
                    )
                self._emit(
                    f"        uint _gn_base = uint({col_start}) + {sg_n} * {TN}u + {j * 8}u + _lc;"
                )

                # Per-element preload map: extra_name -> MSL expression per e.
                preload_per_elem: list[dict[str, str]] = [{}, {}]
                # Register-accumulator extras: a sibling persistent acc (same
                # geometry) read per-lane from its OWN thread_elements() — no
                # device load. Lets a multi-accumulator elementwise epilogue
                # (dot_q4_k_silu's silu(gate)*up) run register-resident.
                for ex_leaf, ex_acc in (op.acc_extras or {}).items():
                    ex_el = f"_accx_{ex_acc}_{i}_{j}_{store_idx}"
                    self._emit(
                        f"        thread auto & {ex_el} = _acc_{ex_acc}_{i}_{j}.thread_elements();"
                    )
                    preload_per_elem[0][ex_leaf] = f"{ex_el}[0]"
                    preload_per_elem[1][ex_leaf] = f"{ex_el}[1]"
                for ename in vec_extras:
                    xf = extras_map[ename]
                    idx0 = xf.tile_2d("_gm", "_gn_base")
                    pv = f"_pl_{ename}_{i}_{j}_{store_idx}"
                    # Use the extra's actual buffer dtype for the vector
                    # reinterpret — half* aliased as float2* misreads 4 halves
                    # as 2 bogus floats and produces NaN/Inf residual adds.
                    etype = self._buffer_dtypes.get(ename, self._acc_dtype)
                    self._emit(
                        f"        {etype}2 {pv} = *(device const {etype}2*)(&{ename}[{idx0}]);"
                    )
                    preload_per_elem[0][ename] = f"{pv}[0]"
                    preload_per_elem[1][ename] = f"{pv}[1]"
                for ename in scalar_extras:
                    xf = extras_map[ename]
                    idx0 = xf.tile_2d("_gm", "_gn_base")
                    pv = f"_pl_{ename}_{i}_{j}_{store_idx}"
                    etype = self._buffer_dtypes.get(ename, self._acc_dtype)
                    self._emit(f"        {etype} {pv} = {ename}[{idx0}];")
                    preload_per_elem[0][ename] = pv
                    preload_per_elem[1][ename] = pv

                if _can_vec_store and not op.transform:
                    # No epilogue: the thread_elements vec goes straight to
                    # global as a single packed store. Per-element cast to
                    # out_dt in the vec ctor — `bfloat2(float, float)` is
                    # rejected by Metal but `bfloat2(bfloat(f), bfloat(f))`
                    # narrows correctly into a single 4-byte device write.
                    self._emit(
                        f"        *(device {out_dt}2*)(&{ptr_name}[{addr_base}_gm * {row_stride}u + _gn_base]) = {out_dt}2({out_dt}({el_name}[0]), {out_dt}({el_name}[1]));"
                    )
                    self._emit("    }")
                    continue

                # When the fusion engine flagged this epilogue as tile_consumer
                # (e.g., gemm + same_shape_residual_add), eager rounds the
                # gemm output to the store dtype before the binop. Match that
                # rounding so alloy's fused single-dispatch kernel produces
                # eager-equivalent output instead of f32-precise-then-rounded.
                _round_acc = (
                    op.transform
                    and op.round_acc_for_eager
                    and out_dt in ("bfloat", "half")
                    and out_dt != self._acc_dtype
                )
                if _can_vec_store and op.transform:
                    # Compute both elements through the epilogue chain into
                    # local scalars, then pack into a float2 and do a single
                    # vectorized store.
                    self._emit(f"        {self._acc_dtype} _ev0_{i}_{j}, _ev1_{i}_{j};")
                    for e in range(2):
                        self._emit("        {")
                        self._emit(f"            uint _gn = _gn_base + {e}u;")
                        self._emit(f"            {self._acc_dtype} _ev = {el_name}[{e}];")
                        if _round_acc:
                            self._emit(f"            _ev = {self._acc_dtype}({out_dt}(_ev));")
                        preload_e = preload_per_elem[e] if (vec_extras or scalar_extras or op.acc_extras) else None
                        self._emit_store_transform(
                            op.transform,
                            self._acc_dtype,
                            "_ev",
                            row_stride=op.row_stride,
                            chain_source_name=op.transform_source_name,
                            extra_transforms=op.transform_extras,
                            preload_extras=preload_e,
                        )
                        self._emit(f"            _ev{e}_{i}_{j} = _ev;")
                        self._emit("        }")
                    self._emit(
                        f"        *(device {out_dt}2*)(&{ptr_name}[{addr_base}_gm * {row_stride}u + _gn_base]) = {out_dt}2({out_dt}(_ev0_{i}_{j}), {out_dt}(_ev1_{i}_{j}));"
                    )
                    self._emit("    }")
                    continue

                # Fallback: per-element scalar stores (tile has bounds guard,
                # column slice, scatter output, or dtype promotion).
                for e in range(2):
                    self._emit("        {")
                    self._emit(f"            uint _gn = _gn_base + {e}u;")
                    if guard_expr:
                        self._emit(f"            if ({guard_expr}) {{")
                        indent = "                "
                    else:
                        indent = "            "
                    preload_e = preload_per_elem[e] if (vec_extras or scalar_extras or op.acc_extras) else None
                    if op.transform:
                        if op.col_slice is not None:
                            cs = op.col_slice
                            self._emit(
                                f"{indent}if (_gn >= {cs.col_start}u && _gn < {cs.col_end}u) {{"
                            )
                            self._emit(f"{indent}    {self._acc_dtype} _ev = {el_name}[{e}];")
                            if _round_acc:
                                self._emit(f"{indent}    _ev = {self._acc_dtype}({out_dt}(_ev));")
                            self._emit_store_transform(
                                op.transform,
                                self._acc_dtype,
                                "_ev",
                                row_stride=op.row_stride,
                                chain_source_name=op.transform_source_name,
                                extra_transforms=op.transform_extras,
                                preload_extras=preload_e,
                            )
                            self._emit(
                                f"{indent}    {ptr_name}[_gm * {cs.out_stride}u + (_gn - {cs.col_start}u)] = {out_dt}(_ev);"
                            )
                            self._emit(f"{indent}}}")
                        else:
                            self._emit(f"{indent}{self._acc_dtype} _ev = {el_name}[{e}];")
                            if _round_acc:
                                self._emit(f"{indent}_ev = {self._acc_dtype}({out_dt}(_ev));")
                            self._emit_store_transform(
                                op.transform,
                                self._acc_dtype,
                                "_ev",
                                row_stride=op.row_stride,
                                chain_source_name=op.transform_source_name,
                                extra_transforms=op.transform_extras,
                                preload_extras=preload_e,
                            )
                            _mma_store_addr = f"{addr_base}_gm * {row_stride}u + _gn"
                            if _has_scatter_out:
                                _mma_store_addr = f"{addr_base}{ScatterTransform(nd_shape=_nd_s_out, nd_strides=_nd_st_out).tile_2d('_gm', '_gn')}"
                            if reduce_add:
                                self._emit(
                                    f"{indent}atomic_fetch_add_explicit(&{ptr_name}[{_mma_store_addr}], float(_ev), memory_order_relaxed);"
                                )
                            else:
                                self._emit(f"{indent}{ptr_name}[{_mma_store_addr}] = {out_dt}(_ev);")
                    elif reduce_add:
                        self._emit(
                            f"{indent}atomic_fetch_add_explicit(&{ptr_name}[{addr_base}_gm * {row_stride}u + _gn], float({el_name}[{e}]), memory_order_relaxed);"
                        )
                    else:
                        self._emit(
                            f"{indent}{ptr_name}[{addr_base}_gm * {row_stride}u + _gn] = {out_dt}({el_name}[{e}]);"
                        )
                    if guard_expr:
                        self._emit("            }")
                    self._emit("        }")
                self._emit("    }")
        self._emit("}")  # close outer scope for _lr/_lc helpers

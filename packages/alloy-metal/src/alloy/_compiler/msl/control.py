"""Control-flow MSL emitter methods."""

from __future__ import annotations

from alloy._compiler.tile_ir import (
    BinOp,
    Constant,
    Dot,
    FlowControl,
    ForLoop,
    IfElse,
    Load,
    TileValue,
    WhileLoop,
    Zeros,
)
from alloy._compiler.msl.context import PER_THREAD, PERSISTENT_MMA


class ControlEmitterMixin:
    def _emit_carried_inits(self, carried, skip_mma=False):
        """Declare carried variables before a loop."""
        for init_val, _ in carried:
            if skip_mma and init_val.name in self._pmma_acc_names:
                continue
            loc = self._val_loc.get(init_val.name, PER_THREAD)
            # "shared" carried tiles need no scalar init: the value already
            # aliases its threadgroup buffer (the load/cast that produced it), and
            # the loop-tail cooperative copy writes each iteration's update back
            # into that same buffer. A `float name = <buffer>` here would both
            # mis-type the 2D tile and strip its shared location.
            if loc.kind not in ("local_array", "shared"):
                init_expr = self._get(init_val)
                if init_expr != init_val.name:
                    # Init is an alias to another variable — declare a fresh copy
                    # to avoid aliasing (e.g. `acc = x; for: acc = acc + x`
                    # would modify x without this).
                    self._emit(f"{self._acc_dtype} {init_val.name} = {init_expr};")
                    self._exprs[init_val.name] = init_val.name

    def _emit_simultaneous_updates(self, temps: list[tuple[str, str]], prefix: str = "_carry"):
        """Emit simultaneous carried variable updates via temporaries."""
        if len(temps) > 1:
            for i, (name, expr) in enumerate(temps):
                self._emit(f"auto {prefix}{i} = {expr};")
            for i, (name, _) in enumerate(temps):
                self._emit(f"{name} = {prefix}{i};")
        else:
            for name, expr in temps:
                self._emit(f"{name} = {expr};")

    def _emit_carried_shmem_copy(self, init_val, init_loc, final_loc) -> None:
        """Write a loop-carried 2D shmem tile's per-iteration update back into its
        carry buffer (cooperative, all threads), bracketed by barriers.

        `init_loc` / `final_loc` are the `ValLoc`s (buffer name + row stride) of
        the carry buffer and this iteration's update; both describe the same
        (rows, cols) tile. The leading barrier waits for every thread to finish
        producing the update (a GEMM result) before it is read; the trailing one
        publishes the new carry buffer before the next iteration's GEMM reads it.
        """
        rows, cols = init_val.shape
        dst, dst_stride = init_loc.name, (init_loc.stride or cols)
        src, src_stride = final_loc.name, (final_loc.stride or cols)
        threads = self._threads
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._emit(f"for (uint _cs = tid; _cs < {rows * cols}u; _cs += {threads}u) {{")
        self._indent += 1
        self._emit(f"uint _csr = _cs / {cols}u; uint _csc = _cs % {cols}u;")
        self._emit(
            f"{dst}[_csr * {dst_stride}u + _csc] = {src}[_csr * {src_stride}u + _csc];"
        )
        self._indent -= 1
        self._emit("}")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def _clear_loop_context(self):
        """Reset loop state after a ForLoop or WhileLoop."""
        self._loop_var = None
        self._loop_step = None
        self._carried_inits = set()
        self._carried_finals = set()
        self._carried_increments = {}
        self._mask_expr = None

    def _compute_carried_increment(self, init_val: TileValue, final_val: TileValue) -> int | None:
        """Extract integer increment from carried var: final = init + C."""
        final_op = self._op_map.get(final_val.name)
        if not isinstance(final_op, BinOp) or final_op.op != "add":
            return None
        for lhs_v, rhs_v in [(final_op.lhs, final_op.rhs), (final_op.rhs, final_op.lhs)]:
            if lhs_v.name == init_val.name:
                other_op = self._op_map.get(rhs_v.name)
                if isinstance(other_op, Constant) and isinstance(
                    other_op.value, (int, float, bool)
                ):
                    return int(other_op.value)
        return None

    # --- ForLoop with carried values ---

    def _emit_for_loop(self, op: ForLoop):
        # Double-buffer path: peel first iteration, interleave load/compute
        if self._double_buffer and self._can_double_buffer(op):
            self._emit_double_buffered_loop(op)
            return

        var = op.var
        self._exprs[var] = var

        self._carried_finals = {final_val.name for _, final_val in op.carried}
        self._emit_carried_inits(op.carried, skip_mma=True)

        # Hoist Zeros declarations from loop body to before the loop.
        # Without this, accumulators declared inside the loop body are scoped
        # to the loop and unavailable for inter-loop carried assignments.
        for body_op in op.body:
            if isinstance(body_op, Zeros) and len(body_op.shape) == 1:
                name = body_op.result.name
                # Declare at outer scope if not already declared here or above
                if self._decl_scope.get(name, self._indent + 1) > self._indent:
                    self._emit(f"{self._acc_dtype} {name};")
                    self._decl_scope[name] = self._indent

        # Set loop context for cooperative loads inside K-loops
        self._loop_var = var
        self._loop_step = op.step
        self._carried_inits = {init_val.name for init_val, _ in op.carried}
        self._carried_increments = {}
        for init_val, final_val in op.carried:
            inc = self._compute_carried_increment(init_val, final_val)
            if inc is not None:
                self._carried_increments[init_val.name] = inc

        # Use signed int when loop bounds are negative
        needs_signed = (
            isinstance(op.start, (int, float))
            and op.start < 0
            or isinstance(op.end, (int, float))
            and op.end < 0
        )
        loop_type = "int" if needs_signed else "uint"

        def _bound_expr(v):
            if isinstance(v, TileValue):
                return f"({loop_type})({self._get(v)})"
            return str(v) if needs_signed else f"{v}u"

        start_e = _bound_expr(op.start)
        end_e = _bound_expr(op.end)
        step_e = _bound_expr(op.step)

        # When loop has negative bounds, cast program_id (uint) expressions
        # to int inside this loop to avoid unsigned wrap on mixed arithmetic.
        if needs_signed:
            for pid_name, pid_expr in list(self._exprs.items()):
                if pid_expr.startswith("gid."):
                    self._exprs[pid_name] = f"(int)({pid_expr})"

        # Causal attention: dynamic loop bound
        causal_bound = self._config.get("causal_n_kv")
        if causal_bound and self._config.get("causal"):
            self._emit(f"{loop_type} _n_kv = {causal_bound};")
            self._emit(f"for ({loop_type} {var} = {start_e}; {var} < _n_kv; {var} += {step_e}) {{")
        else:
            self._emit(
                f"for ({loop_type} {var} = {start_e}; {var} < {end_e}; {var} += {step_e}) {{"
            )
        self._indent += 1
        # Loop preamble (e.g. _j = _jb * BLOCK_N)
        for line in self._config.get("loop_preamble", []):
            self._emit(line)

        self._emit_ops(op.body)

        # Carried updates — simultaneous via temporaries to avoid order-dependent bugs
        temps = []
        for init_val, final_val in op.carried:
            if init_val.name in self._scalar_pmma_acc_names:
                acc_loc = self._val_loc.get(init_val.name, PER_THREAD)
                self._val_loc[final_val.name] = acc_loc
                self._exprs[final_val.name] = self._exprs.get(init_val.name, init_val.name)
                continue
            if init_val.name in self._pmma_acc_names:
                self._val_loc[final_val.name] = PERSISTENT_MMA
                continue
            if final_val.name != init_val.name:
                final_loc = self._val_loc.get(final_val.name, PER_THREAD)
                init_loc = self._val_loc.get(init_val.name, PER_THREAD)
                if init_loc.kind == "shared" and final_loc.kind == "shared":
                    # Loop-carried 2D shmem tile (e.g. DeltaNet's cross-chunk
                    # `state`): cooperatively copy this iteration's update back
                    # into the carry buffer so the next iteration's GEMM operands
                    # read it, and the post-loop store sees the final value.
                    self._emit_carried_shmem_copy(init_val, init_loc, final_loc)
                elif final_loc.kind == "local_array":
                    self._exprs[final_val.name] = init_val.name
                    self._val_loc[final_val.name] = final_loc
                    D = self._local_arrays.get(init_val.name)
                    if D is not None:
                        self._local_arrays[final_val.name] = D
                else:
                    final_expr = self._get(final_val)
                    if final_expr != init_val.name:
                        temps.append((init_val.name, final_expr))
        self._emit_simultaneous_updates(temps)
        for init_val, final_val in op.carried:
            if final_val.name != init_val.name:
                self._exprs[final_val.name] = init_val.name

        # Back-edge race: if the body leaves shmem "dirty" (row-guarded scalar op
        # writes/reads) or has a deferred cooperative-load barrier, iter N's tail
        # can collide with iter N+1's prologue (e.g. a coop load that overwrites
        # the tile being read). The linear emission order hides this — the body's
        # first op was emitted before the flag was set by the body's last op.
        # Fire a barrier at the end of each iteration to sync the boundary.
        if self._scalar_shmem_dirty or self._pending_tg_barrier:
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._scalar_shmem_dirty = False
            self._pending_tg_barrier = False

        self._indent -= 1
        self._emit("}")

        # Hoist variables declared inside the loop body to the outer scope.
        # Without this, references to loop-scoped variables (e.g., accumulators
        # used by carried updates between sibling ForLoops) produce MSL errors.
        loop_indent = self._indent + 1  # the indent level inside the loop body
        for name, decl_depth in list(self._decl_scope.items()):
            if decl_depth >= loop_indent:
                # This variable was declared inside the loop — it's now out of
                # scope. Mark it as declared at the current (outer) scope so
                # subsequent references are valid. The variable retains its
                # last-iteration value via the MSL auto storage class.
                self._decl_scope[name] = self._indent

        self._clear_loop_context()

    def _emit_while_loop(self, op: WhileLoop):
        """Emit a while loop with runtime condition and carried values."""
        self._emit_carried_inits(op.carried)

        self._emit_ops(op.cond_body)
        cond_expr = self._get(op.cond)

        self._emit(f"while ({cond_expr}) {{")
        self._indent += 1

        self._emit_ops(op.body)

        temps = []
        for init_val, final_val in op.carried:
            if final_val.name != init_val.name:
                final_expr = self._get(final_val)
                if final_expr != init_val.name:
                    temps.append((init_val.name, final_expr))
        self._emit_simultaneous_updates(temps, "_wc")

        self._emit_ops(op.cond_body)

        self._indent -= 1
        self._emit("}")
        self._mask_expr = None

        # After loop: carried final values are the in-place-updated init variables
        for init_val, final_val in op.carried:
            if final_val.name != init_val.name:
                self._exprs[final_val.name] = init_val.name

    def _emit_if_else(self, op: IfElse):
        """Emit runtime if/else block."""
        cond_expr = self._get(op.cond)
        # Compare was consumed by the if-condition, not a store mask
        self._mask_expr = None

        # Pre-declare merge results at outer scope with false-branch value
        for result, _, false_val in op.merges:
            false_expr = self._get(false_val)
            self._emit(f"{self._acc_dtype} {result.name} = {false_expr};")
            self._exprs[result.name] = result.name

        self._emit(f"if ({cond_expr}) {{")
        self._indent += 1
        self._emit_ops(op.body)
        # Assign true-branch values inside the if-block (in scope)
        for result, true_val, _ in op.merges:
            true_expr = self._get(true_val)
            self._emit(f"{result.name} = {true_expr};")
        self._indent -= 1
        if op.orelse:
            self._emit("} else {")
            self._indent += 1
            self._emit_ops(op.orelse)
            # Assign else-branch values
            for result, _, else_val in op.merges:
                else_expr = self._get(else_val)
                self._emit(f"{result.name} = {else_expr};")
            self._indent -= 1
        self._emit("}")

    def _emit_flow_control(self, op: FlowControl):
        """Emit break/continue/return."""
        self._emit(f"{op.kind};")

    def _can_double_buffer(self, op: ForLoop) -> bool:
        """Check if a ForLoop has the Load+Dot pattern for double-buffering."""
        has_load = False
        has_dot = False
        for body_op in op.body:
            if isinstance(body_op, Load) and len(body_op.result.shape) == 2:
                has_load = True
            if isinstance(body_op, Dot):
                has_dot = True
        return has_load and has_dot and op.end > op.step

    def _emit_double_buffered_loop(self, op: ForLoop):
        """Emit a double-buffered K-loop: load next tile while computing current.

        Structure:
          1. Preload first K-tile into buffer half 0
          2. Main loop: load next into half 1, compute from half 0, barrier, swap
          3. Final compute from last loaded half

        Each shared memory buffer is 2x its normal size. Half 0 starts at offset 0,
        half 1 at offset (rows * stride). Per-buffer _db_off tracks the current
        read offset; loads write to (half_size - _db_off).

        Body split: setup_ops (masks/addr, no MSL emitted) run before loads.
        tail_ops (Dot + ptr updates) run after loads to ensure correct pointer state.
        """
        var = op.var
        self._exprs[var] = var
        self._emit_carried_inits(op.carried, skip_mma=True)

        # Set loop context
        self._loop_var = var
        self._loop_step = op.step
        self._carried_inits = {init_val.name for init_val, _ in op.carried}
        self._carried_increments = {}
        for init_val, final_val in op.carried:
            inc = self._compute_carried_increment(init_val, final_val)
            if inc is not None:
                self._carried_increments[init_val.name] = inc

        # Split body: setup_ops (before/between loads), load_ops, tail_ops (after last load)
        load_ops = []
        last_load_idx = -1
        for i, o in enumerate(op.body):
            if isinstance(o, Load) and len(o.result.shape) == 2:
                load_ops.append(o)
                last_load_idx = i

        setup_ops = [
            o
            for i, o in enumerate(op.body)
            if i <= last_load_idx and not (isinstance(o, Load) and len(o.result.shape) == 2)
        ]
        tail_ops = [o for i, o in enumerate(op.body) if i > last_load_idx]

        # Compute single-buffer sizes per shared memory buffer
        db_buf_sizes = {}
        for load_op in load_ops:
            plan = self._shmem_plan.get(load_op.result.name)
            if plan:
                buf_name, rows, cols, stride = plan
                db_buf_sizes[buf_name] = rows * stride

        # Phase 1: Preload first tile (k=start) into buffer half 0
        self._emit("// Double-buffer: preload first K-tile")
        self._emit(f"uint {var} = {op.start}u;")
        self._db_shmem_offsets = None
        for s_op in setup_ops:
            self._emit_op(s_op)
        for load_op in load_ops:
            self._emit_coop_load(load_op)

        # Declare per-buffer offset variables
        for buf_name, half_size in db_buf_sizes.items():
            self._emit(f"uint _db_off{buf_name} = 0u;")

        # Phase 2: Main loop (k = start+step to end)
        self._emit(
            f"for ({var} = {op.start + op.step}u; {var} < {op.end}u; {var} += {op.step}u) {{"
        )
        self._indent += 1

        # Compute next-buffer offsets
        for buf_name, half_size in db_buf_sizes.items():
            self._emit(f"uint _db_nxt{buf_name} = {half_size}u - _db_off{buf_name};")

        # Setup masks/address expressions
        for s_op in setup_ops:
            self._emit_op(s_op)

        if self._async_copy:
            # ASYNC: start loads (returns immediately), compute, then wait.
            # This overlaps the DMA transfer with MMA compute.
            self._db_shmem_offsets = {bn: f"_db_nxt{bn}" for bn in db_buf_sizes}
            for load_op in load_ops:
                self._emit_coop_load(load_op)  # emits async copy without wait

            # Compute MMA from CURRENT half while async copy fills NEXT half
            self._db_shmem_offsets = {bn: f"_db_off{bn}" for bn in db_buf_sizes}
            for body_op in tail_ops:
                self._emit_op(body_op)

            # Wait for all async copies (per-SG: only wait if this SG participated)
            for load_op in load_ops:
                lname = load_op.result.name
                self._emit(f"if (_ev_{lname}) {{")
                self._indent += 1
                self._emit(f"thread _simdgroup_event_t* _dbwait_{lname}[1] = {{_ev_{lname}}};")
                self._emit(f"_alloy_wait_events(1, _dbwait_{lname});")
                self._indent -= 1
                self._emit("}")
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        else:
            # SYNC: load next tile, barrier, then compute current
            self._db_shmem_offsets = {bn: f"_db_nxt{bn}" for bn in db_buf_sizes}
            for load_op in load_ops:
                self._emit_coop_load(load_op)

            # Compute MMA from current half
            self._db_shmem_offsets = {bn: f"_db_off{bn}" for bn in db_buf_sizes}
            for body_op in tail_ops:
                self._emit_op(body_op)

            # Barrier
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Swap offsets
        for buf_name in db_buf_sizes:
            self._emit(f"_db_off{buf_name} = _db_nxt{buf_name};")

        # Carried variable assignments (update ptrs for next iteration)
        for init_val, final_val in op.carried:
            if init_val.name in self._pmma_acc_names:
                continue
            if final_val.name != init_val.name:
                final_loc = self._val_loc.get(final_val.name, PER_THREAD)
                if final_loc.kind == "local_array":
                    self._exprs[final_val.name] = init_val.name
                    self._val_loc[final_val.name] = final_loc
                else:
                    final_expr = self._get(final_val)
                    if final_expr != init_val.name:
                        self._emit(f"{init_val.name} = {final_expr};")
                        self._exprs[final_val.name] = init_val.name

        self._indent -= 1
        self._emit("}")

        # Phase 3: Final MMA from last loaded half
        self._db_shmem_offsets = {bn: f"_db_off{bn}" for bn in db_buf_sizes}
        for body_op in tail_ops:
            self._emit_op(body_op)

        # Clean up
        self._db_shmem_offsets = None
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

        # Final carried variable propagation
        for init_val, final_val in op.carried:
            if init_val.name in self._pmma_acc_names:
                self._val_loc[final_val.name] = PERSISTENT_MMA
                continue
            if final_val.name != init_val.name:
                final_loc = self._val_loc.get(final_val.name, PER_THREAD)
                if final_loc.kind == "local_array":
                    self._exprs[final_val.name] = init_val.name
                    self._val_loc[final_val.name] = final_loc

        self._clear_loop_context()

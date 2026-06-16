"""Generate an HTML visualization of a compiled model's internals.

Usage:
    import alloy
    alloy.visualize(lambda: compiled(**inputs), "whisper.html")
"""

from __future__ import annotations

import contextlib
import gc
import html
import json
import time

from alloy._runtime import _metal_ext
from alloy._runtime.alloy_buffer import AlloyBuffer
from alloy._runtime.metal import default_dispatcher


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1048576:
        return f"{n / 1024:.0f}KB"
    return f"{n / 1048576:.1f}MB"


def visualize(fn, path: str = "alloy_viz.html", name: str = "Model", plans=None, compile_ctx=None):
    """Run fn(), capture compilation artifacts, write interactive HTML.

    `plans` (optional): an explicit list of `CompiledPlan`s to profile instead of
    auto-capturing from `_all_compiled_plans`. Use this for plans that run via the
    eager-compiled pinned path — those replay through C++ `dispatch_plan` and never
    re-enter the torch.compile backend, so the clear-and-recapture path would find
    nothing. When a plan carries a `_last_grid_shrink_updates` recipe, its
    dispatches are profiled at that shrunk launch (the exact grid the run
    dispatched), not the registered max-length grid.

    `compile_ctx` (optional): zero-arg context-manager factory wrapped around the
    FIRST fn() call only — the compile run that follows the dynamo reset. Causal
    captures pass the generator's `plan_compile_window` so run-0 is record-only
    (phantom intermediates, no GPU — a real run-0 holds every intermediate of the
    forward live at once, 100+ GB on a 35B MoE 4096-chunk prefill) and the
    captured plan matches production. The timing runs after it execute real.
    """
    import torch  # scoped: optional dep — visualize only runs against torch models
    from alloy_torch import backend as _bmod  # noqa: PLC0415

    # ── Capture FX graph ──
    # Hook rewrite_fx_graph which is called inside _compile_fx and receives
    # the graph module. This works regardless of how aot_autograd binds the
    # compiler function.
    from alloy_torch.rewrites import pipeline  # noqa: PLC0415

    captured_gm: list[torch.fx.GraphModule | None] = [None]
    orig_rewrite = pipeline.rewrite_fx_graph

    def hooking_rewrite(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        captured_gm[0] = gm
        return orig_rewrite(gm)

    pipeline.rewrite_fx_graph = hooking_rewrite
    # Also patch the module-level reference used by _compile_fx
    import alloy_torch.backend as _bmod_inner  # noqa: PLC0415

    _bmod_inner.rewrite_fx_graph = hooking_rewrite

    # Clear plan list so we only capture plans from this visualization run.
    # Skip when the caller supplied explicit `plans` (e.g. eager-compiled pinned
    # one-shot plans) — those don't re-enter the backend, so clearing would just
    # discard the caller's plans and capture nothing.
    if plans is None and hasattr(_bmod, "_all_compiled_plans"):
        _bmod._all_compiled_plans.clear()
    torch._dynamo.reset()
    with compile_ctx() if compile_ctx is not None else contextlib.nullcontext():
        fn()
    pipeline.rewrite_fx_graph = orig_rewrite
    _bmod_inner.rewrite_fx_graph = orig_rewrite
    gm = captured_gm[0]

    fx_nodes = []
    fx_edges = []
    if gm is not None:
        for node in gm.graph.nodes:
            if node.op == "call_function":
                t = str(node.target)
                # aten.mm.default → mm, aten.native_layer_norm.default → native_layer_norm
                # alloy.rms_norm.default → rms_norm
                parts = t.split(".")
                # Drop namespace prefixes and "default" suffix
                parts = [
                    p
                    for p in parts
                    if p not in ("aten", "alloy", "default", "Tensor", "torch", "ops", "_prims")
                ]
                target = (
                    ".".join(parts) if parts else t.split(".")[-2] if len(t.split(".")) >= 2 else t
                )
            else:
                target = str(node.target)
            if len(target) > 50:
                target = target[:47] + "..."
            shape = ""
            meta = node.meta.get("val")
            if meta is not None and hasattr(meta, "shape"):
                shape = "×".join(str(s) for s in meta.shape)

            cat = "other"
            t = str(node.target)
            if "mm" in t or "addmm" in t:
                cat = "gemm"
            elif "attention" in t or "sdpa" in t or "scaled_dot" in t:
                cat = "attn"
            elif "layer_norm" in t or "rms_norm" in t:
                cat = "norm"
            elif "gelu" in t or "relu" in t or "silu" in t or "sigmoid" in t:
                cat = "activation"
            elif "add" in t and "addmm" not in t:
                cat = "residual"
            elif node.op == "placeholder":
                cat = "input"
            elif node.op == "output":
                cat = "output"
            elif node.op == "get_attr":
                cat = "weight"

            fx_nodes.append(
                {
                    "id": node.name,
                    "op": node.op,
                    "target": target,
                    "shape": shape,
                    "cat": cat,
                }
            )
            for arg in node.args:
                if hasattr(arg, "name"):
                    fx_edges.append({"from": arg.name, "to": node.name})
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if hasattr(a, "name"):
                            fx_edges.append({"from": a.name, "to": node.name})

    # ── Dispatch count + timing ──
    disp = default_dispatcher()
    for _ in range(3):
        fn()
    # _all_compiled_plans only contains final (run 1) plans — no pruning needed.
    c0 = disp.dispatch_count
    fn()
    n_dispatches = disp.dispatch_count - c0

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    wall_ms = times[len(times) // 2]

    # ── Dispatch-level timing (encode + wait breakdown) ──
    for _ in range(5):
        fn()  # warmup dispatch_plan path
    timing_runs = []
    for _ in range(10):
        t0 = time.perf_counter()
        fn()
        wall_sample = (time.perf_counter() - t0) * 1000
        timing_runs.append(wall_sample)
    timing_runs.sort()
    wall_for_breakdown = timing_runs[len(timing_runs) // 2]

    # ── Per-kernel GPU profiling ──
    from alloy_torch.backend import (  # noqa: PLC0415
        InputSlot,
        IntermediateSlot,
        OutputSlot,
        OutputPassthrough,
    )
    # Collect per-plan kernel data (forward, backward, etc. are separate DAGs)
    all_plans = plans if plans is not None else _bmod._all_compiled_plans
    plan_profiles = []  # list of (plan, kernel_data_list)
    kernel_data = []  # flat list for total timing
    for pi, p in enumerate(all_plans):
        if p and p.plan_handle:
            iu = []
            for si, slot in enumerate(p.slots):
                if isinstance(slot, IntermediateSlot):
                    arr = p.phys_arrays[slot.physical_idx]
                    if isinstance(arr, AlloyBuffer):
                        iu.append((si, arr.base_ptr, arr.metal_nbytes))
            # Profile at the shrunk launch the run last dispatched (grid-shrink
            # prefill), so the per-kernel μs reflect the real prompt length, not
            # the max-length grid the plan was compiled at. Empty/None → full grid.
            grid_updates = p._last_grid_shrink_updates or []
            try:
                recs = _metal_ext.dispatch_plan_profiled(p.plan_handle, iu, grid_updates)
                plan_kd = []
                for r in sorted(recs, key=lambda x: x["idx"]):
                    entry = {
                        "idx": r["idx"],
                        "name": r["name"],
                        "gpu_us": round(r["gpu_us"], 1),
                        "grid": r["grid"],
                        "tg": r["tg"],
                    }
                    plan_kd.append(entry)
                    kernel_data.append(entry)
                plan_profiles.append((p, plan_kd))
            except Exception:
                pass

    total_gpu = sum(k["gpu_us"] for k in kernel_data) if kernel_data else 0

    # ── Dispatch DAGs (one per compiled plan) ──
    dispatch_dags = []  # list of {name, nodes, edges}
    _plan_labels = ["Forward", "Backward"] + [f"Plan {i}" for i in range(2, 20)]
    for pi, (p, pkd) in enumerate(plan_profiles):
        plan_gpu = sum(k["gpu_us"] for k in pkd) if pkd else 0
        nodes = []
        edges = []
        kd_by_idx = {k["idx"]: k for k in pkd}
        for di, d in enumerate(p.dispatches):
            kd = kd_by_idx.get(di, {})
            gpu = kd.get("gpu_us", 0)
            pct = gpu / plan_gpu * 100 if plan_gpu else 0
            dname = kd.get("name", f"dispatch_{di}")
            cat = (
                "gemm"
                if "dot" in dname
                else "attn"
                if "attention" in dname
                else "norm"
                if "layernorm" in dname or "rms" in dname
                else "activation"
                if "gelu" in dname or "relu" in dname or "sigmoid" in dname
                else "other"
            )
            nodes.append(
                {
                    "id": f"p{pi}d{di}",
                    "label": f"{dname} ({gpu:.0f}μs)",
                    "cat": cat,
                    "gpu_us": gpu,
                    "pct": round(pct, 1),
                }
            )

        slot_last_writer: dict[int, int] = {}
        edge_bytes: dict[tuple[int, int], int] = {}
        for di, d in enumerate(p.dispatches):
            read_slots = set(d.buf_slot_indices) - d.write_slot_indices
            for si in read_slots:
                writer = slot_last_writer.get(si)
                if writer is not None and writer != di:
                    key = (writer, di)
                    edge_bytes[key] = edge_bytes.get(key, 0) + p.slots[si].nbytes
            for si in d.write_slot_indices:
                slot_last_writer[si] = di
        for (src, dst), nbytes in edge_bytes.items():
            label = _fmt_bytes(nbytes)
            edges.append(
                {"from": f"p{pi}d{src}", "to": f"p{pi}d{dst}", "label": label, "bytes": nbytes}
            )

        plabel = _plan_labels[pi] if pi < len(_plan_labels) else f"Plan {pi}"
        dispatch_dags.append(
            {
                "name": f"{plabel} ({len(p.dispatches)} dispatches, {plan_gpu:.0f}μs)",
                "nodes": nodes,
                "edges": edges,
            }
        )

    arg_info: dict[int, dict] = {}
    if gm is not None:
        for ai, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder"):
            meta = node.meta.get("val")
            shape = tuple(meta.shape) if meta is not None and hasattr(meta, "shape") else ()
            arg_info[ai] = {"name": node.name, "shape": shape}

    ptr_to_param_name: dict[int, str] = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.nn.Module):
                continue
            for pname, param in obj.named_parameters():
                ptr_to_param_name[param.data_ptr()] = pname
            for bname, buf in obj.named_buffers():
                ptr_to_param_name[buf.data_ptr()] = bname
        except (ReferenceError, Exception):
            pass

    for dag_idx, dag in enumerate(dispatch_dags):
        if dag_idx >= len(plan_profiles):
            continue
        p = plan_profiles[dag_idx][0]
        written_slots = set()
        for d in p.dispatches:
            written_slots.update(d.write_slot_indices)

        # Group non-written slots by (is_input, root_ptr) so inputs and weights
        # are separate nodes even if they share a pointer.
        ext_by_key: dict[tuple[bool, int], list[int]] = {}
        for si, s in enumerate(p.slots):
            if si not in written_slots and not isinstance(s, IntermediateSlot):
                ext_by_key.setdefault((isinstance(s, InputSlot), s.root_ptr), []).append(si)

        prefix = f"p{dag_idx}"
        for (is_input, ptr), slots in ext_by_key.items():
            nbytes = max(p.slots[si].nbytes for si in slots)
            readers = []
            for di, d in enumerate(p.dispatches):
                read_slots = set(d.buf_slot_indices) - d.write_slot_indices
                if read_slots & set(slots):
                    readers.append(di)
            if not readers:
                continue
            cat = "input" if is_input else "weight"
            wid = f"{prefix}{'i' if is_input else 'w'}{ptr}"
            pname = ptr_to_param_name.get(ptr, "")
            pshape = ""
            for si in slots:
                s = p.slots[si]
                ai = s.arg_idx if isinstance(s, InputSlot) else -1
                if ai >= 0 and ai in arg_info:
                    info = arg_info[ai]
                    if not pname:
                        pname = info["name"]
                    pshape = "×".join(str(d) for d in info["shape"])
                    break
            if pname:
                parts = pname.split(".")
                short = ".".join(parts[-2:]) if len(parts) > 2 else pname
                label = f"{short} ({_fmt_bytes(nbytes)})"
            elif pshape:
                label = f"{pshape} ({_fmt_bytes(nbytes)})"
            else:
                label = _fmt_bytes(nbytes)
            detail = pname or "unknown"
            if pshape:
                detail += f" | {pshape}"
            detail += f" | {_fmt_bytes(nbytes)}"
            if is_input:
                detail += " | dynamic input"
            dag["nodes"].append(
                {
                    "id": wid,
                    "label": label,
                    "cat": cat,
                    "gpu_us": 0,
                    "pct": 0,
                    "detail": detail,
                }
            )
            for di in readers:
                total = sum(
                    p.slots[si].nbytes
                    for si in slots
                    if si
                    in set(p.dispatches[di].buf_slot_indices) - p.dispatches[di].write_slot_indices
                )
                dag["edges"].append(
                    {
                        "from": wid,
                        "to": f"{prefix}d{di}",
                        "label": _fmt_bytes(total),
                        "bytes": total,
                    }
                )

        # ── Output nodes: one per model output slot, edge from the
        # dispatch that produces it. Without these, sink dispatches
        # (logits, param grads, etc.) look orphaned in the DAG.
        for oi, entry in enumerate(p.output_mapping):
            if isinstance(entry, OutputSlot):
                slot_idx = entry.slot_idx
                writer_di: int | None = None
                for di, d in enumerate(p.dispatches):
                    if slot_idx in d.write_slot_indices:
                        writer_di = di
                if writer_di is None:
                    continue
                nbytes = p.slots[slot_idx].nbytes
                shape_str = "×".join(str(s) for s in entry.shape) if entry.shape else "scalar"
                wid = f"{prefix}o{oi}"
                dag["nodes"].append(
                    {
                        "id": wid,
                        "label": f"out[{oi}] {shape_str} ({_fmt_bytes(nbytes)})",
                        "cat": "output",
                        "gpu_us": 0,
                        "pct": 0,
                        "detail": f"output {oi} | {shape_str} | {_fmt_bytes(nbytes)}",
                    }
                )
                dag["edges"].append(
                    {
                        "from": f"{prefix}d{writer_di}",
                        "to": wid,
                        "label": _fmt_bytes(nbytes),
                        "bytes": nbytes,
                    }
                )
            elif isinstance(entry, OutputPassthrough):
                # Input passes directly to output (e.g. a weight returned
                # unchanged for backward). Skip — the producing "node" is
                # the input itself, and the input nodes are already drawn
                # with edges to their readers. Adding a terminal here just
                # clutters the DAG without surfacing a kernel insight.
                pass

    def _categorize(kd_list):
        cats: dict[str, list[float]] = {}
        for k in kd_list:
            n = k["name"]
            c = (
                "GEMM"
                if "dot" in n
                else "Attention"
                if "attention" in n
                else "Norm"
                if "layernorm" in n or "rms" in n
                else "RoPE"
                if "rope" in n
                else "Conv"
                if "im2col" in n
                else "Activation"
                if "gelu" in n or "relu" in n or "sigmoid" in n
                else n
            )
            cats.setdefault(c, []).append(k["gpu_us"])
        return cats

    cats = _categorize(kernel_data)

    # ── Per-plan info for sidebar ──
    all_plan_info = []
    for pi, (p, pkd) in enumerate(plan_profiles):
        plabel = _plan_labels[pi] if pi < len(_plan_labels) else f"Plan {pi}"
        info = {
            "name": plabel,
            "dispatches": len(p.dispatches),
            "slots": len(p.slots),
            "outputs": len(p.output_mapping),
            "physical_bufs": len(p.physical_bufs),
            "dep_groups": len(p.dep_groups),
            "gpu_us": sum(k["gpu_us"] for k in pkd),
            "cats": _categorize(pkd),
        }
        all_plan_info.append(info)
    # Backwards compat
    plan_info = (
        {
            "dispatches": sum(i["dispatches"] for i in all_plan_info),
            "slots": sum(i["slots"] for i in all_plan_info),
        }
        if all_plan_info
        else {}
    )

    _write_html(
        path,
        name,
        wall_ms,
        n_dispatches,
        total_gpu,
        fx_nodes,
        fx_edges,
        kernel_data,
        cats,
        plan_info,
        dispatch_dags,
        all_plan_info,
        wall_for_breakdown,
    )

    # Per-dispatch GPU timings (flat across plans): [{idx, name, gpu_us, grid, tg}].
    # Lets callers print a summary without re-parsing the HTML.
    return kernel_data


def _write_html(
    path,
    name,
    wall_ms,
    n_dispatches,
    total_gpu,
    fx_nodes,
    fx_edges,
    kernel_data,
    cats,
    plan_info,
    dispatch_dags=None,
    all_plan_info=None,
    wall_for_breakdown=0.0,
):
    h = html.escape

    # Per-plan GPU breakdown
    plan_infos = all_plan_info or []
    if plan_infos:
        cat_rows = ""
        for pi_info in plan_infos:
            cat_rows += f"<tr><td colspan='5' style='background:#1a1a2e;font-weight:bold;padding:6px'>{h(pi_info['name'])} — {pi_info['gpu_us']:.0f}μs</td></tr>\n"
            plan_total = pi_info["gpu_us"]
            for c, times in sorted(pi_info["cats"].items(), key=lambda x: -sum(x[1])):
                t = sum(times)
                pct = t / plan_total * 100 if plan_total else 0
                cat_rows += f"<tr><td>{h(c)}</td><td>{len(times)}</td><td>{t:.0f}</td><td>{pct:.1f}%</td><td><div class='bar' style='width:{pct}%'></div></td></tr>\n"
    else:
        cat_rows = ""
        for c, times in sorted(cats.items(), key=lambda x: -sum(x[1])):
            t = sum(times)
            pct = t / total_gpu * 100 if total_gpu else 0
            cat_rows += f"<tr><td>{h(c)}</td><td>{len(times)}</td><td>{t:.0f}</td><td>{pct:.1f}%</td><td><div class='bar' style='width:{pct}%'></div></td></tr>\n"

    kernel_rows = ""
    for k in sorted(kernel_data, key=lambda k: -k["gpu_us"]):
        pct = k["gpu_us"] / total_gpu * 100 if total_gpu else 0
        g = f"{k['grid'][0]}×{k['grid'][1]}×{k['grid'][2]}"
        tg = f"{k['tg'][0]}×{k['tg'][1]}×{k['tg'][2]}"
        cls = "hot" if pct > 5 else ""
        kernel_rows += f'<tr class="{cls}"><td>{k["idx"]}</td><td>{k["gpu_us"]:.0f}</td><td>{pct:.1f}%</td><td>{g}</td><td>{tg}</td><td>{h(k["name"])}</td></tr>\n'

    # Per-plan compiled plan info
    if plan_infos:
        plan_rows = ""
        for pi_info in plan_infos:
            plan_rows += f"<tr><td colspan='2' style='background:#1a1a2e;font-weight:bold;padding:6px'>{h(pi_info['name'])}</td></tr>\n"
            for k in ("dispatches", "slots", "outputs", "physical_bufs", "dep_groups"):
                plan_rows += f"<tr><td>{h(k)}</td><td>{pi_info[k]}</td></tr>\n"
    else:
        plan_rows = "".join(f"<tr><td>{h(k)}</td><td>{v}</td></tr>" for k, v in plan_info.items())

    # Profiler breakdown
    gpu_ms = total_gpu / 1000
    wb = wall_for_breakdown if wall_for_breakdown > 0 else wall_ms
    overhead_ms = max(0, wb - gpu_ms)
    gpu_pct = gpu_ms / wb * 100 if wb > 0 else 0
    overhead_pct = 100 - gpu_pct
    profiler_rows = (
        f"<tr><td>Wall clock</td><td>{wb:.2f} ms</td><td>100%</td>"
        f"<td><div class='bar' style='width:100%;background:#30363d'></div></td></tr>\n"
        f"<tr><td>GPU compute</td><td>{gpu_ms:.2f} ms</td><td>{gpu_pct:.1f}%</td>"
        f"<td><div class='bar' style='width:{gpu_pct}%'></div></td></tr>\n"
        f"<tr><td>Overhead</td><td>{overhead_ms:.2f} ms</td><td>{overhead_pct:.1f}%</td>"
        f"<td><div class='bar' style='width:{overhead_pct}%;background:#f0883e'></div></td></tr>\n"
        f"<tr><td colspan='4' style='color:#8b949e;font-size:11px;padding-top:8px'>"
        f"Overhead = encode + commit + GPU scheduling + waitUntilCompleted wake-up. "
        f"Irreducible Metal driver cost (~0.1-0.3ms).</td></tr>\n"
    )
    # Per-plan breakdown
    for pi_info in all_plan_info or []:
        pgpu = pi_info["gpu_us"] / 1000
        profiler_rows += (
            f"<tr><td colspan='4' style='background:#1a1a2e;font-weight:bold;padding:6px'>"
            f"{h(pi_info['name'])} — {pi_info['dispatches']} dispatches</td></tr>\n"
            f"<tr><td>GPU time</td><td>{pgpu:.2f} ms</td><td></td><td></td></tr>\n"
        )
        for c, times in sorted(pi_info["cats"].items(), key=lambda x: -sum(x[1])):
            t_ms = sum(times) / 1000
            cpct = sum(times) / pi_info["gpu_us"] * 100 if pi_info["gpu_us"] else 0
            profiler_rows += (
                f"<tr><td style='padding-left:20px'>{h(c)}</td><td>{t_ms:.2f} ms</td>"
                f"<td>{cpct:.1f}%</td><td><div class='bar' style='width:{cpct}%'></div></td></tr>\n"
            )

    # JSON data for the graph renderers
    graph_json = json.dumps({"nodes": fx_nodes, "edges": fx_edges})
    dispatch_dags_json = json.dumps(dispatch_dags or [])

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Alloy: {h(name)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background:#0d1117; color:#c9d1d9; }}
  .header {{ padding:20px 24px; border-bottom:1px solid #30363d; }}
  h1 {{ color:#58a6ff; font-size:20px; }}
  .stats {{ display:flex; gap:16px; margin:12px 0 0; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 18px; }}
  .stat .val {{ font-size:22px; font-weight:700; color:#58a6ff; }}
  .stat .lbl {{ font-size:11px; color:#8b949e; }}
  .content {{ display:flex; height:calc(100vh - 120px); }}
  .sidebar {{ width:380px; border-right:1px solid #30363d; overflow-y:auto; flex-shrink:0; }}
  .main {{ flex:1; overflow:hidden; position:relative; }}
  .section {{ padding:12px 16px; border-bottom:1px solid #21262d; }}
  .section h2 {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12px; font-family:monospace; }}
  th {{ text-align:left; padding:4px 8px; color:#8b949e; font-weight:normal; position:sticky; top:0; background:#0d1117; }}
  td {{ padding:3px 8px; border-bottom:1px solid #161b22; }}
  tr:hover {{ background:#161b22; }}
  tr.hot td {{ color:#f85149; font-weight:600; }}
  .bar {{ height:12px; background:#58a6ff; border-radius:2px; min-width:2px; }}
  /* Graph */
  .graph-container {{ width:100%; height:100%; overflow:auto; cursor:grab; }}
  .graph-container:active {{ cursor:grabbing; }}
  svg {{ min-width:100%; min-height:100%; }}
  .node rect {{ rx:4; ry:4; stroke-width:1.5; cursor:pointer; }}
  .node text {{ font-size:11px; font-family:monospace; fill:#c9d1d9; pointer-events:none; }}
  .node.gemm rect {{ fill:#1f1033; stroke:#d2a8ff; }}
  .node.attn rect {{ fill:#2a1800; stroke:#f0883e; }}
  .node.norm rect {{ fill:#0a2612; stroke:#3fb950; }}
  .node.activation rect {{ fill:#0a2612; stroke:#3fb950; }}
  .node.residual rect {{ fill:#1a1a00; stroke:#d29922; }}
  .node.input rect {{ fill:#0c2d48; stroke:#58a6ff; }}
  .node.weight rect {{ fill:#161b22; stroke:#484f58; stroke-dasharray:4,2; }}
  .node.output rect {{ fill:#2a1a3a; stroke:#d2a8ff; stroke-width:2; }}
  .node.output text {{ fill:#f0c5ff; font-weight:600; }}
  .node.other rect {{ fill:#161b22; stroke:#30363d; }}
  .node:hover rect {{ stroke-width:2.5; filter:brightness(1.3); }}
  .edge {{ stroke:#30363d; stroke-width:1; fill:none; marker-end:url(#arrow); }}
  .node-info {{ position:absolute; bottom:12px; left:12px; right:12px; background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; font-size:12px; font-family:monospace; display:none; z-index:10; }}
  .tab-bar {{ display:flex; border-bottom:1px solid #30363d; }}
  .tab {{ padding:8px 14px; font-size:12px; color:#8b949e; cursor:pointer; border-bottom:2px solid transparent; }}
  .tab.active {{ color:#c9d1d9; border-bottom-color:#58a6ff; }}
  .filter {{ width:100%; padding:6px 10px; background:#0d1117; border:1px solid #30363d; color:#c9d1d9; border-radius:4px; font-size:12px; margin-bottom:8px; }}
</style>
</head><body>
<div class="header">
  <h1>⚡ {h(name)}</h1>
  <div class="stats">
    <div class="stat"><div class="val">{wall_ms:.1f}<span style="font-size:12px">ms</span></div><div class="lbl">Wall clock</div></div>
    <div class="stat"><div class="val">{total_gpu / 1000:.1f}<span style="font-size:12px">ms</span></div><div class="lbl">GPU time</div></div>
    <div class="stat"><div class="val">{n_dispatches}</div><div class="lbl">Dispatches</div></div>
    <div class="stat"><div class="val">{len(fx_nodes)}</div><div class="lbl">FX nodes</div></div>
  </div>
</div>
<div class="content">
  <div class="sidebar">
    <div class="tab-bar">
      <div class="tab active" onclick="showSidebar('gpu')">GPU</div>
      <div class="tab" onclick="showSidebar('profiler')">Profiler</div>
      <div class="tab" onclick="showSidebar('dispatches')">Dispatches</div>
      <div class="tab" onclick="showSidebar('plan')">Plan</div>
    </div>
    <div id="gpu" class="section" style="display:block">
      <h2>GPU Breakdown</h2>
      <table><tr><th>Category</th><th>#</th><th>μs</th><th>%</th><th></th></tr>{cat_rows}</table>
    </div>
    <div id="profiler" class="section" style="display:none">
      <h2>Profiler</h2>
      <table><tr><th>Phase</th><th>Time</th><th>%</th><th></th></tr>{profiler_rows}</table>
    </div>
    <div id="dispatches" class="section" style="display:none">
      <h2>Kernel Dispatches</h2>
      <input class="filter" placeholder="Filter..." oninput="filterTable(this,'ktable')">
      <table id="ktable"><tr><th>#</th><th>μs</th><th>%</th><th>Grid</th><th>TG</th><th>Kernel</th></tr>{kernel_rows}</table>
    </div>
    <div id="plan" class="section" style="display:none">
      <h2>Compiled Plan</h2>
      <table>{plan_rows}</table>
    </div>
  </div>
  <div class="main">
    <div class="tab-bar" style="padding:0 12px;border-bottom:1px solid #30363d;display:flex;align-items:center">
      <div class="tab" onclick="showGraph('fx')">FX Graph</div>
      <div class="tab active" onclick="showGraph('dispatch')">Dispatch DAG</div>
      <select id="dag-selector" style="display:none;margin-left:8px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:2px 6px;font-size:12px" onchange="switchDag(this.selectedIndex)"></select>
      <span id="dag-toggles" style="margin-left:auto;font-size:12px;color:#8b949e">
        <label style="padding:4px 8px;cursor:pointer;user-select:none">
          <input type="checkbox" id="show-weights" checked onchange="renderDispatchDAG()"> Weights
        </label>
        <label style="padding:4px 8px;cursor:pointer;user-select:none">
          <input type="checkbox" id="show-inputs" checked onchange="renderDispatchDAG()"> Inputs
        </label>
        <label style="padding:4px 8px;cursor:pointer;user-select:none">
          <input type="checkbox" id="show-outputs" checked onchange="renderDispatchDAG()"> Outputs
        </label>
      </span>
    </div>
    <div class="graph-container" id="graph-container-fx" style="display:none"></div>
    <div class="graph-container" id="graph-container-dispatch"></div>
    <div class="node-info" id="node-info"></div>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/@dagrejs/dagre@3.0.0/dist/dagre.min.js"></script>
<script>
const data = {graph_json};
const dispatchDags = {dispatch_dags_json};
let currentDagIdx = 0;

function showSidebar(id) {{
  document.querySelectorAll('.sidebar .section').forEach(e => e.style.display = 'none');
  document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
  document.getElementById(id).style.display = 'block';
  event.target.classList.add('active');
}}
function filterTable(input, id) {{
  const q = input.value.toLowerCase();
  document.querySelectorAll('#'+id+' tr').forEach((r,i) => {{
    if(i===0) return;
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

let fxRendered = false;
function showGraph(which) {{
  document.getElementById('graph-container-fx').style.display = which === 'fx' ? '' : 'none';
  document.getElementById('graph-container-dispatch').style.display = which === 'dispatch' ? '' : 'none';
  document.getElementById('dag-selector').style.display = which === 'dispatch' && dispatchDags.length > 1 ? '' : 'none';
  document.getElementById('dag-toggles').style.display = which === 'dispatch' ? 'flex' : 'none';
  const tabs = event.target.parentElement.querySelectorAll('.tab');
  tabs.forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  if(which === 'dispatch') {{
    renderDispatchDAG();
  }} else if(which === 'fx' && !fxRendered) {{
    fxRendered = true;
    try {{ renderGraph(); }} catch(e) {{ console.error('FX render error:', e); }}
  }}
}}

function switchDag(idx) {{
  currentDagIdx = idx;
  renderDispatchDAG();
}}

function renderDispatchDAG() {{
  if(!dispatchDags.length) return;
  const src = dispatchDags[currentDagIdx];
  const showWeights = document.getElementById('show-weights').checked;
  const showInputs = document.getElementById('show-inputs').checked;
  const hiddenCats = new Set();
  if (!showWeights) hiddenCats.add('weight');
  if (!showInputs) hiddenCats.add('input');
  const showOutputsEl = document.getElementById('show-outputs');
  const showOutputs = showOutputsEl ? showOutputsEl.checked : true;
  if (!showOutputs) hiddenCats.add('output');
  const hiddenNodes = new Set(src.nodes.filter(n => hiddenCats.has(n.cat)).map(n => n.id));
  const nodes = src.nodes.filter(n => !hiddenCats.has(n.cat));
  const edges = src.edges.filter(e => !hiddenNodes.has(e.from) && !hiddenNodes.has(e.to));
  const container = document.getElementById('graph-container-dispatch');
  container.innerHTML = '';
  if(!nodes.length) return;

  const g = new dagre.graphlib.Graph();
  g.setGraph({{ rankdir: 'TB', nodesep: 12, ranksep: 24, edgesep: 6, marginx: 20, marginy: 20 }});
  g.setDefaultEdgeLabel(() => ({{}}));

  const nodeW = 220, nodeH = 28;
  nodes.forEach(n => {{
    g.setNode(n.id, {{ width: nodeW, height: nodeH, label: n.label, cat: n.cat, node: n }});
  }});
  edges.forEach(e => g.setEdge(e.from, e.to, {{ label: e.label || '', bytes: e.bytes || 0 }}));
  dagre.layout(g);

  const ns = 'http://www.w3.org/2000/svg';
  const graph = g.graph();
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', graph.width + 40);
  svg.setAttribute('height', graph.height + 40);

  // Reuse arrow marker
  const defs = document.createElementNS(ns, 'defs');
  const marker = document.createElementNS(ns, 'marker');
  marker.setAttribute('id', 'arrow2');
  marker.setAttribute('viewBox', '0 0 10 10');
  marker.setAttribute('refX', '10'); marker.setAttribute('refY', '5');
  marker.setAttribute('markerWidth', '5'); marker.setAttribute('markerHeight', '5');
  marker.setAttribute('orient', 'auto-start-reverse');
  const mp = document.createElementNS(ns, 'path');
  mp.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  mp.setAttribute('fill', '#484f58');
  marker.appendChild(mp); defs.appendChild(marker); svg.appendChild(defs);

  // Edges with memory traffic labels
  g.edges().forEach(e => {{
    const edge = g.edge(e);
    if(!edge.points || edge.points.length < 2) return;
    const pts = edge.points;
    let d = `M${{pts[0].x}},${{pts[0].y}}`;
    for(let i = 1; i < pts.length; i++) {{
      d += ` L${{pts[i].x}},${{pts[i].y}}`;
    }}
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', d);
    path.setAttribute('stroke', '#30363d');
    path.setAttribute('stroke-width', '1');
    path.setAttribute('fill', 'none');
    path.setAttribute('marker-end', 'url(#arrow2)');
    svg.appendChild(path);
    // Label at midpoint
    if(edge.label) {{
      const mid = pts[Math.floor(pts.length/2)];
      const txt = document.createElementNS(ns, 'text');
      txt.setAttribute('x', mid.x + 4);
      txt.setAttribute('y', mid.y - 4);
      txt.setAttribute('fill', '#8b949e');
      txt.setAttribute('font-size', '9');
      txt.setAttribute('font-family', 'monospace');
      txt.textContent = edge.label;
      svg.appendChild(txt);
    }}
  }});

  // Nodes — scale brightness by GPU time percentage
  g.nodes().forEach(id => {{
    const nd = g.node(id);
    const n = nd.node;
    const gEl = document.createElementNS(ns, 'g');
    gEl.setAttribute('class', `node ${{nd.cat}}`);
    gEl.setAttribute('transform', `translate(${{nd.x - nodeW/2}},${{nd.y - nodeH/2}})`);

    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', nodeW);
    rect.setAttribute('height', nodeH);
    // Hot dispatches get brighter border
    if(n.pct > 5) rect.setAttribute('stroke-width', '2.5');
    gEl.appendChild(rect);

    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', '6'); text.setAttribute('y', '17');
    text.setAttribute('font-size', '11'); text.setAttribute('font-family', 'monospace');
    text.setAttribute('fill', n.pct > 5 ? '#f85149' : '#c9d1d9');
    let label = nd.label;
    if(label.length > 32) label = label.slice(0,30) + '…';
    text.textContent = label;
    gEl.appendChild(text);

    gEl.addEventListener('click', () => {{
      const info = document.getElementById('node-info');
      info.style.display = 'block';
      if(n.detail) {{
        info.innerHTML = `<b>${{n.detail}}</b>`;
      }} else {{
        info.innerHTML = `<b>#${{n.id.slice(1)}}</b> ${{n.label}}<br>${{n.pct}}% of GPU time`;
      }}
    }});

    svg.appendChild(gEl);
  }});

  container.appendChild(svg);
}}

function renderGraph() {{
  const visible = data.nodes.filter(n => n.op !== 'output');
  const visIds = new Set(visible.map(n => n.id));
  const visEdges = data.edges.filter(e => visIds.has(e.from) && visIds.has(e.to));

  const g = new dagre.graphlib.Graph();
  g.setGraph({{ rankdir: 'TB', nodesep: 16, ranksep: 30, edgesep: 8, marginx: 20, marginy: 20 }});
  g.setDefaultEdgeLabel(() => ({{}}));

  const nodeW = 160, nodeH = 26;
  visible.forEach(n => {{
    let label = n.target;
    if(n.shape) label += ' ' + n.shape;
    if(label.length > 24) label = label.slice(0,22) + '…';
    g.setNode(n.id, {{ width: nodeW, height: nodeH, label, cat: n.cat, node: n }});
  }});
  visEdges.forEach(e => g.setEdge(e.from, e.to));

  dagre.layout(g);

  // Render SVG
  const container = document.getElementById('graph-container-fx');
  const ns = 'http://www.w3.org/2000/svg';
  const graph = g.graph();

  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', graph.width + 40);
  svg.setAttribute('height', graph.height + 40);

  // Arrow marker
  const defs = document.createElementNS(ns, 'defs');
  const marker = document.createElementNS(ns, 'marker');
  marker.setAttribute('id', 'arrow');
  marker.setAttribute('viewBox', '0 0 10 10');
  marker.setAttribute('refX', '10');
  marker.setAttribute('refY', '5');
  marker.setAttribute('markerWidth', '5');
  marker.setAttribute('markerHeight', '5');
  marker.setAttribute('orient', 'auto-start-reverse');
  const mp = document.createElementNS(ns, 'path');
  mp.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  mp.setAttribute('fill', '#484f58');
  marker.appendChild(mp);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // Edges with dagre's routed points
  g.edges().forEach(e => {{
    const edge = g.edge(e);
    if(!edge.points || edge.points.length < 2) return;
    const pts = edge.points;
    let d = `M${{pts[0].x}},${{pts[0].y}}`;
    for(let i = 1; i < pts.length; i++) {{
      const p = pts[i];
      const prev = pts[i-1];
      const mx = (prev.x + p.x) / 2;
      const my = (prev.y + p.y) / 2;
      d += ` Q${{prev.x}},${{my}} ${{p.x}},${{p.y}}`;
    }}
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', d);
    path.setAttribute('class', 'edge');
    svg.appendChild(path);
  }});

  // Nodes
  g.nodes().forEach(id => {{
    const nd = g.node(id);
    const n = nd.node;
    const gEl = document.createElementNS(ns, 'g');
    gEl.setAttribute('class', `node ${{nd.cat}}`);
    gEl.setAttribute('transform', `translate(${{nd.x - nodeW/2}},${{nd.y - nodeH/2}})`);

    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('width', nodeW);
    rect.setAttribute('height', nodeH);
    gEl.appendChild(rect);

    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', '6');
    text.setAttribute('y', '17');
    text.textContent = nd.label;
    gEl.appendChild(text);

    gEl.addEventListener('click', () => {{
      const info = document.getElementById('node-info');
      info.style.display = 'block';
      info.innerHTML = `<b>${{n.id}}</b> (${{n.op}})<br>target: ${{n.target}}<br>shape: ${{n.shape || '—'}}<br>cat: ${{n.cat}}`;
    }});

    svg.appendChild(gEl);
  }});

  container.appendChild(svg);
}}

try {{
  if(dispatchDags.length > 1) {{
    const sel = document.getElementById('dag-selector');
    sel.style.display = '';
    sel.innerHTML = dispatchDags.map((d,i) => `<option ${{i===0?'selected':''}}>${{d.name}}</option>`).join('');
  }}
  renderDispatchDAG();
  // FX graph renders lazily on first tab click
}} catch(e) {{
  console.error('Render error:', e);
}}
</script>
</body></html>"""

    with open(path, "w") as f:
        f.write(doc)

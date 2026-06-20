"""Unwrap torch's auto_functionalized_v2 around alloy custom ops.

AOT autograd wraps any custom op with mutable input annotations
(`Tensor(c!) conv_state` etc.) in `torch.ops.higher_order.auto_functionalized_v2`
so the FX graph stays functional. The HOP returns a tuple
`(orig_output, *cloned_mutated_inputs)` and downstream code reads the
clones instead of the originals.

The alloy backend's handlers do the mutations in-place directly (the
mutable annotations are there to tell AOT to keep the call in-graph,
not to functionalize). Unwrap the HOP back into a direct call:

  before: t = auto_functionalized_v2(op, *args, **mut_kwargs)
          out = t[0]; new_conv = t[1]; new_rec = t[2]; ...
   after: out = op(*args, mut_kwargs unpacked positionally)
          new_conv replaced with the original conv_state arg;
          new_rec replaced with the original recurrent_state arg.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.fx


_HOP = torch.ops.higher_order.auto_functionalized_v2


def _arg_specs_for(op_overload: object) -> list[str]:
    schema = cast(torch._C.FunctionSchema, op_overload._schema)  # type: ignore[attr-defined]
    return [arg.name for arg in schema.arguments]


def rewrite_unwrap_auto_functionalized(graph: torch.fx.Graph) -> int:
    """Replace auto_functionalized_v2 wrappers with direct op calls.

    Also records (mutated_output_node → original_input_placeholder) info on
    the graph's output node so the alloy backend's mutation_map extractor
    can route the alloy op's in-place writes straight to the input's
    storage. Without this, the HOP makes the graph look "functional" to
    AOT and the standard `entry.mutated_input` annotations are empty,
    leaving the alloy backend to allocate clones whose mutations never
    reach the caller's tensors.
    """
    # Collected mutations: the output node's meta gets a sidecar dict mapping
    # output-tuple-index → placeholder_node so the extractor can produce the
    # AOT-shaped mutation map.
    placeholder_for_output_idx: dict[int, torch.fx.Node] = {}
    changed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not _HOP:
            continue
        # args[0] is the wrapped op overload; remaining args/kwargs are its
        # positional args and mutated-input kwargs (named `_<name>_base_index`
        # and a flat `_all_bases` list per torch internals). Pull every kwarg
        # named after a schema arg and pass it positionally to the op.
        op_overload = node.args[0]
        if not isinstance(op_overload, torch._ops.OpOverload):
            continue
        if op_overload.namespace != "alloy":
            continue
        arg_names = _arg_specs_for(op_overload)
        kw = dict(node.kwargs)
        # `_all_bases` is the flat list of mutated input clones. Look up
        # each mutated input by name via the index in `_<name>_base_index`.
        all_bases = kw.pop("_all_bases", None)
        if not isinstance(all_bases, (list, tuple)):
            all_bases = ()
        positional = []
        for name in arg_names:
            base_idx_key = f"_{name}_base_index"
            if base_idx_key in kw:
                idx = kw.pop(base_idx_key)
                if isinstance(idx, int) and 0 <= idx < len(all_bases):
                    positional.append(all_bases[idx])
                    continue
                positional.append(None)
                continue
            if name in kw:
                positional.append(kw.pop(name))
                continue
            positional.append(None)

        with graph.inserting_after(node):
            direct = graph.call_function(op_overload, tuple(positional), {})

        # The HOP returns (output, *mutated_clones). Replace getitem(0) with
        # the direct call; for getitem(i>0), point to the corresponding base
        # (the original mutated input passed in). Track the base for each
        # mutated index so we can wire mutation_map below.
        idx_to_base: dict[int, torch.fx.Node] = {}
        for i_base, base_node in enumerate(all_bases):
            if isinstance(base_node, torch.fx.Node):
                idx_to_base[i_base + 1] = base_node  # HOP getitem 1..N are mutated clones
        for user in list(node.users):
            if user.op != "call_function" or user.target is not __import__(
                "operator"
            ).getitem:
                continue
            idx = user.args[1]
            if idx == 0:
                user.replace_all_uses_with(direct)
            elif isinstance(idx, int) and 1 <= idx <= len(all_bases):
                user.replace_all_uses_with(all_bases[idx - 1])
            else:
                continue
            graph.erase_node(user)

        graph.erase_node(node)
        changed += 1

        # Stash per-base mutation info: the FX output node's tuple may
        # include base placeholders as passthroughs (in lieu of the
        # original HOP-cloned outputs we just spliced away). Record them
        # so the backend can read the kernel-mutated value out of the
        # base's storage instead of returning the unmutated input clone.
        for i_base, base_node in idx_to_base.items():
            if isinstance(base_node, torch.fx.Node):
                placeholder_for_output_idx[id(base_node)] = base_node

    if changed:
        # Walk the graph's output node tuple and mark any output entry that
        # IS a base placeholder. Stamp graph.meta with a {output_idx → arg_idx}
        # dict — `_extract_mutation_map` (in backend.py) reads this dict via
        # the new "alloy_auto_functionalized_mutations" key.
        out_node = next(n for n in graph.nodes if n.op == "output")
        out_tuple = out_node.args[0]
        if isinstance(out_tuple, (list, tuple)):
            placeholder_to_arg_idx: dict[int, int] = {}
            arg_i = 0
            for n in graph.nodes:
                if n.op == "placeholder":
                    placeholder_to_arg_idx[id(n)] = arg_i
                    arg_i += 1
            mut_map: dict[int, int] = {}
            for out_idx, entry in enumerate(out_tuple):
                if isinstance(entry, torch.fx.Node) and id(entry) in placeholder_for_output_idx:
                    arg_idx = placeholder_to_arg_idx.get(id(entry))
                    if arg_idx is not None:
                        mut_map[out_idx] = arg_idx
            if mut_map:
                out_node.meta["alloy_auto_functionalized_mutations"] = mut_map

    return changed

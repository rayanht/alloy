"""Focused in-process benchmark for Alloy local LLM inference.

Skips the server + HTTP lifecycle entirely. Loads the model directly via
`load_native_causal_lm`, builds an `AlloyGenerator`, runs
`eager_compile_all` so no measurement eats compile cost, then drives the
generator from Python.

Three workloads, selected via `--dataset`:

  synthetic (default) — pp/tg throughput on the native cache, median over
    `--reps`. One depth (default 512) reports pp512 / tg128 as two independent
    empty-cache tests (pp prefills from empty; tg decodes from a 1-token seed,
    matching llama-bench). Several `--depths` run a sweep: per depth, prefill
    `depth` then decode AT that occupancy (tg degrades as the cache fills).

  multimodal — an image + prompt through the vision tower (requires --image
    and a vision-capable model, e.g. gemma4:e2b). Reports vision-encode time,
    TTFT (incl. the encode), steady-state decode tok/s, and total wall.

  embeddings — an encoder model (e.g. nomic-embed-text) through the in-process
    alloy `EmbeddingModel.embed`: tok/s per batch/seq regime, median over
    `--reps`.

Driven by `alloy bench` (see `alloy bench --help`); `main(argv)` is the
argparse entrypoint that the typer command delegates to.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from alloy import get_logger
from alloy_server.generation.generator import AlloyGenerator
from alloy_server.generation.sequence import MultimodalInputs, Sequence
from alloy_server.models import load_native_causal_lm
from alloy_server.models.nomic_bert import load_ollama_gguf_embedder
from alloy_server import resolve_prefill_policy

logger = get_logger("alloy_cli.bench")


@dataclass(slots=True)
class DepthPoint:
    """One pp/tg measurement: prefill `depth` tokens (pp), decode `gen_tokens`
    (tg). `run_llama_bench_alloy` decodes tg from empty; `run_depths_alloy`
    decodes at `depth`'s occupancy."""

    depth: int
    gen_tokens: int
    pp_tok_per_s: float
    tg_tok_per_s: float


def _synthetic_token_ids(vocab_size: int, n_tokens: int, *, seed: int) -> torch.Tensor:
    """Reproducible token ids drawn from a safe mid-range slice of the
    vocab so we steer clear of special tokens (<256) and reserved upper
    ids."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1024, max(1025, vocab_size - 256), (1, n_tokens), generator=gen, dtype=torch.long)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# Depth sweep (llama-bench methodology): pp / tg throughput vs cache depth
# ---------------------------------------------------------------------------


def run_depths_alloy(
    alloy_gen: AlloyGenerator, depths: list[int], gen_tokens: int,
    vocab_size: int, *, reps: int,
) -> list[DepthPoint]:
    """Alloy depth sweep on the native cache. Per depth: prefill `depth`
    synthetic tokens (pp) then decode `gen_tokens` (tg), median throughput over
    `reps` (one warmup rep discarded). Synthetic tokens are content-independent
    for throughput at temperature 0 (verified: prose vs random within ~1%).
    `ignore_eos` forces the full `gen_tokens` decode so a random prompt that
    trips an early EOS can't shorten — and corrupt — the tg measurement."""
    out: list[DepthPoint] = []
    for d in depths:
        pp: list[float] = []
        tg: list[float] = []
        for rep in range(reps + 1):
            # Fresh tokens per rep so the prefix cache can't hit across reps
            # (a hit would zero out the prefill and explode pp).
            ids = _synthetic_token_ids(vocab_size, d, seed=(0xD000 ^ d) + rep)
            alloy_gen.reset_prefix_state()
            for _ in alloy_gen.run(Sequence(
                input_ids=ids, max_new_tokens=gen_tokens, ignore_eos=True,
            )):
                pass
            t = alloy_gen.last_gen_timings
            if rep == 0:
                continue  # discard warmup (first-touch page faults, clock ramp)
            pp.append(int(t["prompt_tokens"]) / max(float(t["prefill_ms"]), 1e-9) * 1000.0)
            tg.append(int(t["decode_tokens"]) / max(float(t["decode_ms"]), 1e-9) * 1000.0)
        out.append(DepthPoint(d, gen_tokens, _pct(pp, 0.5), _pct(tg, 0.5)))
    return out


def run_llama_bench_alloy(
    alloy_gen: AlloyGenerator, pp_tokens: int, gen_tokens: int,
    vocab_size: int, *, reps: int,
) -> DepthPoint:
    """pp{pp_tokens} / tg{gen_tokens} as two independent empty-cache tests
    (llama-bench clears the cache per test): pp prefills `pp_tokens` from empty;
    tg decodes `gen_tokens` from a 1-token seed (depth ~0, not after a prefill).
    Median over `reps`, one warmup discarded."""
    pp: list[float] = []
    tg: list[float] = []
    for rep in range(reps + 1):
        ids = _synthetic_token_ids(vocab_size, pp_tokens, seed=(0x9900 ^ pp_tokens) + rep)
        alloy_gen.reset_prefix_state()
        for _ in alloy_gen.run(Sequence(input_ids=ids, max_new_tokens=1, ignore_eos=True)):
            pass
        t_pp = alloy_gen.last_gen_timings
        seed_ids = _synthetic_token_ids(vocab_size, 1, seed=(0x7700 ^ gen_tokens) + rep)
        alloy_gen.reset_prefix_state()
        for _ in alloy_gen.run(Sequence(input_ids=seed_ids, max_new_tokens=gen_tokens, ignore_eos=True)):
            pass
        t_tg = alloy_gen.last_gen_timings
        if rep == 0:
            continue  # discard warmup
        pp.append(int(t_pp["prompt_tokens"]) / max(float(t_pp["prefill_ms"]), 1e-9) * 1000.0)
        tg.append(int(t_tg["decode_tokens"]) / max(float(t_tg["decode_ms"]), 1e-9) * 1000.0)
    return DepthPoint(pp_tokens, gen_tokens, _pct(pp, 0.5), _pct(tg, 0.5))


# ---------------------------------------------------------------------------
# Multimodal (vision) workload — image + prompt
# ---------------------------------------------------------------------------
#
# The depth sweep above measures token-only generation. A vision request has
# one extra phase the user waits on: the image is run through the vision tower
# (alloy's ViT → pooler → projector) BEFORE any text token is prefilled. So the
# primary metric here is TTFT measured as wall-clock from the request to the first
# decoded token — which INCLUDES the vision encode. Every rep runs cold (no warm
# prefix) so it pays the full vision + prefill cost. Alloy's vision-encode time
# is also reported on its own as a diagnostic (the slice the offline tuner moves).

_DEFAULT_MM_PROMPT = "Describe this image in detail."


@dataclass(slots=True)
class MMRep:
    """One vision request, timed end-to-end (wall clock)."""
    vision_ms: float    # encode() — the ViT/pooler/projector
    ttft_ms: float      # request start -> first decoded token (INCLUDES vision encode + prefill)
    decode_ms: float    # first token -> last token (steady-state decode)
    wall_ms: float      # request start -> last token
    prompt_tokens: int  # image soft-tokens + text (input_ids)
    decode_tokens: int  # tokens generated


@dataclass(slots=True)
class MultimodalStats:
    image: str
    prompt: str
    reps: list[MMRep]

    def _derived(self) -> dict[str, list[float]]:
        ttft = [r.ttft_ms for r in self.reps]
        vis = [r.vision_ms for r in self.reps]
        wall = [r.wall_ms for r in self.reps]
        # Steady-state decode rate excludes the first token (already in TTFT).
        dec = [
            (r.decode_tokens - 1) / max(r.decode_ms, 1e-9) * 1000
            for r in self.reps if r.decode_tokens > 1
        ]
        # End-to-end: all generated tokens over the whole request wall (vision +
        # prefill + decode).
        e2e = [
            r.decode_tokens / max(r.wall_ms, 1e-9) * 1000
            for r in self.reps if r.decode_tokens > 0
        ]
        return {"ttft": ttft, "vision": vis, "wall": wall, "decode_tps": dec, "e2e_tps": e2e}


def resolve_bench_image(path: Path) -> bytes:
    """Read an image file for the multimodal workload. We deliberately do not
    auto-download — the caller passes a real file (the release scorecard pins a
    fixed asset so runs are comparable across machines)."""
    if not path.is_file():
        raise FileNotFoundError(f"--image is not a file: {path}")
    return path.read_bytes()


def _mm_encode_alloy(
    tokenizer, vision, image_bytes: bytes, prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mirror the server's multimodal encode (`build_multimodal_hooks`): run the
    vision tower, render the chat with one image placeholder, expand it to the
    soft-token run, tokenize, and locate the placeholder positions. Returns
    (input_ids, features, positions, vision_ms) — vision_ms timing the ViT only."""
    t0 = time.perf_counter()
    feats = vision.encode(image_bytes)  # ViT → pooler → projector (alloy dispatch)
    vision_ms = (time.perf_counter() - t0) * 1000.0
    content = [{"type": "image"}, {"type": "text", "text": prompt}]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    full_text = vision.expand_text(text, [feats])
    enc = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    row = enc[0] if enc and isinstance(enc[0], list) else enc
    input_ids = torch.tensor([row], dtype=torch.long)
    positions = (input_ids[0] == vision.placeholder_token_id).nonzero(as_tuple=True)[0]
    return input_ids, feats, positions, vision_ms


def run_multimodal_alloy(
    alloy_gen: AlloyGenerator, vision, tokenizer, image_bytes: bytes, prompt: str,
    *, reps: int, output_cap: int,
) -> list[MMRep]:
    """Drive the production vision path (encode + the embeds Sequence
    mode) and time each request. One untimed warmup rep compiles the
    multimodal prefill / embed plans (the vision plans are compiled by the
    generator's eager_compile_all upstream)."""

    def mm_stream(ids, feats, pos, max_new):
        return alloy_gen.run(Sequence(
            input_ids=ids, max_new_tokens=max_new, stream=True,
            embeds=MultimodalInputs(features=feats, positions=pos),
        ))

    alloy_gen.reset_prefix_state()
    ids, feats, pos, _ = _mm_encode_alloy(tokenizer, vision, image_bytes, prompt)
    for _ in mm_stream(ids, feats, pos, min(8, output_cap)):
        pass

    out: list[MMRep] = []
    for _ in range(reps):
        alloy_gen.reset_prefix_state()  # cold: every rep pays full vision + prefill
        t0 = time.perf_counter()
        ids, feats, pos, vision_ms = _mm_encode_alloy(tokenizer, vision, image_bytes, prompt)
        stream = mm_stream(ids, feats, pos, output_cap)
        try:
            next(stream)
        except StopIteration:
            now = (time.perf_counter() - t0) * 1000.0
            out.append(MMRep(vision_ms, now, 0.0, now, int(ids.shape[1]), 0))
            continue
        t_first = time.perf_counter()
        n_decoded = 1
        for _tok in stream:
            n_decoded += 1
        t_end = time.perf_counter()
        out.append(MMRep(
            vision_ms=vision_ms,
            ttft_ms=(t_first - t0) * 1000.0,
            decode_ms=(t_end - t_first) * 1000.0,
            wall_ms=(t_end - t0) * 1000.0,
            prompt_tokens=int(ids.shape[1]),
            decode_tokens=n_decoded,
        ))
    return out


# ---------------------------------------------------------------------------
# Embeddings workload — encoder throughput across batch/seq regimes
# ---------------------------------------------------------------------------
#
# In-process alloy embedding (the same `EmbeddingModel.embed` the server calls):
# tokenize + alloy-compiled encoder forward + mean-pool + L2 normalize. Per
# regime (batch x target tokens), tok/s = total real tokens / encoder-forward
# time, median over reps. Inputs slice a fixed built-in word list so the bench
# is self-contained (no corpus server).


@dataclass(slots=True)
class EmbedRegime:
    """One embedding regime: `batch` inputs of ~`seq` tokens; tok/s = total real
    tokens / encoder-forward time, median over reps."""

    regime: str
    batch: int
    seq: int
    tok_per_s: float


_EMBED_REGIMES: tuple[tuple[str, int, int], ...] = (
    ("q_short", 1, 10),    # single short query (RAG / agentic single-shot)
    ("q_long", 1, 256),    # single long-context query
    ("b8_short", 8, 10),   # bulk-index short fragments
    ("b8_long", 8, 128),   # bulk-index chunked passages
)

_EMBED_CORPUS: tuple[str, ...] = tuple(
    (
        "the quick brown fox jumps over the lazy dog while a bright moon rises "
        "above distant mountains and a quiet river winds through the green valley "
        "carrying leaves and small stones toward the wide calm sea beyond the hills "
        "where birds gather at dawn to sing across the open fields and warm light "
        "spreads slowly over rooftops streets and gardens in the waking town"
    ).split()
)


def _embed_inputs(batch: int, target_tokens: int) -> list[str]:
    """`batch` text inputs of ~`target_tokens` words each (word count ≈ token
    count for these common words), offset so batch elements differ."""
    n = max(1, target_tokens)
    pool = _EMBED_CORPUS * ((n // len(_EMBED_CORPUS)) + 1)
    span = max(1, len(pool) - n)
    return [" ".join(pool[(i * 13) % span:(i * 13) % span + n]) for i in range(batch)]


def run_embeddings_alloy(embed_model, *, reps: int) -> list[EmbedRegime]:
    """Drive the in-process alloy `EmbeddingModel.embed` across the regimes. One
    untimed warmup per regime (pins/compiles the bucket shape), then `reps` timed
    forwards; tok/s = total real tokens / median call time."""
    out: list[EmbedRegime] = []
    for name, batch, target in _EMBED_REGIMES:
        if batch > embed_model.max_batch:
            continue
        inputs = _embed_inputs(batch, target)
        n_tokens = sum(int(embed_model.count_tokens(t)) for t in inputs)
        embed_model.embed(inputs)  # warmup: compile / pin this shape
        ms: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            embed_model.embed(inputs)
            ms.append((time.perf_counter() - t0) * 1000.0)
        med = _pct(ms, 0.5)
        out.append(EmbedRegime(name, batch, target, n_tokens / max(med / 1000.0, 1e-9)))
    return out


# ---------------------------------------------------------------------------
# Per-model result aggregate + rich table rendering
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelResult:
    model: str
    alloy_depths: list[DepthPoint] | None = None
    alloy_mm: MultimodalStats | None = None
    alloy_embed: list[EmbedRegime] | None = None


def _fmt_tps(x: float) -> str:
    """t/s formatting: integer at >=100, one decimal below (12391, 456, 71.3)."""
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


def _render_depth_table(results: list[ModelResult]) -> Table | None:
    """pp/tg throughput. One depth renders pp{depth} / tg{gen} columns (one row
    per model); several render the depth-vs-throughput sweep."""
    pts = [r.alloy_depths for r in results if r.alloy_depths]
    if not pts:
        return None
    if all(len(p) == 1 for p in pts):
        ref = pts[0][0]
        t = Table(title="llama-bench — prefill / decode throughput (t/s)",
                  header_style="bold cyan", show_lines=False)
        t.add_column("model", style="bold")
        t.add_column(f"pp{ref.depth}", justify="right", style="green")
        t.add_column(f"tg{ref.gen_tokens}", justify="right", style="green")
        for mr in results:
            if not mr.alloy_depths:
                continue
            d = mr.alloy_depths[0]
            t.add_row(mr.model, _fmt_tps(d.pp_tok_per_s), _fmt_tps(d.tg_tok_per_s))
        return t
    t = Table(title="Depth sweep — pp / tg t/s vs cache depth",
              header_style="bold cyan", show_lines=False)
    t.add_column("model", style="bold")
    t.add_column("depth", justify="right")
    t.add_column("pp t/s", justify="right", style="green")
    t.add_column("tg t/s", justify="right", style="green")
    for mr in results:
        for ad in mr.alloy_depths or []:
            t.add_row(mr.model, str(ad.depth), _fmt_tps(ad.pp_tok_per_s), _fmt_tps(ad.tg_tok_per_s))
    return t


def _render_multimodal_table(results: list[ModelResult]) -> Table | None:
    """Vision request (p50): TTFT (incl. vision encode), steady-state decode
    tok/s, total wall, and the vision-encode slice on its own."""
    if not any(r.alloy_mm is not None for r in results):
        return None
    t = Table(title="Multimodal — image + prompt — Alloy (p50)",
              header_style="bold cyan", show_lines=False)
    t.add_column("model", style="bold")
    t.add_column("vision ms", justify="right", style="green")
    t.add_column("TTFT ms", justify="right", style="green")
    t.add_column("dec tok/s", justify="right", style="green")
    t.add_column("wall ms", justify="right", style="bold green")
    for mr in results:
        s = mr.alloy_mm
        if s is None:
            continue
        a = s._derived()
        t.add_row(
            mr.model,
            f"{_pct(a['vision'], 0.5):.0f}",
            f"{_pct(a['ttft'], 0.5):.0f}",
            f"{_pct(a['decode_tps'], 0.5):.1f}",
            f"{_pct(a['wall'], 0.5):.0f}",
        )
    return t


def _render_embed_table(results: list[ModelResult]) -> Table | None:
    """Embedding encoder throughput (tok/s) per batch/seq regime."""
    if not any(r.alloy_embed for r in results):
        return None
    t = Table(title="Embeddings — encoder throughput (tok/s)",
              header_style="bold cyan", show_lines=False)
    t.add_column("model", style="bold")
    t.add_column("regime")
    t.add_column("batch", justify="right")
    t.add_column("seq", justify="right")
    t.add_column("tok/s", justify="right", style="green")
    for mr in results:
        for e in mr.alloy_embed or []:
            t.add_row(mr.model, e.regime, str(e.batch), str(e.seq), _fmt_tps(e.tok_per_s))
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model refs to bench, e.g. --models qwen3:0.6b llama3.2:1b")
    p.add_argument("--dataset", choices=("synthetic", "multimodal", "embeddings"), default="synthetic",
                   help="Workload to run. synthetic: the llama-bench depth sweep (pp/tg "
                        "throughput vs cache depth). multimodal: an image + prompt through the "
                        "vision tower (requires --image and a vision-capable model). embeddings: "
                        "encoder tok/s per batch/seq regime (e.g. nomic-embed-text).")
    p.add_argument("--depths", nargs="+", type=int, default=[512],
                   help="Prefill depth(s). One value (default 512) reports pp/tg; "
                        "several (e.g. 512 4096 16384) run a depth sweep.")
    p.add_argument("--depth-gen", type=int, default=128,
                   help="Decode length for the tg measurement in the depth sweep (default 128).")
    p.add_argument("--reps", type=int, default=3,
                   help="Timed repetitions per depth point; median reported (default 3).")
    p.add_argument("--image", type=Path, default=None,
                   help="multimodal only: path to the image file to send each request.")
    p.add_argument("--mm-prompt", type=str, default=_DEFAULT_MM_PROMPT,
                   help="multimodal only: the text prompt paired with the image.")
    p.add_argument("--mm-reps", type=int, default=5,
                   help="multimodal only: number of timed requests per model (default 5).")
    p.add_argument("--mm-max-output", type=int, default=128,
                   help="multimodal only: per-request output cap / num_predict (default 128).")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON instead of tables")
    args = p.parse_args(argv)

    if any(d < 1 for d in args.depths):
        p.error("--depths values must be positive")
    args.depths = sorted(set(args.depths))
    if args.dataset == "multimodal" and args.image is None:
        p.error("--dataset multimodal requires --image PATH")

    logger.info("bench_started", models=list(args.models), dataset=args.dataset, depths=args.depths)
    _bench_t0 = time.perf_counter()
    results = [_bench_one_model(m, args) for m in args.models]
    logger.info(
        "bench_complete", n_models=len(results), dataset=args.dataset,
        total_took_s=round(time.perf_counter() - _bench_t0, 2),
    )

    if args.json:
        _emit_json(results)
    else:
        _emit_rich(results)
    return 0


def _bench_one_embed_model(model_name: str, args: argparse.Namespace) -> ModelResult:
    logger.info("model_load_started", model=model_name)
    t0 = time.perf_counter()
    embed_model = load_ollama_gguf_embedder(model_name)
    logger.info(
        "model_load_complete", model=model_name,
        took_ms=round((time.perf_counter() - t0) * 1000.0, 1),
    )
    regimes = run_embeddings_alloy(embed_model, reps=args.reps)
    del embed_model
    gc.collect()
    return ModelResult(model=model_name, alloy_embed=regimes)


def _bench_one_model(model_name: str, args: argparse.Namespace) -> ModelResult:
    print(f"\n=== {model_name} ===", file=sys.stderr)
    if args.dataset == "embeddings":
        return _bench_one_embed_model(model_name, args)
    logger.info("model_load_started", model=model_name)
    t0 = time.perf_counter()
    loaded = load_native_causal_lm(model_name)
    logger.info(
        "model_load_complete", model=model_name,
        took_ms=round((time.perf_counter() - t0) * 1000.0, 1),
    )

    # Match the server's prefill policy (large-chunk grid-shrunk prefill) so the
    # bench measures the same plans production dispatches. Hand the vision
    # front-end (if any) to the generator only for the multimodal workload so
    # eager_compile_all warms its plans without paying for them on a depth sweep.
    alloy_gen = AlloyGenerator.from_model(
        loaded.model.eval(),
        cache_dtype=torch.float16,
        chunk_prefill_size=resolve_prefill_policy(),
        vision=loaded.vision if args.dataset == "multimodal" else None,
    )
    with tqdm(
        total=None, unit="plan", desc=f"compiling {model_name}",
        leave=False, file=sys.stderr, dynamic_ncols=True,
    ) as bar:
        def _on_step(step: int, total: int, desc: str) -> None:
            if bar.total != total:
                bar.reset(total=total)
            bar.set_postfix_str(desc, refresh=False)
            bar.update(step - bar.n)
        alloy_gen.eager_compile_all(progress=_on_step)

    vocab_size = int(loaded.model.config.vocab_size)

    alloy_depths: list[DepthPoint] | None = None
    alloy_mm: MultimodalStats | None = None
    if args.dataset == "synthetic":
        if len(args.depths) == 1:
            # pp / tg as independent empty-cache tests.
            alloy_depths = [run_llama_bench_alloy(
                alloy_gen, args.depths[0], args.depth_gen, vocab_size, reps=args.reps,
            )]
        else:
            # Sweep: tg measured AT each depth.
            alloy_depths = run_depths_alloy(
                alloy_gen, args.depths, args.depth_gen, vocab_size, reps=args.reps,
            )
    elif args.dataset == "multimodal":
        if loaded.vision is None:
            raise RuntimeError(
                f"{model_name} has no vision adapter — the multimodal bench needs a "
                f"vision-capable model (e.g. gemma4:e2b)"
            )
        image_bytes = resolve_bench_image(args.image)
        reps = run_multimodal_alloy(
            alloy_gen, loaded.vision, loaded.tokenizer, image_bytes, args.mm_prompt,
            reps=args.mm_reps, output_cap=args.mm_max_output,
        )
        alloy_mm = MultimodalStats(image=str(args.image), prompt=args.mm_prompt, reps=reps)
        d = alloy_mm._derived()
        logger.info(
            "multimodal_workload_complete", model=model_name, n_reps=len(reps),
            vision_ms_p50=round(_pct(d["vision"], 0.5), 1),
            ttft_ms_p50=round(_pct(d["ttft"], 0.5), 1),
            decode_tok_per_s_p50=round(_pct(d["decode_tps"], 0.5), 1),
            wall_ms_p50=round(_pct(d["wall"], 0.5), 1),
        )

    # Drop the generator + model before loading the next one (each is ~0.5-2 GB).
    del alloy_gen, loaded
    gc.collect()
    return ModelResult(model=model_name, alloy_depths=alloy_depths, alloy_mm=alloy_mm)


def _emit_rich(results: list[ModelResult]) -> None:
    console = Console(width=max(80, Console().width))
    for renderer in (_render_depth_table, _render_multimodal_table, _render_embed_table):
        table = renderer(results)
        if table is not None:
            console.print()
            console.print(table)


def _emit_json(results: list[ModelResult]) -> None:
    def _depth_list(ds: list[DepthPoint] | None) -> list | None:
        if ds is None:
            return None
        return [
            {"depth": d.depth, "gen_tokens": d.gen_tokens,
             "pp_tok_per_s": d.pp_tok_per_s, "tg_tok_per_s": d.tg_tok_per_s}
            for d in ds
        ]
    def _mm_dict(s: MultimodalStats | None) -> dict | None:
        if s is None:
            return None
        d = s._derived()
        return {
            "image": s.image, "prompt": s.prompt, "n_reps": len(s.reps),
            "vision_ms_p50": _pct(d["vision"], 0.5),
            "ttft_ms_p50": _pct(d["ttft"], 0.5),
            "decode_tok_per_s_p50": _pct(d["decode_tps"], 0.5),
            "wall_ms_p50": _pct(d["wall"], 0.5),
            "e2e_tok_per_s_p50": _pct(d["e2e_tps"], 0.5),
            "reps": [
                {"vision_ms": r.vision_ms, "ttft_ms": r.ttft_ms, "decode_ms": r.decode_ms,
                 "wall_ms": r.wall_ms, "prompt_tokens": r.prompt_tokens,
                 "decode_tokens": r.decode_tokens}
                for r in s.reps
            ],
        }
    def _embed_list(es: list[EmbedRegime] | None) -> list | None:
        if es is None:
            return None
        return [
            {"regime": e.regime, "batch": e.batch, "seq": e.seq, "tok_per_s": e.tok_per_s}
            for e in es
        ]
    out = {
        "models": [
            {
                "model": mr.model,
                "alloy_depths": _depth_list(mr.alloy_depths),
                "alloy_multimodal": _mm_dict(mr.alloy_mm),
                "alloy_embeddings": _embed_list(mr.alloy_embed),
            }
            for mr in results
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

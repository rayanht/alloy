"""Model-reference resolution: local path / HuggingFace cache / Ollama store ->
a concrete `.gguf` file on disk."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NotRequired, TypedDict, cast

from huggingface_hub.constants import HF_HUB_CACHE

from alloy_server.gguf.arch import gguf_architecture


class OllamaLayer(TypedDict):
    mediaType: str
    digest: str
    size: NotRequired[int]


class OllamaManifest(TypedDict):
    schemaVersion: int
    mediaType: str
    config: OllamaLayer
    layers: list[OllamaLayer]


@dataclass(frozen=True, slots=True)
class ResolvedGGUF:
    """A model reference resolved to a concrete GGUF file on disk, from any
    source (local path, HuggingFace cache, or Ollama store).

    `ref` is the original user reference (display name); `path` is the `.gguf`
    file; `digest` is a stable cache key for the parsed-metadata sidecar.
    """

    ref: str
    path: Path
    digest: str
    format: ClassVar[str] = "gguf"

    @property
    def location(self) -> Path:
        return self.path

    def architecture(self) -> str:
        return gguf_architecture(self.path)


def resolve_ollama_gguf_blob(model: str, root: Path | None = None) -> ResolvedGGUF:
    """Resolve an installed Ollama model reference to its local GGUF blob."""

    models_root = root if root is not None else Path.home() / ".ollama" / "models"
    namespace, name, tag = split_ollama_ref(model)
    manifest_path = models_root / "manifests" / "registry.ollama.ai" / namespace / name / tag
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ollama manifest not found: {manifest_path}")

    manifest = cast(OllamaManifest, json.loads(manifest_path.read_text()))
    model_layer = find_model_layer(manifest["layers"])
    digest = model_layer["digest"]
    blob_path = models_root / "blobs" / digest.replace(":", "-")
    if not blob_path.exists():
        raise FileNotFoundError(f"Ollama GGUF blob not found: {blob_path}")
    return ResolvedGGUF(ref=model, path=blob_path, digest=digest)


SHARD_RE = re.compile(r"-\d{5}-of-\d{5}")


def looks_like_local_path(ref: str) -> bool:
    return ref.startswith(("./", "../", "/", "~")) or ref.endswith(".gguf")


def file_digest(path: Path) -> str:
    """Stable cache key for a non-Ollama GGUF file (Ollama files carry a content
    digest; here we key on identity + size + mtime, which is enough to invalidate
    the parsed-metadata sidecar when the file is replaced)."""
    st = path.stat()
    payload = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    return "file-" + hashlib.sha256(payload.encode()).hexdigest()[:40]


def resolve_local_gguf(ref: str) -> ResolvedGGUF:
    path = Path(ref).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    if path.suffix != ".gguf":
        raise ValueError(f"not a .gguf file: {path}")
    path = path.resolve()
    return ResolvedGGUF(ref=ref, path=path, digest=file_digest(path))


def hf_snapshot_dir(repo_id: str) -> Path | None:
    """The local HuggingFace-cache snapshot directory for `repo_id`, or None if
    the repo isn't cached. Prefers the `main` ref, else the newest snapshot."""
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text().strip()
        if (snapshots / commit).is_dir():
            return snapshots / commit
    subdirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def select_gguf_by_quant(files: list[Path], quant: str | None, *, ref: str) -> Path:
    """Pick the one GGUF a ref names. With a `:quant` tag, match files whose name
    contains the quant string; without, require exactly one GGUF (the user must
    disambiguate otherwise — we never silently pick a multi-GB variant)."""
    if quant is not None:
        matches = [f for f in files if quant.lower() in f.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            avail = ", ".join(sorted(f.name for f in files))
            raise ValueError(f"no GGUF matching quant {quant!r} for {ref!r}; available: {avail}")
        if all(SHARD_RE.search(f.name) for f in matches):
            raise ValueError(f"sharded (multi-part) GGUFs are not yet supported: {ref!r}")
        avail = ", ".join(sorted(f.name for f in matches))
        raise ValueError(f"quant {quant!r} is ambiguous for {ref!r}; matches: {avail}")
    if len(files) == 1:
        return files[0]
    avail = ", ".join(sorted(f.name for f in files))
    raise ValueError(
        f"{ref!r} has multiple GGUF files; pick one with `<repo>:<quant>`. available: {avail}"
    )


def resolve_hf_gguf(ref: str) -> ResolvedGGUF:
    repo_id, _, quant = ref.partition(":")
    snapshot = hf_snapshot_dir(repo_id)
    files = sorted(snapshot.glob("*.gguf")) if snapshot is not None else []
    if not files:
        raise FileNotFoundError(
            f"no local HuggingFace model for {repo_id!r}; download it first."
        )
    chosen = select_gguf_by_quant(files, quant or None, ref=ref)
    return ResolvedGGUF(ref=ref, path=chosen, digest=file_digest(chosen))


def resolve_gguf(ref: str, *, root: Path | None = None) -> ResolvedGGUF:
    """Resolve a model reference to a concrete GGUF file, from any source.

    Three ref shapes, checked in order:
      - local path  (`./x.gguf`, `/abs/x.gguf`, `~/x.gguf`)
      - HF repo      (`Org/Repo[:quant]`) — contains `/`
      - Ollama OCI   (`name:tag`) — bare name, back-compat

    A `/`-bearing ref that's a locally-installed Ollama namespaced model
    (e.g. `hadad/LFM2.5-1.2B:Q4_K_M`) resolves from the Ollama store first;
    otherwise it's treated as a HuggingFace repo.
    """
    ref = ref.strip()
    if looks_like_local_path(ref):
        return resolve_local_gguf(ref)
    if "/" in ref:
        try:
            return resolve_ollama_gguf_blob(ref, root=root)
        except FileNotFoundError:
            return resolve_hf_gguf(ref)
    return resolve_ollama_gguf_blob(ref, root=root)


def split_ollama_ref(model: str) -> tuple[str, str, str]:
    """Split an Ollama ref into (namespace, name, tag).

    Library models (`qwen3.5:4b`) carry no namespace and resolve under
    `library/`. Namespaced refs (`hadad/LFM2.5-1.2B:Q4_K_M`, the form ollama
    uses for user/community models) keep their `<namespace>/<name>` so the
    manifest lookup finds them under `registry.ollama.ai/<namespace>/<name>`.
    """
    if ":" in model:
        name, tag = model.split(":", 1)
    else:
        name, tag = model, "latest"
    namespace = "library"
    if "/" in name:
        namespace, name = name.rsplit("/", 1)
    if not namespace or not name or not tag:
        raise ValueError(f"Invalid Ollama model reference: {model!r}")
    return namespace, name, tag


def find_model_layer(layers: list[OllamaLayer]) -> OllamaLayer:
    for layer in layers:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return layer
    raise ValueError("Ollama manifest does not contain a GGUF model layer")

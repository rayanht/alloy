"""ASGI transport: a Starlette app + uvicorn, with a single GPU worker thread
behind an async queue.

The event loop handles HTTP; all model work runs on one dedicated worker thread
(`GenerationDispatcher`) draining a queue, which serializes generation (one model,
one GPU, one KV cache). Streaming bridges the worker's sync byte
generator to an `asyncio.Queue`; a client disconnect cancels the response, which
signals the worker to stop pulling the generator (freeing the KV slot early).
"""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import json
import os
import queue
import socket
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from alloy import get_logger
from alloy_server.dialects import (
    OLLAMA_CHAT,
    OPENAI,
    error_dialect_for_path,
    ollama,
)
from alloy_server.modality import (
    CHAT,
    DIALECT_TASKS,
    EMBED,
    ENDPOINT_INDEX,
    TASK_NOUN,
    Endpoint,
    Modality,
)
from alloy_server.schema import (
    ChatCompletionRequest,
    JsonObject,
    JsonValue,
    Rendered,
    RequestError,
    ServedModel,
)

if TYPE_CHECKING:
    from alloy_server.embedding import EmbeddingModel

logger = get_logger("alloy_server.transport")

OLLAMA_VERSION = "0.5.4-alloy"

STREAM_DONE = object()


class GenerationDispatcher:
    """One dedicated GPU worker thread draining a job queue. Async handlers submit
    work and await it over the event loop, so generation never blocks the loop and
    stays serialized (one job at a time)."""

    def __init__(self) -> None:
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self.run_jobs, daemon=True, name="alloy-gen-worker")
        self.thread.start()

    def run_jobs(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            job()

    async def run(self, fn: Callable[[], object]) -> object:
        """Run blocking `fn()` on the worker thread; await its result/exception."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def job() -> None:
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 — relayed to the awaiting handler
                loop.call_soon_threadsafe(fut.set_exception, exc)
            else:
                loop.call_soon_threadsafe(fut.set_result, result)

        self.jobs.put(job)
        return await fut

    async def stream(
        self,
        parse_fn: Callable[[], ChatCompletionRequest],
        render_fn: Callable[[ChatCompletionRequest], Iterator[bytes]],
        cancel: threading.Event,
    ) -> asyncio.Queue:
        """One worker job: parse (may raise — surfaced here BEFORE headers, so the
        caller still returns a 4xx) then run the dialect's byte generator, pushing
        frames to an asyncio.Queue which the caller drains. `cancel` set on
        disconnect makes the worker stop + close the generator."""
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        ready: asyncio.Future = loop.create_future()

        def job() -> None:
            try:
                request = parse_fn()
            except BaseException as exc:  # noqa: BLE001 — parse error -> 4xx before headers
                loop.call_soon_threadsafe(ready.set_exception, exc)
                return
            loop.call_soon_threadsafe(ready.set_result, None)
            gen = render_fn(request)
            try:
                for frame in gen:
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(out.put_nowait, frame)
            except BaseException as exc:  # noqa: BLE001 — mid-stream failure aborts the body
                loop.call_soon_threadsafe(out.put_nowait, exc)
            finally:
                gen.close()
                loop.call_soon_threadsafe(out.put_nowait, STREAM_DONE)

        self.jobs.put(job)
        await ready  # raises a parse RequestError here, before any headers go out
        return out

    def shutdown(self) -> None:
        self.jobs.put(None)


async def drain(out: asyncio.Queue, cancel: threading.Event) -> AsyncIterator[bytes]:
    try:
        while True:
            frame = await out.get()
            if frame is STREAM_DONE:
                return
            if isinstance(frame, BaseException):
                raise frame
            yield frame
    finally:
        # Body fully sent OR client disconnected (response task cancelled): tell the
        # worker to stop pulling the generator so an abandoned request frees its slot.
        cancel.set()


@dataclass(frozen=True)
class ServedState:
    served: object  # the one served model (ServedModel | EmbeddingModel | TranscriptionModel | ...)
    modality: Modality
    spec: str | None
    installed_chat_names: Callable[[], tuple[str, ...]]
    dispatcher: GenerationDispatcher

    @property
    def model_name(self) -> str:
        return self.served.name  # type: ignore[attr-defined]


def resolve_served(
    served: object | None, modality: Modality | None,
    chat_model: object | None, embedding_model: object | None,
) -> tuple[object, Modality]:
    """Normalize the served-model inputs to (served, modality). `served`/`modality`
    is the generic seam; `chat_model`/`embedding_model` are back-compat shims that
    map onto the CHAT / EMBED modalities."""
    if served is not None:
        if modality is None:
            raise ValueError("modality is required when served is given")
        return served, modality
    if chat_model is not None and embedding_model is None:
        return chat_model, CHAT
    if embedding_model is not None and chat_model is None:
        return embedding_model, EMBED
    raise ValueError("exactly one served model must be given (served+modality, or one of chat_model/embedding_model)")


def origin_allowed(origin: str, allowed: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(origin, pattern) for pattern in allowed)


def cors_headers(state: ServedState, request: Request) -> dict[str, str]:
    # Resolved per request (not cached at startup) so `ALLOY_ORIGINS` changes
    # take effect without a restart, matching the stdlib server.
    origin = request.headers.get("origin")
    if origin and origin_allowed(origin, resolve_cors_origins()):
        return {"access-control-allow-origin": origin, "vary": "origin"}
    return {}


def json_response(state: ServedState, request: Request, status: HTTPStatus, payload: JsonObject) -> Response:
    return JSONResponse(payload, status_code=int(status), headers=cors_headers(state, request))


def error_response(state: ServedState, request: Request, path: str, err: RequestError) -> Response:
    logger.warning("request_failed", path=path, status=int(err.status), code=err.code, message=err.message)
    body = error_dialect_for_path(path).render_error(err.status, err.code, err.message)
    return json_response(state, request, err.status, body)


async def read_json_object(request: Request) -> JsonObject:
    raw = await request.body()
    try:
        payload: JsonValue = json.loads(raw.decode() if raw else "{}")
    except json.JSONDecodeError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "request body must be a JSON object")
    return payload


async def read_multipart(request: Request) -> dict:
    """Parse a multipart/form-data body into a dict: file uploads become bytes, all
    other fields stay strings. Used by the audio endpoints (`file` upload + params)."""
    form = await request.form()
    data: dict = {}
    for key in form:
        value = form[key]
        data[key] = await value.read() if isinstance(value, UploadFile) else value
    return data


def served_model_for(state: ServedState, payload: JsonObject, endpoint: Endpoint) -> object:
    """Resolve the served model for an endpoint: 404 if the endpoint's task isn't
    what this server serves (cross-kind), or if a different model name is requested."""
    if endpoint.task not in state.modality.tasks:
        raise RequestError(
            HTTPStatus.NOT_FOUND, "model_not_served",
            f"this server serves {state.modality.kind} model {state.model_name!r}, not "
            f"a {TASK_NOUN.get(endpoint.task, endpoint.task)} model; start one with "
            f"`alloy serve -m <model>`",
        )
    model = state.served
    requested = payload.get("model")
    if isinstance(requested, str) and requested and requested != state.model_name:
        raise RequestError(
            HTTPStatus.NOT_FOUND, "model_not_served",
            f"this server serves {state.model_name!r}; to use {requested!r}, "
            f"restart with `alloy serve -m {requested}`",
        )
    return model


def build_routes(state: ServedState) -> list[Route]:
    async def get_handler(request: Request) -> Response:
        path = request.url.path
        if path == "/v1/models":
            return json_response(state, request, HTTPStatus.OK, OPENAI.render_catalog(state.installed_chat_names()))
        if path == "/api/tags":
            return json_response(state, request, HTTPStatus.OK, OLLAMA_CHAT.render_catalog(state.installed_chat_names()))
        if path == "/api/version":
            return json_response(state, request, HTTPStatus.OK, {"version": OLLAMA_VERSION})
        if path == "/healthz":
            return json_response(state, request, HTTPStatus.OK, {
                "status": "ok", "version": OLLAMA_VERSION,
                "model": state.model_name, "kind": state.modality.kind, "spec": state.spec,
            })
        return error_response(state, request, path,
                              RequestError(HTTPStatus.NOT_FOUND, "not_found", f"unknown endpoint: {path}"))

    async def post_handler(request: Request) -> Response:
        path = request.url.path
        try:
            endpoint = ENDPOINT_INDEX.get(path)
            if endpoint is not None:
                handler = DIALECT_TASKS[(endpoint.dialect, endpoint.task)]
                payload = (
                    await read_multipart(request) if endpoint.request == "multipart"
                    else await read_json_object(request)
                )
                model = served_model_for(state, payload, endpoint)
                if handler.render_stream is not None and handler.wants_stream is not None \
                        and handler.wants_stream(payload):
                    parse, render_stream = handler.parse, handler.render_stream
                    cancel = threading.Event()
                    out = await state.dispatcher.stream(
                        lambda: parse(model, payload), render_stream, cancel,
                    )
                    # content-type set via headers (not media_type) so Starlette
                    # doesn't append `; charset=utf-8` to the SSE/NDJSON type.
                    return StreamingResponse(
                        drain(out, cancel),
                        headers={
                            "content-type": handler.stream_content_type,
                            "cache-control": "no-cache",
                            **cors_headers(state, request),
                        },
                    )
                result = await state.dispatcher.run(lambda: handler.render(model, payload))
                if isinstance(result, Rendered):
                    return Response(result.body, status_code=int(HTTPStatus.OK), headers={
                        "content-type": result.content_type, **cors_headers(state, request),
                    })
                return json_response(state, request, HTTPStatus.OK, result)
            if path == "/api/show":
                payload = await read_json_object(request)
                name = payload.get("name", payload.get("model"))
                if isinstance(name, str) and name and name != state.model_name:
                    raise RequestError(
                        HTTPStatus.NOT_FOUND, "model_not_served",
                        f"this server serves {state.model_name!r}; to use {name!r}, "
                        f"restart with `alloy serve -m {name}`",
                    )
                return json_response(state, request, HTTPStatus.OK, ollama.show_payload())
            if path == "/api/pull":
                raise RequestError(
                    HTTPStatus.NOT_IMPLEMENTED, "not_supported",
                    "alloy does not download or manage models; fetch a GGUF with "
                    "`hf download <org>/<repo> <file>.gguf` (or use an existing "
                    "Ollama install), then `alloy serve -m <ref>`.",
                )
            if path in ("/api/create", "/api/push") or path.startswith("/api/blobs/"):
                raise RequestError(
                    HTTPStatus.NOT_IMPLEMENTED, "not_supported",
                    f"{path} is not supported by alloy v1.",
                )
        except RequestError as err:
            return error_response(state, request, path, err)
        return error_response(state, request, path,
                              RequestError(HTTPStatus.NOT_FOUND, "not_found", f"unknown endpoint: {path}"))

    async def options_handler(request: Request) -> Response:
        return Response(status_code=int(HTTPStatus.NO_CONTENT), headers={
            **cors_headers(state, request),
            "access-control-allow-methods": "GET, POST, OPTIONS",
            "access-control-allow-headers": "*",
            "access-control-max-age": "86400",
        })

    get_paths = ["/v1/models", "/api/tags", "/api/version", "/healthz"]
    # Model endpoints come from the modality registry (every modality's endpoints are
    # routed; the served one decides which run); management endpoints are explicit.
    post_paths = list(ENDPOINT_INDEX) + ["/api/show", "/api/pull", "/api/create", "/api/push"]
    routes = [Route(p, get_handler, methods=["GET"]) for p in get_paths]
    routes += [Route(p, post_handler, methods=["POST"]) for p in post_paths]
    # Catch-all so OPTIONS preflight, blob uploads, and unknown paths/methods still
    # route to a handler that branches by method (instead of Starlette's bare 405).
    # DELETE/PUT/PATCH fall through post_handler to a 404 (no such endpoint).
    routes.append(Route("/{rest:path}", post_handler, methods=["POST", "DELETE", "PUT", "PATCH"]))
    routes.append(Route("/{rest:path}", get_handler, methods=["GET", "HEAD"]))
    routes.append(Route("/{rest:path}", options_handler, methods=["OPTIONS"]))
    return routes


def create_app(state: ServedState) -> Starlette:
    return Starlette(routes=build_routes(state))


# Default CORS allow-list. `ALLOY_ORIGINS` extends it, or replaces it when the
# value starts with `=`. Patterns are fnmatch-style.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:*",
    "https://localhost:*",
    "http://127.0.0.1:*",
    "https://127.0.0.1:*",
    "http://0.0.0.0:*",
    "app://obsidian.md",
    "tauri://*",
)


def resolve_cors_origins() -> tuple[str, ...]:
    """CORS allow-list from `ALLOY_ORIGINS`: unset → default; `=p1,p2` → replace;
    `p1,p2` → extend."""
    raw = os.environ.get("ALLOY_ORIGINS")
    if not raw:
        return DEFAULT_CORS_ORIGINS
    if raw.startswith("="):
        return tuple(p.strip() for p in raw[1:].split(",") if p.strip())
    return DEFAULT_CORS_ORIGINS + tuple(p.strip() for p in raw.split(",") if p.strip())


class PortCollisionError(RuntimeError):
    """Raised when our bind port is already taken."""


class AlloyServer:
    """Foreground ASGI server over a single model, fixed at startup. Binds the
    listen socket up front (so the port is known immediately and EADDRINUSE
    surfaces as a clean `PortCollisionError`), then runs uvicorn on it. Exactly one
    of chat_model / embedding_model is set."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        served: object | None = None,
        modality: Modality | None = None,
        chat_model: ServedModel | None = None,
        embedding_model: "EmbeddingModel | None" = None,
        spec: str | None = None,
        installed_chat_names: Callable[[], tuple[str, ...]] = lambda: (),
    ) -> None:
        served, modality = resolve_served(served, modality, chat_model, embedding_model)
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((host, port))
        except PermissionError as exc:
            raise PortCollisionError(
                f"port {port} requires elevated privileges (try a port >= 1024)",
            ) from exc
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise PortCollisionError(
                    f"port {port} is already in use by another process "
                    f"(run `lsof -nP -iTCP:{port}` to identify)",
                ) from exc
            raise
        self.sock.listen(128)
        self.port = int(self.sock.getsockname()[1])
        self.dispatcher = GenerationDispatcher()
        state = ServedState(
            served=served, modality=modality, spec=spec,
            installed_chat_names=installed_chat_names, dispatcher=self.dispatcher,
        )
        config = uvicorn.Config(create_app(state), log_level="warning", access_log=False, lifespan="off")
        self.server = uvicorn.Server(config)

    @property
    def server_address(self) -> tuple[str, int]:
        return (self.host, self.port)

    @property
    def server_port(self) -> int:
        return self.port

    def serve_forever(self) -> None:
        self.server.run(sockets=[self.sock])

    def shutdown(self) -> None:
        self.server.should_exit = True
        self.dispatcher.shutdown()

    def server_close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

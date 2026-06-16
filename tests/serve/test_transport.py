"""Unit tests for the ASGI transport's GPU worker bridge (GenerationDispatcher).

No HTTP, no model: drive the dispatcher directly to pin the run/stream contract
and the cancellation path — a disconnect (the async body closed) must stop the
worker and close the sync generator so an abandoned request frees its slot."""

from __future__ import annotations

import asyncio
import threading

import pytest

from alloy_server.schema import RequestError
from alloy_server.transport import GenerationDispatcher, drain


def test_dispatcher_run_returns_result():
    async def go():
        d = GenerationDispatcher()
        try:
            return await d.run(lambda: 6 * 7)
        finally:
            d.shutdown()

    assert asyncio.run(go()) == 42


def test_dispatcher_run_propagates_exception():
    async def go():
        d = GenerationDispatcher()
        try:
            await d.run(lambda: (_ for _ in ()).throw(RequestError(400, "x", "boom")))
        finally:
            d.shutdown()

    with pytest.raises(RequestError):
        asyncio.run(go())


def test_stream_yields_all_frames():
    def render(_request):
        yield b"a"
        yield b"b"
        yield b"c"

    async def go():
        d = GenerationDispatcher()
        try:
            cancel = threading.Event()
            out = await d.stream(lambda: "req", render, cancel)
            return [frame async for frame in drain(out, cancel)]
        finally:
            d.shutdown()

    assert asyncio.run(go()) == [b"a", b"b", b"c"]


def test_stream_parse_error_surfaces_before_frames():
    def render(_request):  # never reached
        yield b"unreachable"

    async def go():
        d = GenerationDispatcher()
        try:
            def parse():
                raise RequestError(404, "model_not_served", "nope")
            await d.stream(parse, render, threading.Event())
        finally:
            d.shutdown()

    with pytest.raises(RequestError):
        asyncio.run(go())


def test_stream_cancellation_closes_generator():
    """Client disconnect → the drained async-gen is closed → cancel is set → the
    worker stops pulling and closes the sync generator (GeneratorExit), so the
    request frees its slot instead of running to completion."""
    started = threading.Event()
    proceed = threading.Event()
    closed = threading.Event()

    def render(_request):
        try:
            started.set()
            yield b"chunk0"
            proceed.wait(timeout=5)  # park mid-stream until the test releases us
            yield b"chunk1"
        finally:
            closed.set()  # set by gen.close()'s GeneratorExit

    async def go():
        d = GenerationDispatcher()
        cancel = threading.Event()
        out = await d.stream(lambda: "req", render, cancel)
        agen = drain(out, cancel)
        first = await agen.__anext__()
        await agen.aclose()              # simulate the client disconnecting
        proceed.set()                    # let the worker advance past its park
        for _ in range(200):             # wait for the worker to observe cancel
            if closed.is_set():
                break
            await asyncio.sleep(0.01)
        d.shutdown()
        return first, cancel.is_set(), closed.is_set()

    first, cancelled, gen_closed = asyncio.run(go())
    assert first == b"chunk0"
    assert cancelled        # drain's finally set cancel on aclose
    assert gen_closed       # worker closed the generator early — slot freed

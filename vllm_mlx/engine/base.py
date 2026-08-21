# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for vllm-mlx inference.
"""

import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    mtp_drafts: int = 0
    mtp_accepted: int = 0
    # For streaming
    new_text: str = ""
    finished: bool = True
    # MTP speculative decoding counters. Zero means no MTP attempt occurred.
    mtp_drafts: int = 0
    mtp_accepted: int = 0


class EngineBusy(RuntimeError):
    """Raised when a serialized engine route is already serving a request."""

    code = "text_generation_busy"


class EngineStopped(RuntimeError):
    """Raised when generation work is submitted after the engine has stopped.

    Engines that pin MLX work to one thread tear that thread down in ``stop()``.
    Anything still holding a reference to it gets a defined error instead of the
    raw ``RuntimeError("cannot schedule new futures after shutdown")`` that
    ``ThreadPoolExecutor`` raises, so callers can tell a shutdown apart from a
    real generation failure.
    """

    code = "engine_stopped"


@contextmanager
def suspend_cancellation():
    """Temporarily clear task cancellation so cleanup can finish deterministically."""
    task = asyncio.current_task()
    if task is None:
        yield
        return

    cancelling = getattr(task, "cancelling", None)
    uncancel = getattr(task, "uncancel", None)
    if cancelling is None or uncancel is None:
        yield
        return

    pending_cancels = cancelling()
    for _ in range(pending_cancels):
        uncancel()
    try:
        yield
    finally:
        for _ in range(pending_cancels):
            task.cancel()


# Per-task bookkeeping for shield_task(), keyed weakly so a task that's
# never (fully) shielded to completion doesn't outlive its own lifetime
# because of this cache. Each entry's "waiters" list holds every waiter
# Future currently interested in `task`'s outcome -- shield_task() may be
# called concurrently, or repeatedly in a retry loop, for the same task.
_shield_waiters: "weakref.WeakKeyDictionary[asyncio.Task, list[asyncio.Future]]" = (
    weakref.WeakKeyDictionary()
)


async def shield_task(task: asyncio.Task) -> Any:
    """Equivalent to ``await asyncio.shield(task)``, minus one asyncio quirk.

    On CPython 3.14+, asyncio.shield()'s implementation unconditionally
    reassigns `task`'s done-callback to an internal "log on exception"
    handler the instant its wrapper future is cancelled (CPython
    asyncio.tasks.shield / _outer_done_callback / _log_on_exception). That
    handler calls loop.call_exception_handler() for whatever `task`
    eventually raises, regardless of whether the caller goes on to retrieve
    that exception itself afterward -- and pytest-anyio (and similar loop
    exception hooks) treat that call as an unhandled-exception test failure
    even when the exception is fully expected and handled.

    This does not reproduce on CPython 3.10-3.13: there, shield() uses two
    callbacks instead of one -- an outer-cancel callback that removes the
    inner task's done-callback once the caller cancels (so a later `await
    task` by the caller is the only thing that retrieves the exception),
    and, for the race where the inner task finishes before that removal
    lands, the inner done-callback itself still fires but explicitly calls
    `inner.exception()` to mark it retrieved before discarding it -- so
    nothing is ever logged. This helper's value is therefore specific to
    CPython 3.14+ (see the dedicated ``test-python-3-14`` CI job).

    Two more invariants, both required by callers that re-shield the same
    still-running `task` in a retry loop (see run_blocking_startup_work):

    - If `task` is already done, this returns/raises immediately via a
      plain `await task` -- matching asyncio.shield()'s own fast path --
      instead of adding a callback and waiter that cost an extra event
      loop turn.
    - Exactly one done-callback is ever registered on `task`, for its
      entire lifetime, regardless of how many times it gets shielded and
      cancelled. A caller that re-shields the same task after every
      cancellation (exactly what the retry loop above does) only adds and
      removes its own *waiter* from a shared per-task waiter list --
      never a fresh done-callback -- so callback count on `task` never
      grows with the number of cancellations. A naive add-then-remove
      done-callback per call would bound growth to "at most one at a
      time" too, but only by *dropping* the exception-retrieval guarantee
      the moment nobody is left waiting; keeping the one callback
      permanently attached (it self-retrieves via `completed_task.exception()`
      when the waiter list is empty) preserves that guarantee unconditionally,
      not just for callers that happen to retry.
    """
    if task.done():
        return await task  # matches asyncio.shield()'s own fast path

    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    # Shared per-task waiter list: repeated/concurrent shield_task(task)
    # calls reuse the one _propagate callback below instead of each adding
    # their own, bounding callback growth on `task` to exactly one.
    waiters = _shield_waiters.get(task)
    if waiters is None:
        waiters = []
        _shield_waiters[task] = waiters

        def _propagate(completed_task: asyncio.Task) -> None:
            pending, waiters[:] = waiters[:], []
            if not pending:
                if not completed_task.cancelled():
                    # Nobody is waiting right now -- mark the exception
                    # retrieved so a later GC of `completed_task` doesn't
                    # log "Task exception was never retrieved".
                    completed_task.exception()
                return
            if completed_task.cancelled():
                # `task` itself was cancelled directly (not via a
                # shield_task() awaiter -- cancelling those only cancels
                # their own `waiter`, see below). Propagate to every waiter.
                for w in pending:
                    if not w.done():
                        w.cancel()
                return
            exc = completed_task.exception()
            for w in pending:
                if w.done():
                    continue
                if exc is not None:
                    w.set_exception(exc)
                else:
                    w.set_result(completed_task.result())

        task.add_done_callback(_propagate)

    waiters.append(waiter)
    try:
        return await waiter
    except asyncio.CancelledError:
        # Only our own waiter is cancelled here, not `task` itself.
        if waiter in waiters:
            waiters.remove(waiter)
        raise


async def run_blocking_startup_work(
    work: Callable[[], Any], executor: Any | None = None
) -> None:
    """Run blocking startup work off-loop without leaking cancellation races.

    Pass ``executor`` to pin the work to a specific thread. MLX buffers carry
    the stream of the thread that built them, so a model must be loaded on the
    same thread that later generates from it; ``None`` keeps the previous
    behaviour of using asyncio's default thread pool.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(loop.run_in_executor(executor, work))
    try:
        await shield_task(task)
    except asyncio.CancelledError:
        # `task` (e.g. an in-progress model load) must run to completion even
        # though our own caller gave up -- keep re-shielding it, ignoring
        # further cancels of *this* coroutine, until it's actually done.
        with suspend_cancellation():
            while not task.done():
                try:
                    await shield_task(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break  # task's own exception -- already retrieved above
        raise


async def cleanup_startup_cancellation(cleanup: Callable[[], Awaitable[None]]) -> None:
    """Run startup cleanup without letting cleanup failures replace cancellation."""
    with suspend_cancellation():
        try:
            await cleanup()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error(
                "Engine startup cleanup failed while preserving cancellation",
                exc_info=(type(exc), exc, exc.__traceback__),
            )


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @property
    def preserve_native_tool_format(self) -> bool:
        """
        Whether to preserve native tool message format.

        When True, role="tool" messages and tool_calls fields are preserved
        instead of being converted to text. Set by server based on tool parser.
        """
        return getattr(self, "_preserve_native_tool_format", False)

    @preserve_native_tool_format.setter
    def preserve_native_tool_format(self, value: bool) -> None:
        self._preserve_native_tool_format = value

    def prepare_for_start(self) -> None:
        """Run blocking startup work before async engine start.

        Engines can override this to perform heavyweight synchronous model
        loads off the serving event loop. The default implementation is a
        no-op so lightweight engines do not need extra plumbing.
        """
        return None

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics. Override in subclasses."""
        return {}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics. Override in subclasses."""
        return None

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches. Override in subclasses."""
        return None

    async def abort_request(self, request_id: str) -> bool:
        """Abort an active or queued request when the engine supports it."""
        return False

# SPDX-License-Identifier: Apache-2.0
"""Streaming must not drop tokens when --stream-interval > 1.

Regression for the GLM-5.2 "word salad" bug: with stream_interval=5 the engine
forwarded only every 5th per-step RequestOutput to the collector and dropped the
4 in between. Since each step's RequestOutput carries only that step's ~1-token
``new_text``, 4 of every 5 tokens' text was silently lost, so streamed output
looked like scrambled fragments (e.g. "1alyze  Count by ...").

The collector already aggregates ``new_text`` on ``put()``; the interval must
gate *notification/release* to the consumer, never drop the accumulated text.
"""

from vllm_mlx.output_collector import RequestOutputCollector, RequestStreamState
from vllm_mlx.request import RequestOutput


def _step(i: int, n: int) -> RequestOutput:
    """A single decode step emitting token ``i`` of ``n`` (text ``"i "``)."""
    return RequestOutput(
        request_id="r",
        new_token_ids=[i],
        new_text=f"{i} ",
        output_token_ids=list(range(1, i + 1)),
        output_text="".join(f"{j} " for j in range(1, i + 1)),
        finished=(i == n),
        finish_reason="stop" if i == n else None,
        completion_tokens=i,
    )


def _drive(n_tokens: int, interval: int):
    """Mimic the engine loop + a consumer draining the collector each step."""
    collector = RequestOutputCollector(aggregate=True)
    state = RequestStreamState(stream_interval=interval)
    received = []
    releases = 0
    for i in range(1, n_tokens + 1):
        out = _step(i, n_tokens)
        send = state.should_send(out.completion_tokens, out.finished)
        collector.put(out, notify=send)
        if send:
            state.mark_sent(out.completion_tokens)
        got = collector.get_nowait()
        if got is not None:
            received.append(got.new_text)
            releases += 1
    return received, releases


def test_no_text_dropped_across_interval():
    expected = "".join(f"{j} " for j in range(1, 13))  # 12 tokens
    received, _ = _drive(12, interval=5)
    assert "".join(received) == expected


def test_interval_batches_releases():
    # 12 tokens, interval 5: releases at token 1 (first), 6, 11, and 12 (finish).
    _, releases = _drive(12, interval=5)
    assert releases == 4


def test_get_nowait_holds_back_until_notified():
    collector = RequestOutputCollector(aggregate=True)
    collector.put(_step(2, 12), notify=False)  # mid-interval, not released yet
    assert collector.get_nowait() is None
    collector.put(_step(3, 12), notify=True)  # interval boundary
    out = collector.get_nowait()
    assert out is not None
    # both the held-back and the boundary token's text are present
    assert out.new_text == "2 3 "


def test_interval_one_releases_every_token():
    received, releases = _drive(5, interval=1)
    assert "".join(received) == "1 2 3 4 5 "
    assert releases == 5


def _drive_lagging(n_tokens: int, interval: int, drain_at: set[int]):
    """Same loop, but the consumer only drains on the listed steps.

    A real SSE consumer is a coroutine that may not be scheduled every engine
    step, so releases pile up in the collector and get merged by ``put()``'s
    aggregation. Nothing may be lost across that merge -- including the
    terminal ``finished`` / ``finish_reason``.
    """
    collector = RequestOutputCollector(aggregate=True)
    state = RequestStreamState(stream_interval=interval)
    received = []
    for i in range(1, n_tokens + 1):
        out = _step(i, n_tokens)
        send = state.should_send(out.completion_tokens, out.finished)
        collector.put(out, notify=send)
        if send:
            state.mark_sent(out.completion_tokens)
        if i in drain_at:
            got = collector.get_nowait()
            if got is not None:
                received.append(got)
    # Whatever the consumer has not picked up yet, it picks up at the end.
    tail = collector.get_nowait()
    if tail is not None:
        received.append(tail)
    return received


def test_lagging_consumer_loses_no_text():
    """Consumer never runs until the request is over."""
    expected = "".join(f"{j} " for j in range(1, 13))
    received = _drive_lagging(12, interval=5, drain_at=set())
    assert len(received) == 1, "everything merged into one pending output"
    assert received[0].new_text == expected


def test_lagging_consumer_still_sees_the_finish():
    received = _drive_lagging(12, interval=5, drain_at=set())
    assert received[-1].finished is True
    assert received[-1].finish_reason == "stop"


def test_finish_survives_a_merge_with_earlier_releases():
    """Finish lands mid-interval (token 12, last boundary was 11).

    The consumer drains at step 6 only, so the token-11 release and the
    token-12 finish merge; the merged output must carry both the text and
    the terminal state.
    """
    received = _drive_lagging(12, interval=5, drain_at={6})
    assert "".join(r.new_text for r in received) == "".join(
        f"{j} " for j in range(1, 13)
    )
    assert received[-1].finished is True
    assert received[-1].finish_reason == "stop"


def test_run_shorter_than_the_interval_still_delivers():
    """3 tokens at interval 5 never reaches a boundary -- finish must release."""
    received = _drive_lagging(3, interval=5, drain_at=set())
    assert "".join(r.new_text for r in received) == "1 2 3 "
    assert received[-1].finished is True


def test_single_token_response_delivers():
    received = _drive_lagging(1, interval=5, drain_at=set())
    assert "".join(r.new_text for r in received) == "1 "
    assert received[-1].finished is True
    assert received[-1].finish_reason == "stop"


def test_cumulative_output_text_is_the_latest_not_the_concatenation():
    """_merge_outputs keeps the newest cumulative fields, not a sum of them."""
    received = _drive_lagging(12, interval=5, drain_at=set())
    merged = received[-1]
    assert merged.output_text == "".join(f"{j} " for j in range(1, 13))
    assert merged.completion_tokens == 12

#!/usr/bin/env python3
"""
OpenClaw TUI simulator — tests event loop responsiveness and request queuing.

Verifies two scenarios:
  1. ESC (disconnect) while generating → generation stops, server responsive
  2. New request while generating → queued, starts after current finishes

Usage:
    python3.12 tests/test_preemption_simulator.py
    python3.12 tests/test_preemption_simulator.py --url http://localhost:8000
"""

import argparse
import asyncio
import json
import logging
import time

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulator")

T0 = 0.0


def wall() -> str:
    return f"T+{time.perf_counter() - T0:.3f}s"


def fmt(t):
    return f"{t:.3f}s" if t is not None else "N/A"


async def stream_completions(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    label: str,
    max_tokens: int = 512,
    first_token_event: asyncio.Event | None = None,
    abort_after_tokens: int | None = None,
) -> dict:
    """
    Send a streaming /v1/completions request.
    If abort_after_tokens is set, close the connection after receiving
    that many tokens (simulates ESC in OpenClaw).
    """
    body = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    text_parts = []
    finish_reason = None
    aborted = False

    try:
        async with session.post(
            f"{url}/v1/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                text = choices[0].get("text", "")
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                if text:
                    token_count += 1
                    text_parts.append(text)
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        log.info(
                            f"[{wall()}] [{label}] First token at "
                            f"+{t_first_token - t_start:.3f}s"
                        )
                        if first_token_event:
                            first_token_event.set()

                    if abort_after_tokens and token_count >= abort_after_tokens:
                        log.info(
                            f"[{wall()}] [{label}] Aborting after "
                            f"{token_count} tokens (simulating ESC)"
                        )
                        aborted = True
                        break

    except aiohttp.ClientError as e:
        log.info(f"[{wall()}] [{label}] Connection error: {e}")
    except Exception as e:
        log.warning(f"[{wall()}] [{label}] Error: {type(e).__name__}: {e}")

    t_end = time.perf_counter()

    return {
        "t_start": t_start,
        "t_first_token": t_first_token,
        "t_end": t_end,
        "ttft": (t_first_token - t_start) if t_first_token else None,
        "total_time": t_end - t_start,
        "token_count": token_count,
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
        "aborted": aborted,
    }


async def stream_chat(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    label: str,
    max_tokens: int = 64,
) -> dict:
    """Send a streaming /v1/chat/completions request."""
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_chunk = None
    t_first_visible = None
    chunk_count = 0
    visible_count = 0
    text_parts = []
    finish_reason = None

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                chunk_count += 1
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                if content:
                    visible_count += 1
                    text_parts.append(content)
                    if t_first_visible is None:
                        t_first_visible = time.perf_counter()
                        log.info(
                            f"[{wall()}] [{label}] First visible at "
                            f"+{t_first_visible - t_start:.3f}s: "
                            f"{content[:50]!r}"
                        )

    except Exception as e:
        log.warning(f"[{wall()}] [{label}] Error: {type(e).__name__}: {e}")

    t_end = time.perf_counter()

    return {
        "t_start": t_start,
        "t_first_chunk": t_first_chunk,
        "t_first_visible": t_first_visible,
        "t_end": t_end,
        "ttft_visible": (t_first_visible - t_start) if t_first_visible else None,
        "total_time": t_end - t_start,
        "chunk_count": chunk_count,
        "visible_count": visible_count,
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
    }


async def test_event_loop_responsive(session, url: str) -> bool:
    """
    Verify that the event loop stays responsive during generation.
    While A is generating tokens, a simple GET /v1/models should respond
    quickly (not blocked for 100+ seconds like before the fix).
    """
    log.info(f"\n[{wall()}] --- Test: Event loop responsiveness ---")

    a_started = asyncio.Event()

    # Start long generation
    task_a = asyncio.create_task(
        stream_completions(
            session, url,
            "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
            label="A", max_tokens=256,
            first_token_event=a_started,
        )
    )

    await asyncio.wait_for(a_started.wait(), timeout=60)
    await asyncio.sleep(0.5)

    # While A is generating, do a simple GET
    log.info(f"[{wall()}] Sending GET /v1/models while A is generating...")
    t_get = time.perf_counter()
    async with session.get(
        f"{url}/v1/models",
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        await resp.json()
    get_time = time.perf_counter() - t_get
    log.info(f"[{wall()}] GET /v1/models responded in {get_time:.3f}s")

    await task_a

    if get_time < 2.0:
        log.info(f"OK: Event loop responsive ({get_time:.3f}s)")
        return True
    else:
        log.error(f"FAIL: Event loop blocked for {get_time:.1f}s")
        return False


async def test_esc_disconnect(session, url: str) -> bool:
    """
    Scenario 1: ESC while generating.
    Client disconnects mid-stream. Server should stop generation
    and accept a new request quickly.
    """
    log.info(f"\n[{wall()}] --- Test: ESC (disconnect) while generating ---")

    a_started = asyncio.Event()

    # Start generation, abort after 30 tokens (simulate ESC)
    result_a = await stream_completions(
        session, url,
        "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
        label="A-esc", max_tokens=1024,
        first_token_event=a_started,
        abort_after_tokens=30,
    )
    log.info(
        f"[{wall()}] A aborted: {result_a['token_count']} tokens, "
        f"aborted={result_a['aborted']}"
    )

    # Wait for server to detect disconnect and release the lock
    await asyncio.sleep(1.0)

    # Send a new request — should start quickly (not blocked)
    log.info(f"[{wall()}] Sending follow-up request after ESC...")
    result_b = await stream_completions(
        session, url, "Hello", label="B-after-esc", max_tokens=10,
    )

    b_ttft = result_b["ttft"]
    log.info(
        f"[{wall()}] B: TTFT={fmt(b_ttft)}, "
        f"tokens={result_b['token_count']}"
    )

    if b_ttft is not None and b_ttft < 5.0:
        log.info(f"OK: After ESC, next request started in {fmt(b_ttft)}")
        return True
    else:
        log.error(f"FAIL: After ESC, next request TTFT={fmt(b_ttft)}")
        return False


async def test_queue_new_request(session, url: str) -> bool:
    """
    Scenario 2: New request while generating.
    Query2 should be queued, not preempt query1. Query1 finishes
    normally, then query2 starts.
    """
    log.info(f"\n[{wall()}] --- Test: Queue new request (no preemption) ---")

    a_started = asyncio.Event()

    # A: generate 128 tokens
    log.info(f"[{wall()}] [A] Sending generation (max_tokens=128)...")
    task_a = asyncio.create_task(
        stream_completions(
            session, url,
            "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
            label="A-gen", max_tokens=128,
            first_token_event=a_started,
        )
    )

    await asyncio.wait_for(a_started.wait(), timeout=60)
    await asyncio.sleep(1.0)

    # B: send while A is still generating (like typing query2 + Enter)
    log.info(f"[{wall()}] [B] Sending queued request while A generates...")
    t_b_sent = time.perf_counter()
    task_b = asyncio.create_task(
        stream_completions(
            session, url, "Hello", label="B-queued", max_tokens=10,
        )
    )

    result_a, result_b = await asyncio.gather(task_a, task_b)

    a_finish = result_a["finish_reason"]
    a_tokens = result_a["token_count"]
    b_ttft = (
        (result_b["t_first_token"] - t_b_sent)
        if result_b["t_first_token"] else None
    )

    log.info(
        f"[A] tokens={a_tokens}, finish={a_finish}, "
        f"total={result_a['total_time']:.2f}s"
    )
    log.info(
        f"[B] TTFT(from send)={fmt(b_ttft)}, "
        f"tokens={result_b['token_count']}"
    )

    passed = True

    # A should finish normally (not be preempted)
    if a_tokens >= 120:  # Close to max_tokens=128
        log.info(f"OK: A completed normally ({a_tokens} tokens)")
    else:
        log.error(
            f"FAIL: A was preempted ({a_tokens}/128 tokens) — "
            "should have completed fully"
        )
        passed = False

    # B should start after A finishes (queued, not preempted)
    if result_b["t_first_token"] and result_a["t_end"]:
        b_started_after_a = result_b["t_first_token"] >= result_a["t_end"] - 0.5
        if b_started_after_a:
            log.info("OK: B started after A finished (queued correctly)")
        else:
            log.warning(
                "WARN: B started before A finished — "
                "possible preemption instead of queuing"
            )

    return passed


async def run_all_tests(url: str) -> bool:
    global T0
    T0 = time.perf_counter()

    log.info("=" * 60)
    log.info("OpenClaw TUI Simulator — Event Loop & Queuing Tests")
    log.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        # Check server
        try:
            async with session.get(
                f"{url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                models = await resp.json()
                log.info(f"Server OK: {[m['id'] for m in models.get('data', [])]}")
        except Exception as e:
            log.error(f"Server not available: {e}")
            return False

        # Warmup
        log.info(f"\n[{wall()}] Warmup...")
        await stream_completions(session, url, "Hi", label="warmup", max_tokens=5)

        # Run tests
        results = []
        results.append(await test_event_loop_responsive(session, url))
        results.append(await test_esc_disconnect(session, url))
        results.append(await test_queue_new_request(session, url))

        log.info(f"\n{'='*60}")
        passed = all(results)
        if passed:
            log.info(f"ALL PASSED ({sum(results)}/{len(results)})")
        else:
            log.error(
                f"FAILED ({sum(results)}/{len(results)} passed)"
            )
        log.info("=" * 60)
        return passed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    success = await run_all_tests(args.url)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

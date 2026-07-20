#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# openclaw_tui_jaeger_benchmark.py
#
# Multi-agent + multi-tool benchmark for OpenClaw that
#   (1) drives the interactive OpenClaw **TUI** (`openclaw chat`) over a real
#       pseudo-terminal (pty), because the TUI needs a TTY and ignores a plain
#       stdin pipe, and
#   (2) reads all KPIs from **Jaeger traces** (the OTLP spans exported by the
#       `diagnostics-otel` plugin), instead of the JSONL diagnostic logs.
#
# Why traces / Jaeger:
#   * The `diagnostics-otel` plugin exports OTLP spans; point it at Jaeger
#     (or an OTLP collector feeding Jaeger). Jaeger exposes a simple JSON query
#     API (`/api/traces`) so this script needs only the Python standard library
#     (urllib) — no pip packages, no heavy client.
#   * Spans already carry per-call timing/usage, and the trace hierarchy
#     (CHILD_OF references) lets us regroup model-call/tool spans under their
#     run without any run-id attribute (OpenClaw drops raw ids from spans).
#
# Span model (verified against the OpenClaw GitHub source
# extensions/diagnostics-otel/*, default semantic conventions):
#   openclaw.run ............ one agent request; span duration = request latency
#   openclaw.harness.run .... one CLI-backend turn (used if no openclaw.run)
#   openclaw.model.call ..... provider request; tag
#                             openclaw.model_call.time_to_first_byte_ms + usage
#   openclaw.tool.execution . one tool call; span duration = tool exec time;
#                             tag openclaw.toolName / gen_ai.tool.name
#   openclaw.model.usage .... run-level token accounting (gen_ai.usage.*)
#
# KPIs produced (identical to the JSONL benchmark):
#   1. Request latency ......... openclaw.run span duration
#   2. Time to first response .. first model.call tag time_to_first_byte_ms
#   3. Planning duration ....... run start -> first tool.execution start
#   4. Aggregation duration .... last tool.execution end -> run end
#   5. Each agent time ......... request latency grouped by driven agent
#   6. Each tool exec time ..... tool.execution span duration by toolName
#   7. Throughput .............. runs/min (wall clock) + output tokens/sec
#
# Prerequisite OpenClaw config (traces on, pointed at Jaeger's OTLP HTTP port):
#   {
#     "plugins": { "allow": ["diagnostics-otel"],
#                  "entries": { "diagnostics-otel": { "enabled": true } } },
#     "diagnostics": { "enabled": true, "otel": {
#         "enabled": true,
#         "endpoint": "http://localhost:4318",   // Jaeger OTLP/HTTP ingest
#         "protocol": "http/protobuf",           // required; grpc disables export
#         "traces": true,                        // <- traces ON for this workflow
#         "metrics": false,
#         "logs": false,
#         "serviceName": "openclaw"
#     } },
#     "logging": { "level": "debug" }            // debug -> ttfb/usage on spans
#   }
#   Run Jaeger all-in-one (OTLP enabled) e.g.:
#     docker run --rm -p16686:16686 -p4318:4318 \
#       -e COLLECTOR_OTLP_ENABLED=true jaegertracing/all-in-one:latest
#   Restart the OpenClaw gateway after patching config so spans are exported.
#
# Usage:
#   python3 openclaw_tui_jaeger_benchmark.py <prompts.txt> [options]
#
# The script reads prompts (one per line; blank/'#' lines ignored; an optional
# `agentId :: prompt` prefix targets a specific agent), drives them through the
# TUI, waits for spans to flush, queries Jaeger for traces in the benchmark
# window, and prints the KPI report (add --json for machine-readable output).
#
# Examples:
#   python3 openclaw_tui_jaeger_benchmark.py prompts.txt
#   python3 openclaw_tui_jaeger_benchmark.py prompts.txt --jaeger-url http://localhost:16686
#   python3 openclaw_tui_jaeger_benchmark.py prompts.txt --service openclaw --json > report.json

import argparse
import fcntl
import json
import math
import os
import pty
import select
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from statistics import mean


# --------------------------------------------------------------------------- #
# Prompt loading (shared convention with the JSONL benchmark)
# --------------------------------------------------------------------------- #
def read_prompts(path):
    """Return a list of (agent_id_or_None, prompt).

    One prompt per line; blank lines and '#' comments are skipped. A line may
    target a specific agent by prefixing it with ``agentId :: ``.
    """
    prompts = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                agent = None
                if "::" in line:
                    head, _, tail = line.partition("::")
                    head, tail = head.strip(), tail.strip()
                    if head and tail and " " not in head:
                        agent, line = head, tail
                prompts.append((agent, line))
    except OSError as exc:
        print(f"error: cannot read prompts file {path}: {exc}", file=sys.stderr)
    return prompts


# --------------------------------------------------------------------------- #
# Interactive TUI driver (pseudo-terminal)
# --------------------------------------------------------------------------- #
def _wait_idle(fd, settle, max_wait, sink=None):
    """Drain the pty until output is idle for `settle`s or `max_wait`s elapse."""
    deadline = time.monotonic() + max_wait
    last_data = time.monotonic()
    collected = bytearray()
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        try:
            readable, _, _ = select.select([fd], [], [], 0.1)
        except (OSError, ValueError):
            break
        if readable:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break  # EOF: child closed the pty
            collected += chunk
            if sink is not None:
                sink(chunk)
            last_data = time.monotonic()
        elif time.monotonic() - last_data >= settle:
            break
    return bytes(collected)


def drive_tui(prompts, command, settle=3.0, ready_timeout=45.0,
              per_prompt_timeout=600.0, submit=b"\r", exit_keys=b"/exit\r",
              echo=True):
    """Drive prompts through the OpenClaw TUI over a pty.

    Returns ``(start_ms, end_ms, windows)`` where ``windows`` is a list of
    ``(agent_id_or_None, prompt_start_ms, prompt_end_ms)`` used later to
    attribute each Jaeger trace to the agent that produced it.
    """
    sink = None
    if echo:
        def sink(chunk):
            try:
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
            except (OSError, ValueError):
                pass

    windows = []
    pid, fd = pty.fork()
    if pid == 0:
        # Child: replace with the TUI. stdin/stdout/stderr are the pty slave.
        try:
            os.execvp(command[0], command)
        except OSError:
            os._exit(127)

    # Parent: make the master non-blocking so os.read never hangs.
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    start_ms = time.time() * 1000.0
    total = len(prompts)
    # Let the TUI finish booting (model banner, prompt, etc.).
    _wait_idle(fd, settle, ready_timeout, sink)

    for idx, (agent, prompt) in enumerate(prompts, 1):
        label = f"[{agent}] " if agent else ""
        print(f"\n[{idx}/{total}] -> TUI {label}: {prompt}", file=sys.stderr)
        w_start = time.time() * 1000.0
        try:
            os.write(fd, prompt.encode("utf-8", "replace") + submit)
        except OSError:
            print("  warning: TUI pty closed early", file=sys.stderr)
            break
        _wait_idle(fd, settle, per_prompt_timeout, sink)
        windows.append((agent, w_start, time.time() * 1000.0))

    # Ask the TUI to exit, then reap the child.
    try:
        os.write(fd, exit_keys)
        _wait_idle(fd, settle, 10.0, sink)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass

    end_ms = time.time() * 1000.0
    return start_ms, end_ms, windows


# --------------------------------------------------------------------------- #
# Jaeger query client (stdlib urllib)
# --------------------------------------------------------------------------- #
def fetch_traces(jaeger_url, service, start_ms, end_ms, limit=1500):
    """Fetch traces for `service` within [start_ms, end_ms] from Jaeger."""
    query = urllib.parse.urlencode({
        "service": service,
        "start": int(start_ms * 1000.0),   # Jaeger expects microseconds
        "end": int(end_ms * 1000.0),
        "limit": limit,
    })
    url = f"{jaeger_url.rstrip('/')}/api/traces?{query}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    return payload.get("data") or []


def fetch_traces_with_retry(jaeger_url, service, start_ms, end_ms, limit,
                            retries, delay):
    """Poll Jaeger until traces appear (spans flush asynchronously)."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            traces = fetch_traces(jaeger_url, service, start_ms, end_ms, limit)
        except urllib.error.URLError as exc:
            last_error = exc
            traces = []
        if traces:
            return traces, None
        if attempt < retries:
            print(f"  waiting for spans in Jaeger (attempt {attempt}/{retries})...",
                  file=sys.stderr)
            time.sleep(delay)
    return [], last_error


# --------------------------------------------------------------------------- #
# Span helpers
# --------------------------------------------------------------------------- #
RUN_OP = "openclaw.run"
HARNESS_OP = "openclaw.harness.run"
MODEL_CALL_OP = "openclaw.model.call"
TOOL_OP = "openclaw.tool.execution"
USAGE_OP = "openclaw.model.usage"

GENAI_MODEL_OPS = {"chat", "text_completion", "generate_content", "invoke_agent"}


def tags_of(span):
    """Flatten a Jaeger span's tag list into a {key: value} dict."""
    out = {}
    for tag in span.get("tags") or []:
        key = tag.get("key")
        if key is not None:
            out[key] = tag.get("value")
    return out


def num_tag(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def is_model_call(op, tags):
    if op == MODEL_CALL_OP:
        return True
    # gen_ai latest-semconv opt-in renames the span to "<operation> <model>".
    if tags.get("gen_ai.operation.name") in GENAI_MODEL_OPS:
        return True
    return "openclaw.model_call.observation_unit" in tags


def is_tool(op, tags):
    if op == TOOL_OP:
        return True
    if tags.get("gen_ai.operation.name") == "execute_tool":
        return True
    return "openclaw.toolName" in tags


def start_ms_of(span):
    st = span.get("startTime")
    return st / 1000.0 if isinstance(st, (int, float)) else None


def duration_ms_of(span):
    dur = span.get("duration")
    return dur / 1000.0 if isinstance(dur, (int, float)) else None


# --------------------------------------------------------------------------- #
# Per-run aggregation from a trace
# --------------------------------------------------------------------------- #
class Run:
    __slots__ = ("agent_id", "provider", "model", "outcome",
                 "start_ts", "end_ts", "latency_ms", "first_ttfb",
                 "model_calls", "tools", "tokens_in", "tokens_out", "tokens_total")

    def __init__(self):
        self.agent_id = None
        self.provider = None
        self.model = None
        self.outcome = None
        self.start_ts = None
        self.end_ts = None
        self.latency_ms = None
        self.first_ttfb = None
        self.model_calls = []   # {start, dur, ttfb}
        self.tools = []         # {name, start, end, dur, error}
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_total = 0


def _add_usage(run, tags):
    out = (num_tag(tags.get("gen_ai.usage.output_tokens"))
           or num_tag(tags.get("openclaw.model_call.usage.output_tokens")))
    inp = (num_tag(tags.get("gen_ai.usage.input_tokens"))
           or num_tag(tags.get("openclaw.model_call.usage.input_tokens")))
    total = num_tag(tags.get("openclaw.model_call.usage.total_tokens"))
    if inp is not None:
        run.tokens_in += inp
    if out is not None:
        run.tokens_out += out
    if total is not None:
        run.tokens_total += total
    elif inp is not None and out is not None:
        run.tokens_total += inp + out


def agent_for(start_ts, windows):
    """Map a run start time to the agent of the prompt whose window contains it."""
    if start_ts is None:
        return None
    for agent, w_start, w_end in windows:
        if w_start <= start_ts <= w_end:
            return agent
    return None


def build_runs_from_traces(traces, windows):
    runs = []
    for trace in traces:
        spans = trace.get("spans") or []
        if not spans:
            continue
        by_id = {s.get("spanID"): s for s in spans}
        children = defaultdict(list)
        for s in spans:
            for ref in s.get("references") or []:
                if ref.get("refType") == "CHILD_OF" and ref.get("spanID") in by_id:
                    children[ref["spanID"]].append(s.get("spanID"))

        run_span_ids = [s.get("spanID") for s in spans
                        if s.get("operationName") == RUN_OP]
        if not run_span_ids:
            run_span_ids = [s.get("spanID") for s in spans
                            if s.get("operationName") == HARNESS_OP]
        if not run_span_ids:
            continue

        for run_id in run_span_ids:
            run_span = by_id.get(run_id)
            if run_span is None:
                continue
            run = Run()
            rtags = tags_of(run_span)
            run.provider = rtags.get("openclaw.provider")
            run.model = rtags.get("openclaw.model") or rtags.get("gen_ai.request.model")
            run.outcome = rtags.get("openclaw.outcome")
            run.start_ts = start_ms_of(run_span)
            dur = duration_ms_of(run_span)
            run.latency_ms = dur
            if run.start_ts is not None and dur is not None:
                run.end_ts = run.start_ts + dur
            run.agent_id = (agent_for(run.start_ts, windows)
                            or rtags.get("openclaw.agent"))

            # Walk descendants (model calls + tools nested under the run).
            stack = list(children.get(run_id, []))
            seen = set()
            while stack:
                sid = stack.pop()
                if sid in seen:
                    continue
                seen.add(sid)
                stack.extend(children.get(sid, []))
                span = by_id.get(sid)
                if span is None:
                    continue
                op = span.get("operationName")
                stags = tags_of(span)
                if is_model_call(op, stags):
                    ttfb = num_tag(stags.get("openclaw.model_call.time_to_first_byte_ms"))
                    run.model_calls.append({
                        "start": start_ms_of(span),
                        "dur": duration_ms_of(span),
                        "ttfb": ttfb,
                    })
                    _add_usage(run, stags)
                elif op == USAGE_OP:
                    _add_usage(run, stags)
                elif is_tool(op, stags):
                    s_ts = start_ms_of(span)
                    t_dur = duration_ms_of(span)
                    run.tools.append({
                        "name": stags.get("openclaw.toolName")
                        or stags.get("gen_ai.tool.name") or "unknown",
                        "start": s_ts,
                        "end": (s_ts + t_dur) if (s_ts is not None and t_dur is not None) else None,
                        "dur": t_dur,
                        "error": (str(stags.get("openclaw.outcome")) == "error"
                                  or "openclaw.errorCategory" in stags),
                    })

            # First time-to-first-byte in the run (earliest model call).
            timed = sorted((c for c in run.model_calls if c["start"] is not None),
                           key=lambda c: c["start"])
            for c in timed:
                if c["ttfb"] is not None:
                    run.first_ttfb = c["ttfb"]
                    break
            runs.append(run)
    return runs


# --------------------------------------------------------------------------- #
# KPI computation (mirrors the JSONL benchmark)
# --------------------------------------------------------------------------- #
def compute_kpis(runs):
    request_latency = []
    ttfr = []
    planning = []
    aggregation = []
    per_agent = defaultdict(list)
    per_agent_tools = defaultdict(list)
    per_tool = defaultdict(list)
    per_tool_errors = defaultdict(int)
    model_call_dur = []
    model_tps = []
    tokens_in = tokens_out = tokens_total = 0

    for r in runs:
        latency = r.latency_ms
        if latency is None and r.start_ts is not None and r.end_ts is not None:
            latency = r.end_ts - r.start_ts
        agent = r.agent_id or "default"
        if latency is not None:
            request_latency.append(latency)
            per_agent[agent].append(latency)

        if r.first_ttfb is not None:
            ttfr.append(r.first_ttfb)
        elif r.model_calls and r.start_ts is not None:
            starts = [c["start"] for c in r.model_calls if c["start"] is not None]
            if starts:
                ttfr.append(max(0.0, min(starts) - r.start_ts))

        tool_starts = [t["start"] for t in r.tools if t["start"] is not None]
        tool_ends = [t["end"] for t in r.tools if t["end"] is not None]

        if tool_starts and r.start_ts is not None:
            planning.append(max(0.0, min(tool_starts) - r.start_ts))
        elif not r.tools and r.model_calls:
            durs = [c["dur"] for c in r.model_calls if c["dur"] is not None]
            if durs:
                planning.append(durs[0])

        if tool_ends and r.end_ts is not None:
            aggregation.append(max(0.0, r.end_ts - max(tool_ends)))

        for t in r.tools:
            if t["error"]:
                per_tool_errors[t["name"]] += 1
            if t["dur"] is not None:
                per_tool[t["name"]].append(t["dur"])
                per_agent_tools[agent].append(t["dur"])

        for c in r.model_calls:
            if c["dur"] is not None:
                model_call_dur.append(c["dur"])

        tokens_in += r.tokens_in
        tokens_out += r.tokens_out
        tokens_total += r.tokens_total

        if latency and latency > 0 and r.tokens_out:
            model_tps.append(r.tokens_out / (latency / 1000.0))

    starts = [r.start_ts for r in runs if r.start_ts is not None]
    ends = [r.end_ts for r in runs if r.end_ts is not None]
    wall_secs = None
    if starts and ends:
        wall_secs = max(0.0, (max(ends) - min(starts)) / 1000.0)
    n_runs = len(runs)
    runs_per_min = (n_runs / wall_secs * 60.0) if wall_secs and wall_secs > 0 else None
    out_tps_wall = (tokens_out / wall_secs) if wall_secs and wall_secs > 0 else None

    return {
        "n_runs": n_runs,
        "agents": sorted(per_agent.keys()),
        "tools": sorted(per_tool.keys()),
        "request_latency": request_latency,
        "ttfr": ttfr,
        "planning": planning,
        "aggregation": aggregation,
        "per_agent": per_agent,
        "per_agent_tools": per_agent_tools,
        "per_tool": per_tool,
        "per_tool_errors": per_tool_errors,
        "model_call_dur": model_call_dur,
        "model_tps": model_tps,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "wall_secs": wall_secs,
        "runs_per_min": runs_per_min,
        "out_tps_wall": out_tps_wall,
        "window_start": min(starts) if starts else None,
        "window_end": max(ends) if ends else None,
    }


# --------------------------------------------------------------------------- #
# Stats + formatting helpers
# --------------------------------------------------------------------------- #
def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    if not values:
        return None
    return {
        "count": len(values),
        "avg": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
        "min": min(values),
        "total": sum(values),
    }


def fmt(value, unit=""):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.1f}{unit}"
    return f"{value:,}{unit}"


def stat_line(label, values, unit="ms", width=30):
    s = summarize(values)
    if not s:
        return f"  {label:<{width}} n/a"
    return (
        f"  {label:<{width}} count={s['count']:<4} "
        f"avg={fmt(s['avg'], unit)}  p50={fmt(s['p50'], unit)}  "
        f"p95={fmt(s['p95'], unit)}  max={fmt(s['max'], unit)}"
    )


def iso(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(k):
    line = "=" * 78
    print(line)
    print("OpenClaw TUI + tool benchmark (Jaeger traces)")
    print(line)
    print(f"Runs / requests observed : {k['n_runs']}")
    print(f"Agents observed          : {', '.join(k['agents']) or 'n/a'}")
    print(f"Tools observed           : {', '.join(k['tools']) or 'n/a'}")
    if k["window_start"] is not None:
        print(f"Observed window          : {iso(k['window_start'])} .. {iso(k['window_end'])}")
    print()

    print("KPI 1 — Request latency (openclaw.run span duration)")
    print(stat_line("request_latency", k["request_latency"], "ms"))
    print()
    print("KPI 2 — Time to first response (model.call time_to_first_byte_ms)")
    print(stat_line("time_to_first_response", k["ttfr"], "ms"))
    print()
    print("KPI 3 — Planning duration (run start -> first tool call)")
    print(stat_line("planning_duration", k["planning"], "ms"))
    print()
    print("KPI 4 — Aggregation duration (last tool done -> run end)")
    print(stat_line("aggregation_duration", k["aggregation"], "ms"))
    print()

    print("KPI 5 — Each agent time (request latency grouped by agent)")
    if k["per_agent"]:
        print(f"  {'agent':<24} {'runs':>5} {'avg':>10} {'p50':>10} {'p95':>10} {'max':>10}")
        for agent in sorted(k["per_agent"]):
            s = summarize(k["per_agent"][agent])
            print(f"  {agent:<24} {s['count']:>5} "
                  f"{fmt(s['avg'],'ms'):>10} {fmt(s['p50'],'ms'):>10} "
                  f"{fmt(s['p95'],'ms'):>10} {fmt(s['max'],'ms'):>10}")
            tools = k["per_agent_tools"].get(agent)
            if tools:
                ts = summarize(tools)
                print(f"    └ tool time: calls={ts['count']} avg={fmt(ts['avg'],'ms')} "
                      f"total={fmt(ts['total'],'ms')}")
    else:
        print("  n/a")
    print()

    print("KPI 6 — Each tool execution time (grouped by toolName)")
    if k["per_tool"]:
        print(f"  {'toolName':<24} {'calls':>5} {'err':>4} {'avg':>10} {'p50':>10} {'p95':>10} {'max':>10}")
        for tool in sorted(k["per_tool"]):
            s = summarize(k["per_tool"][tool])
            errs = k["per_tool_errors"].get(tool, 0)
            print(f"  {tool:<24} {s['count']:>5} {errs:>4} "
                  f"{fmt(s['avg'],'ms'):>10} {fmt(s['p50'],'ms'):>10} "
                  f"{fmt(s['p95'],'ms'):>10} {fmt(s['max'],'ms'):>10}")
    else:
        print("  n/a  (no openclaw.tool.execution spans found)")
    print()

    print("KPI 7 — Throughput")
    print(f"  requests_per_min        {fmt(k['runs_per_min'])} req/min "
          f"(over {fmt(k['wall_secs'],'s')} wall clock)")
    print(f"  output_tokens_per_sec   {fmt(k['out_tps_wall'],' tok/s')} (wall clock)")
    print(stat_line("output_tokens_per_sec/run", k["model_tps"], " tok/s"))
    print()

    print("Supporting detail")
    print(stat_line("model_call.duration", k["model_call_dur"], "ms"))
    print(f"  tokens  input={fmt(k['tokens_in'])}  output={fmt(k['tokens_out'])}  "
          f"total={fmt(k['tokens_total'])}")
    print(line)


def json_report(k):
    def s(values):
        return summarize(values)

    payload = {
        "runs_observed": k["n_runs"],
        "agents": k["agents"],
        "tools": k["tools"],
        "window": {"start_ms": k["window_start"], "end_ms": k["window_end"]},
        "source": "jaeger-traces",
        "kpi": {
            "request_latency_ms": s(k["request_latency"]),
            "time_to_first_response_ms": s(k["ttfr"]),
            "planning_duration_ms": s(k["planning"]),
            "aggregation_duration_ms": s(k["aggregation"]),
            "per_agent_latency_ms": {a: s(v) for a, v in k["per_agent"].items()},
            "per_tool_execution_ms": {t: s(v) for t, v in k["per_tool"].items()},
            "per_tool_errors": dict(k["per_tool_errors"]),
            "throughput": {
                "requests_per_min": k["runs_per_min"],
                "output_tokens_per_sec_wall": k["out_tps_wall"],
                "output_tokens_per_sec_per_run": s(k["model_tps"]),
                "wall_seconds": k["wall_secs"],
            },
        },
        "tokens": {
            "input": k["tokens_in"],
            "output": k["tokens_out"],
            "total": k["tokens_total"],
        },
        "model_call_duration_ms": s(k["model_call_dur"]),
    }
    print(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv):
    parser = argparse.ArgumentParser(
        description="Drive the OpenClaw TUI over a pty and benchmark multi-agent "
                    "+ tool KPIs from Jaeger traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompts", help="Text file with one prompt per line "
                                        "(blank/'#' ignored; 'agentId :: ' prefix "
                                        "targets a specific agent).")
    parser.add_argument("--jaeger-url", default="http://localhost:16686",
                        help="Jaeger query base URL (default: http://localhost:16686).")
    parser.add_argument("--service", default="openclaw",
                        help="Jaeger service name (diagnostics.otel.serviceName; "
                             "default: openclaw).")
    parser.add_argument("--tui-command", default="openclaw tui",
                        help="Interactive TUI command to spawn "
                             "(default: 'openclaw chat').")
    parser.add_argument("--settle", type=float, default=3.0,
                        help="Idle seconds that mark a TUI response as complete "
                             "(default: 3).")
    parser.add_argument("--ready-timeout", type=float, default=45.0,
                        help="Max seconds to wait for the TUI to boot (default: 45).")
    parser.add_argument("--prompt-timeout", type=float, default=600.0,
                        help="Max seconds to wait for one prompt's response "
                             "(default: 600).")
    parser.add_argument("--exit-keys", default="/exit\\r",
                        help="Keys sent to close the TUI (default: '/exit' + Enter). "
                             "Use '\\x03' for Ctrl-C, '\\x04' for Ctrl-D.")
    parser.add_argument("--flush-wait", type=float, default=8.0,
                        help="Seconds to wait after the run for spans to reach "
                             "Jaeger before querying (default: 8).")
    parser.add_argument("--query-retries", type=int, default=6,
                        help="How many times to poll Jaeger for traces (default: 6).")
    parser.add_argument("--query-delay", type=float, default=5.0,
                        help="Seconds between Jaeger poll attempts (default: 5).")
    parser.add_argument("--limit", type=int, default=1500,
                        help="Max traces to fetch from Jaeger (default: 1500).")
    parser.add_argument("--no-echo", action="store_true",
                        help="Do not mirror TUI output to stderr.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a machine-readable JSON report instead of text.")
    args = parser.parse_args(argv[1:])

    prompts = read_prompts(args.prompts)
    if not prompts:
        print(f"error: no prompts found in {args.prompts}", file=sys.stderr)
        return 1

    command = args.tui_command.split()
    exit_keys = args.exit_keys.encode("utf-8").decode("unicode_escape").encode("utf-8")

    # 1) Drive the interactive TUI and capture per-prompt windows.
    try:
        start_ms, end_ms, windows = drive_tui(
            prompts, command,
            settle=args.settle,
            ready_timeout=args.ready_timeout,
            per_prompt_timeout=args.prompt_timeout,
            exit_keys=exit_keys,
            echo=not args.no_echo,
        )
    except FileNotFoundError:
        print(f"error: '{command[0]}' not found on PATH — is OpenClaw installed?",
              file=sys.stderr)
        return 3

    # 2) Wait for spans to flush, then query Jaeger over the benchmark window
    #    (padded to tolerate clock skew and the batch span processor delay).
    if args.flush_wait > 0:
        time.sleep(args.flush_wait)
    pad = 5000.0
    traces, err = fetch_traces_with_retry(
        args.jaeger_url, args.service,
        start_ms - pad, time.time() * 1000.0 + pad,
        args.limit, max(1, args.query_retries), args.query_delay,
    )
    if err is not None and not traces:
        print(f"error: cannot reach Jaeger at {args.jaeger_url}: {err}", file=sys.stderr)
        return 4

    # 3) Reconstruct runs from the trace hierarchy and compute KPIs.
    runs = build_runs_from_traces(traces, windows)
    kpis = compute_kpis(runs)

    if args.json:
        json_report(kpis)
    else:
        print_report(kpis)

    if kpis["n_runs"] == 0:
        print(
            "\nNo openclaw.run spans found in Jaeger. Make sure:\n"
            "  * diagnostics.otel.traces is true and protocol is 'http/protobuf',\n"
            "  * diagnostics.otel.endpoint points at Jaeger's OTLP port (:4318),\n"
            "  * logging.level is 'debug' and the gateway was restarted,\n"
            f"  * --service matches diagnostics.otel.serviceName ('{args.service}'),\n"
            "  * the TUI actually answered at least one prompt.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

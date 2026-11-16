# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Locust user for Real-Time Scene Understanding (RTSU) hardware sizing.

A single user starts the profile's RTSP streams through the RTSU REST API, polls
``GET /api/streams`` for alert checks and the alert endpoints for completed deep
analyses, and stops when ``target_deep_analyses`` is reached or ``max_timeout_seconds``
elapses. The quitting listener deletes the streams and writes the reports, also after
Ctrl+C. All measurements are taken on the client side.
"""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

import gevent
import requests
from locust import HttpUser, events, task

from common.config import read_yaml_config
from common.metrics import extract_rtsu_alert_check_samples, rtsu_stop_reason, save_rtsu_reports
from common.video import (
    RTSUApi,
    RTSUResponseLog,
    describe_rtsu_error,
    estimate_rtsu_clock_offset,
    parse_rtsu_timestamp,
    redact_rtsp_url,
    redact_rtsu_data,
    start_rtsu_stream,
    stop_rtsu_streams,
)
from src.real_time_scene_understanding.utilities.config import get_rtsu_settings

_STOP_CHECK_INTERVAL_SECONDS = 0.5
_PROGRESS_INTERVAL_SECONDS = 30
_ALERTS_PAGE_LIMIT = 100
_USER_KILL_TIMEOUT_SECONDS = 5
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    """
    Adds custom command-line arguments for the Locust test.

    Only file paths are passed, so RTSP URLs and prompts never appear in the process arguments.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
    """
    parser.add_argument("--request_count", type=int, default=1, help="Ignored; the run stops at the deep-analysis target or the timeout.")
    parser.add_argument("--config_path", type=str, default="", help="Path to the scene understanding application config.")
    parser.add_argument("--profile_path", type=str, default="", help="Path to the input profiles YAML file.")
    parser.add_argument("--report_dir", type=str, default="reports", help="Directory to save reports.")
    parser.add_argument("--warmup_time", type=int, default=0, help="Warmup duration in seconds (warmup runs before Locust starts).")


class RTSUBenchmarkState:
    """Run state shared by the Locust user, its pollers, and the quitting listener."""

    def __init__(self, settings, report_dir, host, client):
        self.settings = settings
        self.report_dir = report_dir
        self.host = host
        self.run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
        self.started_at = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.join(report_dir, "responses"), exist_ok=True)
        self.response_log = (
            RTSUResponseLog(os.path.join(report_dir, "responses", "api_responses.jsonl"))
            if settings.save_raw_responses else None
        )
        self.api = RTSUApi(client, host, settings.endpoints, self.response_log, locust_session=True)

        self.clock = {"offset_s": 0.0, "rtt_ms": None, "samples": 0}
        self.health = None
        self.runtime_config = None
        self.stream_info = []
        self.attempted_stream_ids = []
        self.created_stream_ids = []
        self.rejected_stream_ids = set()
        self.post_mid_client = {}
        self.initial_anchors = {}
        self.publishing_first_seen = {}
        self.reconnects = {}
        self.trackers = {}
        self.alert_check_samples = []
        self.alerts_first_seen = {}
        self.ignored_alerts = set()
        self.deep_analyses = {}
        self.counters = Counter()
        self.warnings = list(settings.warnings)
        self.errors = []
        self.cleanup = {}

        self.run_start = None
        self.stop_time = None
        self.run_end = None
        self.stop_reason = None
        self.stream_start_status = None
        self.greenlets = []
        self.user_greenlet = None
        self.finalized = False

    @property
    def offset(self):
        return self.clock["offset_s"]

    def completed_by_stream(self):
        """Deep analyses that count toward the target (only ``confirmed=true`` when ``confirmed_only``)."""
        return Counter(
            analysis["stream_id"] for analysis in self.deep_analyses.values()
            if not self.settings.confirmed_only or analysis["detail"].get("confirmed") is True
        )

    def target_progress(self):
        counted = sum(self.completed_by_stream().values())
        label = "confirmed deep analyses" if self.settings.confirmed_only else "deep analyses"
        return f"{label}: {counted}/{self.settings.target_deep_analyses}"

    def warn(self, message):
        self.warnings.append(message)
        print(f"Warning: {message}")

    def report_input(self):
        """Collected data in the form expected by ``save_rtsu_reports``."""
        metadata = {
            "app": "scene_understanding",
            "profile": self.settings.profile_name,
            "run_tag": self.run_tag,
            "host": self.host,
            "started_at": self.started_at,
            "stop_reason": self.stop_reason,
            "stream_start_status": self.stream_start_status,
            "run_start_client": self.run_start,
            "stop_time_client": self.stop_time,
            "run_end_client": self.run_end,
            "clock": self.clock,
            "health": self.health,
            "runtime_config": self.runtime_config,
            "settings": self.settings.to_metadata(),
            "streams": self.stream_info,
            "cleanup": self.cleanup,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        return {
            "metadata": metadata,
            "stop_reason": self.stop_reason,
            "stream_start_status": self.stream_start_status,
            "target_deep_analyses": self.settings.target_deep_analyses,
            "stream_ids": list(self.created_stream_ids),
            "alert_check_samples": list(self.alert_check_samples),
            "deep_analyses": list(self.deep_analyses.values()),
            "initial_anchors": dict(self.initial_anchors),
            "post_mid_client": dict(self.post_mid_client),
            "publishing_first_seen_client": dict(self.publishing_first_seen),
            "clock_offset_s": self.offset,
            "run_start_client": self.run_start,
            "run_end_client": self.run_end,
            "counters": dict(self.counters),
            "reconnects": dict(self.reconnects),
        }


def run_preflight(state):
    """Check health, estimate the clock offset, and record the runtime config."""
    clock, health = estimate_rtsu_clock_offset(state.api, state.settings.clock_sync_samples)
    if health is None:
        state.stop_reason = "preflight_failed"
        state.errors.append("GET /api/health did not return HTTP 200")
        print("Preflight failed: the RTSU health check did not return HTTP 200. No streams were created.")
        return

    state.health = {key: health.body.get(key) for key in ("status", "streams_active", "uptime_seconds")}
    if clock is None:
        state.warn("Could not read the server timestamp from /api/health; assuming a clock offset of 0")
    else:
        state.clock = clock
    active = health.body.get("streams_active")
    if isinstance(active, int) and not isinstance(active, bool) and active > 0:
        state.warn(f"{active} stream(s) already active on RTSU; they compete with the benchmark streams for the models")

    runtime = state.api.call("GET", "runtime_config")
    if runtime.ok and isinstance(runtime.body, dict):
        state.runtime_config = runtime.body
        if runtime.body.get("deepAnalyzerEnabled") is False:
            state.warn("The deep analyzer is disabled on RTSU; no deep analyses will be produced")
    else:
        state.warn("Could not read /api/runtime-config.js; models and devices are not recorded")


def start_streams(state):
    """Create the profile streams with unique ``bench-<run_tag>-<stream_id>`` IDs."""
    state.run_start = time.time()
    for stream in state.settings.streams:
        if state.finalized:
            break
        stream_id = f"bench-{state.run_tag}-{stream['stream_id']}"
        state.attempted_stream_ids.append(stream_id)
        response = start_rtsu_stream(state.api, stream_id, stream)
        info = {
            "stream_id": stream_id,
            "profile_stream_id": stream["stream_id"],
            "url": redact_rtsp_url(stream["url"]),
            "url_env": stream["url_env"],
            "alert_prompt": stream["alert_prompt"],
            "deep_analyzer_prompt": stream["deep_analyzer_prompt"],
            "start_status": response.status,
            "start_error": None,
        }
        if response.ok:
            midpoint = (response.t_send + response.t_recv) / 2
            state.created_stream_ids.append(stream_id)
            state.post_mid_client[stream_id] = midpoint
            state.initial_anchors[stream_id] = midpoint + state.offset
            print(f"Started stream '{stream_id}' ({info['url']})")
        else:
            if response.status:
                state.rejected_stream_ids.add(stream_id)
            info["start_error"] = describe_rtsu_error(response)
            state.errors.append(f"POST /api/streams for '{stream_id}': {info['start_error']}")
            print(f"Failed to start stream '{stream_id}': {info['start_error']}")
        state.stream_info.append(info)

    created, total = len(state.created_stream_ids), len(state.settings.streams)
    if created == total:
        state.stream_start_status = "all_started"
    else:
        state.stream_start_status = "partial_streams" if created else "no_streams_started"


def poll_streams_once(state):
    """Sample alert checks from ``GET /api/streams``."""
    response = state.api.call("GET", "streams")
    if not response.ok or not isinstance(response.body, dict):
        state.counters["streams_poll_errors"] += 1
        return
    created = set(state.created_stream_ids)
    for entry in response.body.get("streams") or []:
        if not isinstance(entry, dict) or entry.get("stream_id") not in created:
            continue
        stream_id = entry["stream_id"]
        if entry.get("publishing") is True:
            state.publishing_first_seen.setdefault(stream_id, response.t_recv)
        reconnects = entry.get("reconnect_count")
        if isinstance(reconnects, int) and not isinstance(reconnects, bool):
            state.reconnects[stream_id] = max(reconnects, state.reconnects.get(stream_id, 0))
        tracker = state.trackers.setdefault(stream_id, {})
        state.alert_check_samples.extend(extract_rtsu_alert_check_samples(tracker, entry, response.t_recv))


def save_deep_analysis_detail(state, stream_id, frame_id, detail):
    """Save the full alert detail (description and metrics) of one deep analysis."""
    safe_frame_id = _UNSAFE_FILENAME_CHARS.sub("_", frame_id)[:64]
    path = os.path.join(state.report_dir, "responses", f"deep_analysis_{stream_id}_{safe_frame_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(redact_rtsu_data(detail), file, indent=4, ensure_ascii=False)
    except OSError as exc:
        state.errors.append(f"Could not save the deep-analysis detail for frame '{safe_frame_id}': {exc.strerror}")


def collect_deep_analysis(state, stream_id, frame_id, first_seen):
    """Fetch the alert detail and trigger-frame registry record of a new deep analysis."""
    detail = state.api.call("GET", "alert_detail", stream_id=stream_id, frame_id=frame_id)
    if not detail.ok or not isinstance(detail.body, dict):
        # Retried on the next poll; the first-seen time is kept.
        state.counters["alert_detail_errors"] += 1
        return

    # The registry evicts old frame records, so a 404 is expected and reported as data quality.
    registry = state.api.call("GET", "registry_frame", expected=(200, 404), frame_id=frame_id)
    created_ts = None
    if registry.ok and isinstance(registry.body, dict):
        created_ts = registry.body.get("created_ts")
        registry_status = "ok"
    else:
        registry_status = "not_found" if registry.status == 404 else "error"

    artifacts = {}
    if state.settings.fetch_artifacts:
        for kind in ("thumbnail", "video"):
            artifact = state.api.call("GET", kind, expected=(200, 404), binary=True,
                                      stream_id=stream_id, frame_id=frame_id)
            artifacts[kind] = {
                "status": artifact.status,
                "bytes": artifact.nbytes,
                "sha256": artifact.body.get("sha256") if isinstance(artifact.body, dict) else None,
                "http_latency_ms": artifact.http_latency_ms,
            }

    state.deep_analyses[(stream_id, frame_id)] = {
        "stream_id": stream_id,
        "frame_id": frame_id,
        "first_seen_client": first_seen,
        "detail": detail.body,
        "created_ts": created_ts,
        "registry_status": registry_status,
        "artifacts": artifacts,
    }
    save_deep_analysis_detail(state, stream_id, frame_id, detail.body)
    print(f"Deep analysis completed (stream '{stream_id}', confirmed={detail.body.get('confirmed')}); "
          f"{state.target_progress()}")


def poll_alerts_once(state):
    """Discover completed deep analyses from each stream's alert list."""
    run_start_server = state.run_start + state.offset
    for stream_id in state.created_stream_ids:
        response = state.api.call("GET", "alerts", params={"limit": _ALERTS_PAGE_LIMIT}, stream_id=stream_id)
        if not response.ok or not isinstance(response.body, dict):
            state.counters["alerts_poll_errors"] += 1
            continue
        for alert in response.body.get("alerts") or []:
            frame_id = alert.get("frame_id") if isinstance(alert, dict) else None
            if not isinstance(frame_id, str) or not frame_id:
                continue
            key = (stream_id, frame_id)
            if key in state.deep_analyses or key in state.ignored_alerts:
                continue
            uploaded = parse_rtsu_timestamp(alert.get("uploaded_at"))
            if uploaded is not None and uploaded < run_start_server:
                state.ignored_alerts.add(key)
                state.counters["ignored_pre_run_alerts"] += 1
                continue
            first_seen = state.alerts_first_seen.setdefault(key, response.t_recv)
            collect_deep_analysis(state, stream_id, frame_id, first_seen)


def _poll_forever(state, poll_once, interval):
    while True:
        started = time.time()
        try:
            poll_once(state)
        except Exception as exc:  # keep sampling; a lost poller would silently end collection
            state.counters["poller_exceptions"] += 1
            if state.counters["poller_exceptions"] == 1:
                state.errors.append(f"{poll_once.__name__} raised {type(exc).__name__}")
        gevent.sleep(max(0.0, interval - (time.time() - started)))


def print_progress(state, now):
    print(f"[{int(now - state.run_start)} s] alert checks: {len(state.alert_check_samples)}, "
          f"deep analyses: {len(state.deep_analyses)}, {state.target_progress()}")


class SceneUnderstandingHwSize(HttpUser):
    """
    Locust user class for Real-Time Scene Understanding hardware sizing.

    The whole benchmark runs as a single task iteration (``-i 1``).
    """

    state = None

    def on_start(self):
        opts = self.environment.parsed_options
        try:
            settings = get_rtsu_settings(opts.profile_path, read_yaml_config(opts.config_path))
        except Exception as exc:  # the parent validated the files already; report and do nothing
            print(f"Invalid scene_understanding configuration: {exc}")
            return

        state = RTSUBenchmarkState(settings, os.path.join(opts.report_dir, "scene_understanding"), self.host, self.client)
        state.user_greenlet = gevent.getcurrent()
        SceneUnderstandingHwSize.state = state
        # Exceptions must not escape on_start: Locust would stop the user before its only task runs.
        try:
            run_preflight(state)
            if state.stop_reason is None:
                start_streams(state)
        except Exception as exc:
            state.errors.append(f"Setup failed: {type(exc).__name__}: {redact_rtsp_url(str(exc))}")
            state.stop_reason = state.stop_reason or "interrupted"

    @task
    def run_benchmark(self):
        """Collect alert checks and deep analyses until the stop rule fires."""
        state = SceneUnderstandingHwSize.state
        if state is None or state.stop_reason is not None:
            return
        if not state.created_stream_ids:
            state.stop_reason = "no_streams_started"
            return

        settings = state.settings
        kind = "confirmed deep analyses" if settings.confirmed_only else "deep analyses"
        print(f"Collecting until {settings.target_deep_analyses} {kind} complete "
              f"or {settings.max_timeout_seconds} s elapse...")
        state.greenlets = [
            gevent.spawn(_poll_forever, state, poll_streams_once, settings.streams_poll_interval_seconds),
            gevent.spawn(_poll_forever, state, poll_alerts_once, settings.alerts_poll_interval_seconds),
        ]
        deadline = state.run_start + settings.max_timeout_seconds
        next_progress = time.time() + _PROGRESS_INTERVAL_SECONDS
        while state.stop_reason is None:
            now = time.time()
            reason = rtsu_stop_reason(
                state.completed_by_stream(), state.created_stream_ids, settings.target_deep_analyses,
                settings.min_deep_analyses_per_stream, now, deadline,
            )
            if reason is not None:
                state.stop_reason, state.stop_time = reason, now
                break
            if now >= next_progress:
                print_progress(state, now)
                next_progress = now + _PROGRESS_INTERVAL_SECONDS
            gevent.sleep(_STOP_CHECK_INTERVAL_SECONDS)

        print(f"Stopping collection: {state.stop_reason}")
        if settings.drain_seconds > 0:
            print(f"Draining for {settings.drain_seconds} s to collect in-flight deep analyses...")
            gevent.sleep(settings.drain_seconds)
        gevent.killall(state.greenlets)
        poll_alerts_once(state)  # final sweep before the streams are deleted
        state.run_end = time.time()


@events.quitting.add_listener
def finalize_rtsu_run(environment, **kwargs):
    """Stop collection, delete the benchmark streams, and write the reports (also after Ctrl+C)."""
    state = SceneUnderstandingHwSize.state
    if state is None or state.finalized:
        return
    state.finalized = True
    gevent.killall(state.greenlets)
    # On Ctrl+C, Locust fires this event before it stops the user, so stop the user first.
    if state.user_greenlet is not None and state.user_greenlet is not gevent.getcurrent():
        state.user_greenlet.kill(block=True, timeout=_USER_KILL_TIMEOUT_SECONDS)
    now = time.time()
    state.stop_reason = state.stop_reason or "interrupted"
    state.stop_time = state.stop_time or now
    state.run_end = state.run_end or now

    try:
        # Streams rejected by RTSU are skipped: their ID may belong to another client.
        to_delete = [sid for sid in state.attempted_stream_ids if sid not in state.rejected_stream_ids]
        if to_delete and state.settings.keep_streams:
            print(f"keep_streams is true; benchmark streams left running: {', '.join(to_delete)}")
        elif to_delete:
            cleanup_api = RTSUApi(requests.Session(), state.host, state.settings.endpoints, state.response_log)
            state.cleanup = stop_rtsu_streams(cleanup_api, to_delete)
            print(f"Deleted benchmark streams: {state.cleanup}")
        save_rtsu_reports(state.report_dir, state.report_input())
    finally:
        if state.response_log is not None:
            state.response_log.close()

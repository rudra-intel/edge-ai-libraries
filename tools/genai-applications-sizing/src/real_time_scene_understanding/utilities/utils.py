# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for Real-Time Scene Understanding profiling.

This module provides the warmup phase and the Locust launcher for
Real-Time Scene Understanding (RTSU) hardware sizing.
"""

import os
import subprocess
import time
from datetime import datetime

import requests

from common.video import RTSUApi, describe_rtsu_error, start_rtsu_stream, stop_rtsu_streams
from src.real_time_scene_understanding.locust_files import scene_understanding
from src.real_time_scene_understanding.utilities.config import get_rtsu_settings

# Time Locust gets after Ctrl+C to delete its streams and write a partial report.
_INTERRUPT_GRACE_SECONDS = 60


def run_rtsu_warmup(warmup_time, ip, profile_path, config):
    """
    Load the RTSU models before measuring.

    RTSU loads the alert VLM when the first stream starts and the deep analyzer when the
    first segment finalizes. This starts the ``rtsu_warmup_profile`` streams, waits
    ``warmup_time`` seconds, deletes them, and then waits ``warmup_settle_seconds`` because
    deleting a stream does not cancel deep-analysis jobs that are already queued.

    Args:
        warmup_time: Duration in seconds to keep the warmup streams running.
        ip: Host IP address where the application is deployed.
        profile_path: Path to the profile YAML file.
        config: Pre-loaded configuration dict.
    """
    settings = get_rtsu_settings(profile_path, config, warmup=True)
    api = RTSUApi(requests.Session(), f"http://{ip}", settings.endpoints)
    health = api.call("GET", "health")
    if not health.ok:
        print(f"Warmup skipped: the RTSU health check failed ({describe_rtsu_error(health)}).")
        return

    run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    created = []
    print(f"Sending warmup streams to the scene understanding API for {warmup_time} seconds...")
    try:
        for index, stream in enumerate(settings.streams, start=1):
            stream_id = f"bench-warmup-{run_tag}-{index}"
            response = start_rtsu_stream(api, stream_id, stream)
            if response.ok:
                created.append(stream_id)
            else:
                print(f"Warmup stream '{stream_id}' failed to start: {describe_rtsu_error(response)}")
        if created:
            time.sleep(warmup_time)
    finally:
        if created:
            stop_rtsu_streams(api, created)

    if not created:
        print("Warmup skipped: no warmup stream started.")
        return
    if settings.warmup_settle_seconds > 0:
        print(f"Waiting {settings.warmup_settle_seconds:g} seconds for queued warmup deep analyses to finish...")
        time.sleep(settings.warmup_settle_seconds)
    print("Warmup completed.")


def run_rtsu_hw_sizing(ip, config_path, profile_path, report_dir, warmup_time, config):
    """
    Run the Locust test for Real-Time Scene Understanding hardware sizing.

    Only file paths are passed to Locust, so RTSP URLs and prompts stay out of the
    process arguments.

    Args:
        ip: Host IP address where the application is deployed.
        config_path: Path to the application config YAML file.
        profile_path: Path to the profile YAML file.
        report_dir: Directory to save the test reports.
        warmup_time: Duration in seconds of the warmup phase.
        config: Pre-loaded configuration dict.
    """
    settings = get_rtsu_settings(profile_path, config)
    app_report_dir = os.path.join(report_dir, "scene_understanding")
    os.makedirs(app_report_dir, exist_ok=True)
    print(f"Hardware sizing started for the '{settings.profile_name}' profile...")

    cmd = [
        "locust",
        "-f", f"{scene_understanding.__file__}",
        "--headless",
        "--users", "1",
        "--spawn-rate", "1",
        "-i", "1",
        "--host", f"http://{ip}",
        f"--config_path={os.path.abspath(config_path)}",
        f"--profile_path={os.path.abspath(profile_path)}",
        f"--report_dir={report_dir}",
        f"--warmup_time={warmup_time}",
        "--only-summary",
        "--loglevel", "CRITICAL",
    ]
    process = subprocess.Popen(cmd)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        # Locust received the same Ctrl+C and is deleting its streams; do not kill it.
        print("Interrupted: waiting for the benchmark to delete its streams and write a partial report...")
        try:
            return_code = process.wait(timeout=_INTERRUPT_GRACE_SECONDS)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
            print("Benchmark process killed; delete any remaining 'bench-*' streams with DELETE /api/streams/<id>.")
            return
    if return_code != 0:
        print(f"Locust exited with code {return_code}; "
              f"see {os.path.join(app_report_dir, 'rtsu_run_metadata.json')} for run status.")

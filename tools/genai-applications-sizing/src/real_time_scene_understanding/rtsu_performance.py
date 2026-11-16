# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Real-Time Scene Understanding Performance Profiling.

Entry point for profiling the Real-Time Scene Understanding application via Locust.
The optional warmup runs in this process before Locust starts.
"""

import sys

import yaml

from src.base import BasePerformanceProfiler
from src.real_time_scene_understanding.utilities.config import get_rtsu_settings, is_rtsu_enabled
from src.real_time_scene_understanding.utilities.utils import run_rtsu_hw_sizing, run_rtsu_warmup


class RTSUProfiler(BasePerformanceProfiler):
    """
    Performance profiler for the Real-Time Scene Understanding application.

    Delegates hardware-sizing execution to `run_rtsu_hw_sizing`,
    which launches a headless single-user Locust test.
    """

    @property
    def app_name(self) -> str:
        return "scene_understanding"

    def get_enabled_apis(self) -> bool:
        """Return True if the scene_understanding API is enabled in config."""
        return is_rtsu_enabled(self.config)

    def run_warmup(self, profile_path, input_file):
        """Start and delete the warmup streams so that both models are loaded."""
        if self.get_enabled_apis():
            run_rtsu_warmup(self.warmup_time, self.ip, profile_path, self.config)

    def run_profiling(self, report_dir: str) -> None:
        """Execute the hardware-sizing test if the API is enabled."""
        if self.get_enabled_apis():
            run_rtsu_hw_sizing(
                self.ip,
                self.input_file,
                self.profile_path,
                report_dir,
                self.warmup_time,
                self.config,
            )


def rtsu_performance(
    users: int,
    request_count: int,
    ip: str,
    input_file: str,
    collect_resource_metrics: bool,
    warmup_time: int,
) -> None:
    """
    Entry point for Real-Time Scene Understanding hardware sizing.

    Validates the configuration, then constructs an RTSUProfiler and runs the full
    profiling workflow. The run stops at ``target_deep_analyses`` or
    ``max_timeout_seconds``, so ``request_count`` is ignored.

    Args:
        users: Number of concurrent users (use 1 for single-user sizing).
        request_count: Ignored for this application.
        ip: Host IP address where the application is deployed.
        input_file: Path to the input YAML configuration file.
        collect_resource_metrics: Whether to collect CPU/GPU/memory metrics.
        warmup_time: Duration in seconds for the warmup phase.
    """
    if request_count != 1:
        print("Note: request_count is ignored for scene_understanding; "
              "the run stops at target_deep_analyses or max_timeout_seconds.")
    profiler = RTSUProfiler(
        users=users,
        request_count=1,
        ip=ip,
        input_file=input_file,
        collect_resource_metrics=collect_resource_metrics,
        warmup_time=warmup_time,
    )
    if not profiler.get_enabled_apis():
        print(f"The scene_understanding API is not enabled in '{input_file}'; nothing to profile.")
        return

    # Fail fast, before any report directory, warmup, or stream is created.
    try:
        settings = get_rtsu_settings(profiler.profile_path, profiler.config)
        if warmup_time > 0:
            get_rtsu_settings(profiler.profile_path, profiler.config, warmup=True)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Invalid scene_understanding configuration: {exc}")
        sys.exit(1)
    for warning in settings.warnings:
        print(f"Warning: {warning}")

    profiler.execute()

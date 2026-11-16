# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Configuration utilities for the Real-Time Scene Understanding application.

This module reads the RTSU application config and input profile, and validates
them before anything is sent to the application.

Functions:
  - is_rtsu_enabled(config)                          : returns whether the RTSU API is enabled
  - get_rtsu_settings(profile_path, config, warmup)  : validated run settings and streams
"""

import math
import os
import re
import string
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from common.config import get_api_config, get_profile_details

RTSU_API_NAME = "scene_understanding"
RTSU_WARMUP_PROFILE = "rtsu_warmup_profile"

# RTSU allows 64 characters; the "bench-<YYYYmmddHHMMSS>-" prefix uses 21 of them.
_STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,43}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENDPOINT_RE = re.compile(r"^\d{1,5}/\S*$")
_ALLOWED_URL_SCHEMES = ("rtsp", "rtsps")
_MAX_URL_LENGTH = 2048
_MAX_PROMPT_LENGTH = 4096
_SECONDS_PER_DEEP_ANALYSIS_HINT = 90

# Endpoint key -> placeholders it must contain.
_ENDPOINT_PLACEHOLDERS = {
    "health": set(),
    "runtime_config": set(),
    "streams": set(),
    "alerts": {"stream_id"},
    "alert_detail": {"stream_id", "frame_id"},
    "registry_frame": {"frame_id"},
    "thumbnail": {"stream_id", "frame_id"},
    "video": {"stream_id", "frame_id"},
}
_ARTIFACT_ENDPOINTS = ("thumbnail", "video")


@dataclass
class RTSUSettings:
    """Validated settings for one RTSU benchmark run."""

    profile_name: str
    endpoints: dict
    streams: list
    target_deep_analyses: int = 5
    confirmed_only: bool = True
    min_deep_analyses_per_stream: int = 0
    max_timeout_seconds: int = 900
    streams_poll_interval_seconds: float = 0.5
    alerts_poll_interval_seconds: float = 2.0
    drain_seconds: float = 0.0
    clock_sync_samples: int = 10
    warmup_settle_seconds: float = 30.0
    fetch_artifacts: bool = False
    save_raw_responses: bool = True
    keep_streams: bool = False
    warnings: list = field(default_factory=list)

    def to_metadata(self):
        """Return the settings without streams and warnings, for the run metadata."""
        data = asdict(self)
        data.pop("streams")
        data.pop("warnings")
        return data


def is_rtsu_enabled(config):
    """
    Return whether the Real-Time Scene Understanding API is enabled.

    Args:
        config: Pre-loaded configuration dict.

    Returns:
        bool: True if ``apis.scene_understanding.enabled`` is true.
    """
    return bool(get_api_config(config, RTSU_API_NAME).get("enabled", False))


def _get_int(details, key, default, minimum, errors, maximum=None):
    value = details.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        limit = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        errors.append(f"'{key}' must be an integer {limit} (got {value!r})")
        return default
    return value


def _get_number(details, key, default, minimum, maximum, errors):
    value = details.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        limit = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        errors.append(f"'{key}' must be a number {limit} (got {value!r})")
        return float(default)
    return float(value)


def _get_bool(details, key, default, errors):
    value = details.get(key, default)
    if not isinstance(value, bool):
        errors.append(f"'{key}' must be true or false (got {value!r})")
        return default
    return value


def _load_endpoints(raw_endpoints, fetch_artifacts, errors):
    if not isinstance(raw_endpoints, dict):
        errors.append("'endpoints' must be a mapping of endpoint names to '<port>/<path>' values")
        return {}

    endpoints = {}
    for key, placeholders in _ENDPOINT_PLACEHOLDERS.items():
        value = raw_endpoints.get(key)
        if value is None:
            if key not in _ARTIFACT_ENDPOINTS or fetch_artifacts:
                errors.append(f"endpoints.{key} is required")
            continue
        if not isinstance(value, str) or not _ENDPOINT_RE.match(value):
            errors.append(f"endpoints.{key} must use the '<port>/<path>' format (got {value!r})")
            continue
        try:
            fields = {name for _, name, _, _ in string.Formatter().parse(value) if name is not None}
        except ValueError:
            errors.append(f"endpoints.{key} has unbalanced braces (got {value!r})")
            continue
        if fields != placeholders:
            expected = ", ".join("{" + name + "}" for name in sorted(placeholders)) or "no placeholders"
            errors.append(f"endpoints.{key} must contain {expected} (got {value!r})")
            continue
        endpoints[key] = value
    return endpoints


def _resolve_stream_url(entry, where, errors):
    url_env = entry.get("url_env")
    if url_env is not None:
        if not isinstance(url_env, str) or not _ENV_NAME_RE.match(url_env):
            errors.append(f"{where}.url_env must be an environment variable name")
            return None
        url = os.environ.get(url_env, "")
        if not url.strip():
            errors.append(f"{where}.url_env: environment variable '{url_env}' is not set or is empty")
            return None
    else:
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            errors.append(f"{where}: set 'url' or 'url_env'")
            return None

    url = url.strip()
    # URLs are not echoed in errors because they may contain credentials.
    if len(url) > _MAX_URL_LENGTH:
        errors.append(f"{where}: URL must be at most {_MAX_URL_LENGTH} characters")
        return None
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        errors.append(f"{where}: URL must use rtsp:// or rtsps:// (got scheme '{scheme or 'none'}')")
        return None
    if "<" in url or ">" in url:
        errors.append(f"{where}: URL still contains a '<...>' placeholder; set the RTSP URL of your test video")
        return None
    return url


def _load_streams(profile_name, profile, errors):
    streams = profile.get("streams")
    if not isinstance(streams, list) or not streams:
        errors.append(f"Profile '{profile_name}' must define a non-empty 'streams' list")
        return []

    result, seen = [], set()
    for index, entry in enumerate(streams, start=1):
        where = f"{profile_name}.streams[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be a mapping")
            continue

        stream_id = entry.get("stream_id")
        if not isinstance(stream_id, str) or not _STREAM_ID_RE.match(stream_id):
            errors.append(f"{where}.stream_id must match ^[A-Za-z0-9_-]{{1,43}}$")
            stream_id = None
        elif stream_id in seen:
            errors.append(f"{where}.stream_id '{stream_id}' is used more than once")
            stream_id = None
        else:
            seen.add(stream_id)

        url = _resolve_stream_url(entry, where, errors)

        prompts = {}
        for key in ("alert_prompt", "deep_analyzer_prompt"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}.{key} is required")
            elif len(value) > _MAX_PROMPT_LENGTH:
                errors.append(f"{where}.{key} must be at most {_MAX_PROMPT_LENGTH} characters (got {len(value)})")
            else:
                prompts[key] = value

        if stream_id and url and len(prompts) == 2:
            result.append({
                "stream_id": stream_id,
                "url": url,
                "url_env": entry.get("url_env"),
                "alert_prompt": prompts["alert_prompt"],
                "deep_analyzer_prompt": prompts["deep_analyzer_prompt"],
            })
    return result


def get_rtsu_settings(profile_path, config, warmup=False):
    """
    Load and validate the RTSU application config and input profile.

    Args:
        profile_path: Path to the profiles YAML file.
        config: Pre-loaded configuration dict.
        warmup: Use the ``rtsu_warmup_profile`` streams instead of ``input_profile``.

    Returns:
        RTSUSettings: Validated settings. Non-fatal findings are listed in ``warnings``.

    Raises:
        ValueError: If any setting, endpoint, or stream is invalid (all problems are listed).
    """
    details = get_api_config(config, RTSU_API_NAME)
    profile_name = RTSU_WARMUP_PROFILE if warmup else details.get("input_profile", "")
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError(f"apis.{RTSU_API_NAME}.input_profile is required")

    profile = get_profile_details(profile_path=profile_path, profile_name=profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Profile '{profile_name}' must be a mapping")

    errors = []
    fetch_artifacts = _get_bool(details, "fetch_artifacts", False, errors)
    settings = RTSUSettings(
        profile_name=profile_name,
        endpoints=_load_endpoints(details.get("endpoints"), fetch_artifacts, errors),
        streams=_load_streams(profile_name, profile, errors),
        target_deep_analyses=_get_int(details, "target_deep_analyses", 5, 1, errors),
        confirmed_only=_get_bool(details, "confirmed_only", True, errors),
        min_deep_analyses_per_stream=_get_int(details, "min_deep_analyses_per_stream", 0, 0, errors),
        max_timeout_seconds=_get_int(details, "max_timeout_seconds", 900, 60, errors),
        streams_poll_interval_seconds=_get_number(details, "streams_poll_interval_seconds", 0.5, 0.1, 2.0, errors),
        alerts_poll_interval_seconds=_get_number(details, "alerts_poll_interval_seconds", 2.0, 0.5, 10.0, errors),
        drain_seconds=_get_number(details, "drain_seconds", 0.0, 0.0, None, errors),
        clock_sync_samples=_get_int(details, "clock_sync_samples", 10, 1, errors, maximum=100),
        warmup_settle_seconds=_get_number(details, "warmup_settle_seconds", 30.0, 0.0, None, errors),
        fetch_artifacts=fetch_artifacts,
        save_raw_responses=_get_bool(details, "save_raw_responses", True, errors),
        keep_streams=_get_bool(details, "keep_streams", False, errors),
    )
    if errors:
        raise ValueError(
            f"Invalid scene_understanding settings for profile '{profile_name}':\n  - " + "\n  - ".join(errors)
        )

    stream_count = len(settings.streams)
    hint = _SECONDS_PER_DEEP_ANALYSIS_HINT * math.ceil(settings.target_deep_analyses / stream_count)
    if settings.max_timeout_seconds < hint:
        settings.warnings.append(
            f"max_timeout_seconds={settings.max_timeout_seconds} may be too short for "
            f"{settings.target_deep_analyses} deep analyses on {stream_count} stream(s); consider at least {hint}"
        )
    if settings.streams_poll_interval_seconds > 1.0:
        settings.warnings.append(
            f"streams_poll_interval_seconds={settings.streams_poll_interval_seconds} is above 1.0; "
            "alert checks may be missed"
        )
    if settings.min_deep_analyses_per_stream * stream_count > settings.target_deep_analyses:
        settings.warnings.append(
            f"min_deep_analyses_per_stream x streams ({settings.min_deep_analyses_per_stream} x {stream_count}) "
            f"exceeds target_deep_analyses ({settings.target_deep_analyses}); the per-stream minimum sets the stop point"
        )
    return settings

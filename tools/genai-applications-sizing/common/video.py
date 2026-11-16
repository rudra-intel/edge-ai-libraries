# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video file processing and analysis utilities.

This module provides functions for uploading, processing, and analyzing
video files including metadata extraction, embedding creation, and summary handling.
It also contains the Real-Time Scene Understanding (RTSU) API helpers.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

import cv2
import requests


def get_video_details(video_file_path):
    """
    Extract comprehensive metadata from a video file.
    
    Args:
        video_file_path: Path to the video file.
        
    Returns:
        dict: Video metadata including file size, duration, and FPS.
    """
    file_size_mb = os.path.getsize(video_file_path) / (1024 * 1024)

    cap = cv2.VideoCapture(video_file_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_file_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()

    return {
        "File_Size (MB)": round(file_size_mb, 2),
        "File_Duration (s)": round(duration, 2),
        "File_videoFPS": round(fps, 2)
    }


def upload_video_file(url, filename, filepath):
    """
    Upload a video file to the specified endpoint and retrieve the video ID.
    
    Args:
        url: The upload API endpoint URL.
        filename: Name of the video file.
        filepath: Path to the video file.
        
    Returns:
        str: Video ID if upload succeeded, None otherwise.
    """
    try:
        if filepath is None or filename is None:
            print("Error: Filepath or filename is None.")
            return None
        if not os.path.isfile(filepath):
            print(f"Error: File not found at {filepath}")
            return None
        
        print(f"Video file to be uploaded: {filename} at {filepath}")

        with open(filepath, 'rb') as video_file:
            files = [('video', (filename, video_file, 'application/octet-stream'))]
            response = requests.post(url, headers={}, files=files)
            response.raise_for_status()
            video_id = response.json().get("videoId")
            if video_id:
                print(f"Video upload complete. Video ID: {video_id}")
                return video_id
            else:
                print("Video upload succeeded but no video ID returned.")
                return None
                
    except Exception as e:
        print(f"Error: Unexpected error during video upload: {e}")
        return None


def embedding_video_file(url, video_id):
    """
    Initiate embedding generation for an uploaded video.
    
    Args:
        url: The embedding API endpoint URL.
        video_id: ID of the uploaded video.
        
    Returns:
        int: HTTP status code if successful, None on error.
    """
    try:
        print("Waiting for video embedding creation to complete...")
        headers = {'Content-Type': 'application/json'}
        endpoint = f"{url}/{video_id}"
        start_time = time.time()
        response = requests.post(endpoint, headers=headers, data={})
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Embedding creation took {elapsed_time} seconds.")
        response.raise_for_status()
        return response.status_code
        
    except requests.exceptions.Timeout:
        print(f"Error: Embedding request timed out for video ID {video_id}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error during embedding creation: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Network error during embedding creation: {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during video embedding creation: {e}")
        return None


def wait_for_video_summary_complete(url):
    """
    Poll the video summary API endpoint until processing is complete.
    
    Args:
        url: The API endpoint URL to poll for status.
        
    Returns:
        tuple: (completion_status, response) where completion_status is True if
               video summary completed successfully, False otherwise.
    """
    chunking_time = 0
    chunking_time_captured = False
    video_summary_complete = False
    response = ""
    must_end = time.time() + 3600 # Set a maximum wait time of 1 hour for video summary completion
    print("Waiting for video summary to complete...")
    
    while time.time() < must_end:
        try:
            response = requests.get(url, timeout=10)
            status_code = response.status_code
            
            if status_code != 200:
                print(f"Error: Received status code {status_code}. Response: {response.text}")
                break
            
            json_response = response.json()
            
            if json_response.get("videoSummaryStatus") == "complete":
                video_summary_complete = True
                break

            if not chunking_time_captured and json_response.get("chunkingStatus") == "complete":
                chunking_time = time.time()
                chunking_time_captured = True
            
            time.sleep(10)

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Exiting...")
            raise SystemExit(130)
        except requests.exceptions.RequestException as e:
            print(f"Connection error, retrying: {e}...")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Unexpected error, retrying: {e}...")
            break
    
    if not video_summary_complete:
        print("Video summarization failed.")
    return video_summary_complete, chunking_time, response


def get_video_summary(report_dir, response, summary_id):
    """
    Process a video summary API response and save formatted summaries to a file.
    
    Args:
        report_dir: Directory to save the summary file.
        response: API response object containing the summary.
        summary_id: ID of the summary for filename.
    """
    filename = os.path.join(report_dir, f"video_response_{summary_id}.txt")
    
    try:
        response_data = response.json()
        overall_summary = response_data.get("summary", "")
        frame_summaries = response_data.get("frameSummaries", [])
        
        with open(filename, "w") as file:
            if overall_summary:
                file.write(overall_summary + "\n\n")
            
            for frame_summary in frame_summaries:
                start_frame = frame_summary.get('startFrame', 'N/A')
                end_frame = frame_summary.get('endFrame', 'N/A')
                summary_text = frame_summary.get('summary', '')
                
                file.write(f"\nFrames: {start_frame} -- {end_frame}\n")
                file.write(f"{summary_text}\n")
        
        print(f"Video summary saved to {filename}")
        
    except ValueError as e:
        print(f"Error: Invalid JSON response format: {e}")
    except KeyError as e:
        print(f"Error: Missing expected key in response: {e}")
    except IOError as e:
        print(f"Error: Failed to write summary to {filename}: {e}")
    except Exception as e:
        print(f"Error: Unexpected error saving video summary: {e}")


def embedding_creation_per_sec(video_details, embedding_time):
    """
    Calculate the embedding creation rate in frames per second.
    
    Args:
        video_details: Dictionary containing video metadata (Duration, FPS).
        embedding_time: Time taken for embedding creation in seconds.
        
    Returns:
        float: Embedding creation rate in frames per second, or 0.0 on error.
    """
    try:
        duration = video_details.get("Duration (s)", 0)
        fps = video_details.get("FPS", 0)
        
        if duration <= 0 or fps <= 0 or embedding_time <= 0:
            return 0.0
        
        total_frames = duration * fps
        extracted_frames = total_frames / 15
        embedding_rate = extracted_frames / embedding_time
        
        return round(embedding_rate, 2)
        
    except (TypeError, ZeroDivisionError) as e:
        print(f"Error calculating embedding creation rate: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error calculating embedding creation rate: {e}")
        return 0.0


def summarization_fps(video_details, summarization_time):
    """
    Calculate the video summarization rate in frames per second.
    
    Args:
        video_details: Dictionary containing video metadata (Duration, FPS).
        summarization_time: Time taken for summarization in seconds.
        
    Returns:
        float: Summarization rate in frames per second, or 0.0 on error.
    """
    try:
        duration = video_details.get("Duration (s)", 0)
        fps = video_details.get("FPS", 0)
        
        if duration <= 0 or fps <= 0 or summarization_time <= 0:
            return 0.0
        
        total_frames = duration * fps
        frames_per_second = total_frames / summarization_time
        
        return round(frames_per_second, 2)
        
    except (TypeError, ZeroDivisionError) as e:
        print(f"Error calculating summarization frames per second: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error calculating summarization fps: {e}")
        return 0.0


def convert_timestamp_to_float(timestamp):
    """
    Convert an ISO 8601 formatted timestamp string to a Unix epoch float.
    
    Args:
        timestamp: ISO 8601 timestamp string (e.g., '2024-01-01T12:00:00.000000Z').
        
    Returns:
        float: Unix epoch timestamp.
        
    Raises:
        TypeError: If timestamp is not a string.
        ValueError: If timestamp format is invalid.
    """
    try:
        if not isinstance(timestamp, str):
            raise TypeError(f"Timestamp must be a string, not {type(timestamp).__name__}")
        
        if not timestamp:
            raise ValueError("Timestamp string cannot be empty")
        
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        dt_utc = dt.replace(tzinfo=timezone.utc)
        timestamp_float = dt_utc.timestamp()
        
        return timestamp_float
        
    except TypeError as e:
        print(f"Error: Invalid timestamp type - {e}")
        raise
    except ValueError as e:
        print(f"Error: Invalid timestamp format - {e}. Expected format: 'YYYY-MM-DDTHH:MM:SS.ffffffZ'")
        raise
    except Exception as e:
        print(f"Error: Unexpected error converting timestamp to float: {e}")
        raise

def wait_for_search_to_complete(search_url, query_id):
    querySearchStatus = None 
    search_time = 0 
    each_response = {}
    while querySearchStatus != "idle":
        get_response = requests.get(search_url)

        if get_response.status_code != 200:
            print(f"Error: Failed to fetch search status. HTTP status code: {get_response.status_code}")
            break
        
        for each in get_response.json():
            if each.get("queryId") == query_id:
                querySearchStatus = each.get("queryStatus")
                each_search_time = convert_timestamp_to_float(each.get("updatedAt", 0)) - convert_timestamp_to_float(each.get("createdAt", 0))
                each_response = each
            time.sleep(1)
    search_time = round(each_search_time, 4)
    return search_time, each_response

def get_live_caption_metadata(url, duration_seconds=120):
    """
    Collect metadata from live caption stream for the specified duration.
    
    Args:
        url: The streaming API endpoint URL.
        duration_seconds: Duration to collect data in seconds (default: 120).
        
    Returns:
        list: List of collected data entries.
    """
    collected_data = []
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    print(f"Collecting video caption data for {duration_seconds} seconds...")
    
    with requests.get(url, stream=True) as response:
        try:
            for line in response.iter_lines(decode_unicode=True):
                if time.time() >= end_time:
                    print(f"Collection complete. Collected {len(collected_data)} data entries.")
                    break
                if line and "data" in line:
                    collected_data.append(line)
        except KeyboardInterrupt:
            print("\nStopped streaming")
    
    return collected_data


def stop_all_run_request(run_url, run_ids):
    """
    Stop all live caption runs by sending DELETE requests.
    
    Args:
        run_url: Base URL for the runs API.
        run_ids: List of run IDs to stop.
    """
    for run_id in run_ids:
        response = requests.request("DELETE", f"{run_url}/{run_id}")
        print("Stopped live caption:", response.json())


# ==============================================================================
# Real-Time Scene Understanding (RTSU) API helpers
# ==============================================================================

# Greedy up to the last "@" of the authority, so passwords containing "@" are fully hidden.
_RTSP_CREDENTIALS_RE = re.compile(r"(?i)(rtsps?://)[^\s/?#]*@")
_RUNTIME_CONFIG_PREFIX = "window.RUNTIME_CONFIG ="
_MAX_LOGGED_TEXT_LENGTH = 4096


def redact_rtsp_url(text):
    """Replace RTSP credentials (``user:pass@``) with ``***@`` in a string."""
    if not isinstance(text, str):
        return text
    return _RTSP_CREDENTIALS_RE.sub(r"\1***@", text)


def redact_rtsu_data(value):
    """Return a copy of a JSON-like value with RTSP credentials redacted in every string."""
    if isinstance(value, str):
        return redact_rtsp_url(value)
    if isinstance(value, dict):
        return {key: redact_rtsu_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_rtsu_data(item) for item in value]
    return value


def parse_rtsu_timestamp(value):
    """
    Convert an RTSU timestamp to Unix epoch seconds.

    Args:
        value: ISO 8601 string (for example ``uploaded_at``) or epoch number.

    Returns:
        float: Epoch seconds, or None if the value cannot be parsed.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def parse_rtsu_runtime_config(text):
    """Parse the ``window.RUNTIME_CONFIG = {...};`` body of ``/api/runtime-config.js``."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if not body.startswith(_RUNTIME_CONFIG_PREFIX):
        return None
    body = body[len(_RUNTIME_CONFIG_PREFIX):].strip().rstrip(";").strip()
    try:
        config = json.loads(body)
    except ValueError:
        return None
    return config if isinstance(config, dict) else None


def compute_rtsu_clock_offset(measurements):
    """
    Estimate the server-minus-client clock offset (NTP style).

    Args:
        measurements: List of ``(t_send, t_recv, server_ts)`` tuples, client times in epoch seconds.

    Returns:
        dict: ``offset_s``, ``rtt_ms`` and ``samples`` from the shortest round trip, or None.
    """
    if not measurements:
        return None
    t_send, t_recv, server_ts = min(measurements, key=lambda item: item[1] - item[0])
    return {
        "offset_s": round(server_ts - (t_send + t_recv) / 2, 6),
        "rtt_ms": round((t_recv - t_send) * 1000, 3),
        "samples": len(measurements),
    }


@dataclass
class RTSUResponse:
    """Outcome of one RTSU API call. Times are client epoch seconds; status 0 means no response."""

    status: int
    body: object
    t_send: float
    t_recv: float
    nbytes: int = 0
    error: str = ""

    @property
    def ok(self):
        return 200 <= self.status < 300

    @property
    def http_latency_ms(self):
        return round((self.t_recv - self.t_send) * 1000, 2)


class RTSUResponseLog:
    """Append-only JSON Lines log of RTSU API calls with credentials redacted."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8")

    def write(self, record):
        if self._file is None:
            return
        self._file.write(json.dumps(redact_rtsu_data(record), ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _decode_rtsu_body(content):
    """Decode a JSON (or runtime-config JavaScript) response body."""
    try:
        return json.loads(content)
    except ValueError:
        pass
    text = content.decode("utf-8", errors="replace")
    runtime_config = parse_rtsu_runtime_config(text)
    if runtime_config is not None:
        return runtime_config
    return text[:_MAX_LOGGED_TEXT_LENGTH]


class RTSUApi:
    """
    Calls RTSU REST endpoints and records every response.

    Works with a Locust ``HttpSession`` (``locust_session=True``) so calls appear in the
    Locust statistics, or with a plain ``requests.Session`` for warmup and cleanup.
    Endpoints use the tool's ``'<port>/<path>'`` format with ``{stream_id}`` and
    ``{frame_id}`` placeholders.
    """

    def __init__(self, session, host, endpoints, response_log=None, timeout=10.0, locust_session=False):
        self.session = session
        self.host = host.rstrip("/")
        self.endpoints = endpoints
        self.response_log = response_log
        self.timeout = timeout
        self.locust_session = locust_session

    def url(self, endpoint_key, path_suffix="", **path_params):
        quoted = {key: quote(str(value), safe="") for key, value in path_params.items()}
        return f"{self.host}:{self.endpoints[endpoint_key].format(**quoted)}{path_suffix}"

    def label(self, method, endpoint_key, path_suffix=""):
        """Stable request name for Locust statistics, for example ``GET /api/streams/[id]/alerts``."""
        path = self.endpoints[endpoint_key].split("/", 1)[-1]
        path = path.replace("{stream_id}", "[id]").replace("{frame_id}", "[frame_id]")
        return f"{method} /{path}{'/[id]' if path_suffix else ''}"

    def call(self, method, endpoint_key, *, expected=(200,), params=None, json_body=None,
             binary=False, path_suffix="", **path_params):
        """
        Send one request.

        Args:
            method: HTTP method.
            endpoint_key: Key in the ``endpoints`` config section.
            expected: Status codes counted as success in the Locust statistics.
            params: Optional query parameters.
            json_body: Optional JSON request body.
            binary: Log only the size and SHA-256 hash of the body.
            path_suffix: Already-quoted suffix appended to the endpoint path.
            **path_params: Values for the endpoint placeholders.

        Returns:
            RTSUResponse: Status, decoded body, and client send/receive times.
        """
        url = self.url(endpoint_key, path_suffix, **path_params)
        name = self.label(method, endpoint_key, path_suffix)
        status, content, error = 0, b"", ""
        t_send = time.time()
        try:
            if self.locust_session:
                with self.session.request(method, url, name=name, catch_response=True, timeout=self.timeout,
                                          params=params, json=json_body) as response:
                    status, content = response.status_code or 0, response.content or b""
                    if status == 0 and getattr(response, "error", None) is not None:
                        error = type(response.error).__name__
                    if status in expected:
                        response.success()
                    else:
                        response.failure(f"Unexpected HTTP status {status}")
            else:
                response = self.session.request(method, url, timeout=self.timeout, params=params, json=json_body)
                status, content = response.status_code, response.content or b""
        except requests.RequestException as exc:
            error = type(exc).__name__
        t_recv = time.time()

        body = None
        if content:
            if binary:
                body = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            else:
                body = _decode_rtsu_body(content)
        result = RTSUResponse(status=status, body=body, t_send=t_send, t_recv=t_recv,
                              nbytes=len(content), error=error)

        if self.response_log is not None:
            self.response_log.write({
                "t_send": round(t_send, 6),
                "t_recv": round(t_recv, 6),
                "method": method,
                "name": name,
                "path": urlsplit(url).path,
                "params": params,
                "request_body": json_body,
                "status": status,
                "http_latency_ms": result.http_latency_ms,
                "bytes": result.nbytes,
                "body": body,
                "error": error,
            })
        return result


def describe_rtsu_error(response):
    """Return a short, credential-free description of a failed RTSU call."""
    if response.status == 0:
        return f"request failed ({response.error or 'no response'})"
    detail = response.body.get("detail") if isinstance(response.body, dict) else None
    if isinstance(detail, str):
        return f"HTTP {response.status}: {redact_rtsp_url(detail)[:300]}"
    return f"HTTP {response.status}"


def estimate_rtsu_clock_offset(api, samples):
    """
    Call ``GET /api/health`` several times and estimate the clock offset.

    Args:
        api: RTSUApi instance.
        samples: Number of health calls.

    Returns:
        tuple: (offset dict or None, last successful RTSUResponse or None)
    """
    measurements, last_ok = [], None
    for _ in range(samples):
        response = api.call("GET", "health")
        if not response.ok or not isinstance(response.body, dict):
            continue
        last_ok = response
        server_ts = parse_rtsu_timestamp(response.body.get("timestamp"))
        if server_ts is not None:
            measurements.append((response.t_send, response.t_recv, server_ts))
    return compute_rtsu_clock_offset(measurements), last_ok


def start_rtsu_stream(api, stream_id, stream):
    """
    Start one stream with ``POST /api/streams``.

    Args:
        api: RTSUApi instance.
        stream_id: Full stream ID to create.
        stream: Validated stream dict with ``url``, ``alert_prompt`` and ``deep_analyzer_prompt``.

    Returns:
        RTSUResponse: Response of the POST call.
    """
    payload = {
        "url": stream["url"],
        "stream_id": stream_id,
        "alert_prompt": stream["alert_prompt"],
        "deep_analyzer_prompt": stream["deep_analyzer_prompt"],
    }
    return api.call("POST", "streams", json_body=payload)


def stop_rtsu_streams(api, stream_ids):
    """
    Delete streams with ``DELETE /api/streams/{id}``. A 404 counts as already removed.

    Returns:
        dict: Stream ID mapped to ``removed``, ``not_found`` or an error description.
    """
    results = {}
    for stream_id in stream_ids:
        response = api.call("DELETE", "streams", expected=(200, 404), path_suffix="/" + quote(stream_id, safe=""))
        if response.status == 200:
            results[stream_id] = "removed"
        elif response.status == 404:
            results[stream_id] = "not_found"
        else:
            results[stream_id] = describe_rtsu_error(response)
    return results

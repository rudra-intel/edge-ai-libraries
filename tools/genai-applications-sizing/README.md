# GenAI Applications Sizing Tool

The GenAI Applications Sizing Tool is a performance profiling utility for benchmarking Intel's Edge AI library applications. This guide will walk you through the initial setup, configuration, and usage of the tool.

## Overview

The GenAI Applications Sizing Tool key features:

- **Mutli application supports** ChatQnA (Modular & Core), Video Summary and Search, live captioning, real-time scene understanding
- **Resource Monitoring** including CPU, GPU, and memory consumption
- **Flexible Configuration** Two layer Yaml based configuration (Profiles and App config)
- **Comprehensive Reporting** reports with logs, JSON/CSV, plots

### Supported Applications

| Application | Description | Config File |
|-------------|-------------|-------------|
| `video_summary_search` | Video summarization and search profiling | `profiles/video-summary-config.yaml` |
| `chatqna` | ChatQnA modular application | `profiles/chatqna-config.yaml` |
| `chatqna_core` | ChatQnA core application | `profiles/chatqna-core-config.yaml` |
| `live_caption` | Live video captioning | `profiles/live-video-caption-config.yaml` |
| `scene_understanding` | Real-Time Scene Understanding | `profiles/real-time-scene-understanding-config.yaml` |


## Prerequisites

### Software Requirements

- Python 3.11 or higher
- Docker (optional, for containerized execution)
- Git (for cloning performance tools)

### Target Application Deployment

Before running the sizing tool, ensure the target application is deployed and accessible. Refer to the deployment guides for each application:

**Video Search and Summarization:**
- [Get Started](../../sample-applications/video-search-and-summarization/docs/user-guide/get-started.md)
- [System Requirements](../../sample-applications/video-search-and-summarization/docs/user-guide/get-started/system-requirements.md)

**Chat Question and Answer Core:**
- [Sample Application README](../../sample-applications/chat-question-and-answer-core/README.md)

**Live Video Captioning:**
- [Sample Application README](https://github.com/open-edge-platform/metro-ai-suite/blob/main/live-video-analysis/live-video-captioning/README.md)

**Real-Time Scene Understanding:**
- [Sample Application README](https://github.com/open-edge-platform/metro-ai-suite/blob/main/live-video-analysis/real-time-scene-understanding/README.md)
- [User Guide](https://github.com/open-edge-platform/metro-ai-suite/tree/main/live-video-analysis/real-time-scene-understanding/docs/user-guide)

## Installation

### Option 1: Local Installation

1. **Clone the repository** (if not already done):

   ```bash
   git clone <repository-url>
   cd edge-ai-libraries/tools/genai-applications-sizing
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation

1. **Build the Docker image**:

   ```bash
   docker build -t genai-sizing-tool .
   ```

2. **Run with Docker** (see [Running with Docker](#running-with-docker) section below).

## Configuration

### Video Summary Configuration

The tool uses YAML configuration files to define API endpoints and input profiles. The video summary configuration is located at `profiles/video-summary-config.yaml`:

```yaml
global:
  report_dir: 'reports'
  input_profiles_path: 'profiles/profiles.yaml'

apis:
  video_summary:
    enabled: true
    service_name: 'video'
    endpoints:
      upload: '12345/manager/videos'
      summary: '12345/manager/summary'
      states: '12345/manager/states'
      telemetry: '9766/v1/telemetry'
    input_profile: 'video_summary_wsf'
```

### Input Profiles

Input profiles define the test data and parameters. Video summary profiles are defined in `profiles/profiles.yaml`:

```yaml
profiles:
  video_summary_wsf:
    input_type: "video"
    input_size: "medium"
    files:
      - name: "one-by-one-person-detection.mp4"
        path: "data/one-by-one-person-detection.mp4"
    payload:
      # Sampling and prompt configuration
      ...
```

### Preparing Test Videos

Place your test video files in the `data/` directory. The profile configuration should reference these files:

```bash
# Copy your test video to the data directory
cp /path/to/your/video.mp4 data/
```

Update the profile in `profiles/profiles.yaml` to reference your video file.

## Running the Tool

### Basic Usage

Run the Video Summary sizing profile:

```bash
python profile-runner.py \
  --app=video_summary_search \
  --input=profiles/video-summary-config.yaml \
  --host_ip=<IP_ADDRESS_OF_DEPLOYED_APP> \
  --request_count=1 \
  --collect_resource_metrics=yes
```

### Command-Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--app` | Application to profile (`video_summary_search`, `chatqna`, `chatqna_core`, `live_caption`, `scene_understanding`) | - | Yes |
| `--input` | Path to configuration YAML file | `config.yaml` | No |
| `--host_ip` | IP address of the deployed application | - | Yes |
| `--request_count` | Total number of requests to execute (ignored for `live_caption` and `scene_understanding`) | `1` | No |
| `--users` | Number of concurrent users (currently set to 1) | `1` | No |
| `--spawn_rate` | Rate at which users are spawned per second | `1` | No |
| `--warmup_time` | Duration in seconds for warmup requests | `0` | No |
| `--collect_resource_metrics` | Enable resource metrics collection (`yes`/`no`). When set to `yes`, the tool and target application must be running on the same machine. | `no` | No |

### Example: Video Summary with Warmup

For more accurate results, use a warmup period to prime the system:

```bash
python profile-runner.py \
  --app=video_summary_search \
  --input=profiles/video-summary-config.yaml \
  --host_ip=192.168.1.100 \
  --request_count=10 \
  --warmup_time=60 \
  --collect_resource_metrics=yes
```

### Running with Docker

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  genai-sizing-tool \
  --app=video_summary_search \
  --input=profiles/video-summary-config.yaml \
  --host_ip=192.168.1.100 \
  --request_count=1 \
  --collect_resource_metrics=yes
```

## Understanding the Output

### Report Directory Structure

Reports are saved in the `reports/` directory with timestamped folders:

```
reports/
└── video_summary_search_20260320_134736/
    ├── perf_tool_logs/           # Metrics Manager logs
    └── video_summary/
        ├── video_summary_metrics_wsf.csv              # metrics for wsf
        ├── video_summary_search_metrics.json          # metrics summary
        ├── video_summary_search_telemetry_details.json # Detailed Telemetry data
        └── video_response_*.txt                       # Video response details
```

### Key Metrics

The tool captures the following metrics:

| Metric | Description |
|--------|-------------|
| **Response Time** | End-to-end latency for video summarization |
| **Throughput** | Requests processed per second |
| **Upload Time** | Time to upload video to the service |
| **Processing Time** | Time for video analysis and summarization |
| **Resource Utilization** | CPU, GPU, and memory usage (when enabled) |

### Sample Output

```
Hardware sizing started for the 'video_summary_wsf' profile...
Sending warmup requests to video summary API...
Completed warmup requests.!

[Locust output with request statistics]

Report saved to: reports/video_summary_search_20260320_143044/
```

## Other Application References

### ChatQnA Core

The ChatQnA Core application is a streamlined version optimized for core question-answering functionality.

#### Configuration

Configuration file: `profiles/chatqna-core-config.yaml`

```yaml
apis:
  stream_log:
    enabled: true
    service_name: 'chatqna'
    endpoints:
      chat: '8102/v1/chatqna/chat'
      document: '8102/v1/chatqna/documents'
    input_profile: 'chatqna_wsf'
```

#### Example Command

```bash
python profile-runner.py \
  --app=chatqna_core \
  --input=profiles/chatqna-core-config.yaml \
  --host_ip=<IP_ADDRESS> \
  --request_count=10 \
  --collect_resource_metrics=yes
```

#### Deployment Reference

- [Chat Question and Answer Core Sample Application](../../sample-applications/chat-question-and-answer-core/README.md)

---

### Live Video Caption

The Live Video Caption application provides real-time captioning for video streams using AI models.

#### Configuration

Configuration file: `profiles/live-video-caption-config.yaml`

```yaml
apis:
  live_caption:
    enabled: true
    service_name: 'live_caption'
    endpoints:
      runs: '4173/api/runs'
      metadata: '4173/api/runs/metadata-stream'
    captioning_time: 180  # Time in seconds for captioning duration
    input_profile: 'live_video_caption_wsf'
```

#### Example Command

```bash
python profile-runner.py \
  --app=live_caption \
  --input=profiles/live-video-caption-config.yaml \
  --host_ip=<IP_ADDRESS> \
  --warmup_time=30 \
  --collect_resource_metrics=yes
```

> [!NOTE]
> The `--request_count` parameter is not applicable for live captioning as it operates on continuous streams.

#### Key Configuration Options

| Option | Description |
|--------|-------------|
| `captioning_time` | Duration in seconds to run the captioning session |
| `runs` endpoint | API endpoint for starting caption runs |
| `metadata` endpoint | Streaming endpoint for caption metadata |

---

### Real-Time Scene Understanding

Real-Time Scene Understanding (RTSU) checks RTSP streams with an alert VLM and runs a multi-frame deep analysis when an alert check answers Yes. The tool drives only the RTSU REST API and measures everything on the client side: it starts the profile's streams, samples alert checks from `GET /api/streams`, collects each completed deep analysis from the alert and frame-registry endpoints, and deletes its streams at the end.

#### Configuration

Configuration file: `profiles/real-time-scene-understanding-config.yaml`

```yaml
apis:
  scene_understanding:
    enabled: true
    service_name: 'scene_understanding'
    endpoints:
      health: '9100/api/health'
      runtime_config: '9100/api/runtime-config.js'
      streams: '9100/api/streams'
      alerts: '9100/api/streams/{stream_id}/alerts'
      alert_detail: '9100/api/streams/{stream_id}/alerts/{frame_id}'
      thumbnail: '9100/api/streams/{stream_id}/alerts/{frame_id}/thumbnail'
      video: '9100/api/streams/{stream_id}/alerts/{frame_id}/video'
      registry_frame: '9100/api/registry/frame/{frame_id}'
    target_deep_analyses: 5
    max_timeout_seconds: 900
    input_profile: 'rtsu_wsf'
```

| Option | Description | Default |
|--------|-------------|---------|
| `target_deep_analyses` | Completed deep analyses to collect, from any mix of streams (integer >= 1) | `5` |
| `confirmed_only` | Count only deep analyses with `confirmed=true` toward `target_deep_analyses` and `min_deep_analyses_per_stream`; unconfirmed ones are still reported | `true` |
| `min_deep_analyses_per_stream` | Optional per-stream minimum (0 = off) | `0` |
| `max_timeout_seconds` | Hard stop (integer >= 60); a partial report is still written | `900` |
| `streams_poll_interval_seconds` | Alert-check sampling interval (0.1 to 2.0) | `0.5` |
| `alerts_poll_interval_seconds` | Deep-analysis discovery interval (0.5 to 10) | `2.0` |
| `drain_seconds` | Extra collection time after the target is met | `0` |
| `clock_sync_samples` | `/api/health` calls used to estimate the client-server clock offset | `10` |
| `warmup_settle_seconds` | Wait after the warmup streams are deleted | `30` |
| `fetch_artifacts` | Also download each alert's thumbnail and video (size, SHA-256, HTTP latency) | `false` |
| `save_raw_responses` | Write every API call to `responses/api_responses.jsonl` | `true` |
| `keep_streams` | Do not delete the benchmark streams at the end (debugging) | `false` |

The configuration and profile are validated before any stream is created.

#### Input Profiles

RTSU profiles in `profiles/profiles.yaml` list the streams to start: `rtsu_wsf` (1 stream), `rtsu_multi_stream` (4 streams), and `rtsu_warmup_profile` (used by `--warmup_time`).

```yaml
  rtsu_wsf:
    input_type: "rtsp stream"
    input_size: "single stream"
    streams:
      - stream_id: "cam1"
        url: "rtsp://<rtsp-server-ip>:8554/<event-video>"
        alert_prompt: "Is there visible fire or smoke in the scene? Answer yes or no."
        deep_analyzer_prompt: "Describe the fire or smoke event chronologically."
```

- `stream_id` must match `^[A-Za-z0-9_-]{1,43}$`. Each run creates `bench-<YYYYmmddHHMMSS>-<stream_id>`, so earlier alerts stored by RTSU are never counted.
- `url` must use `rtsp://` or `rtsps://`. Replace the `<...>` placeholders before running.
- For sources that need credentials, replace `url` with `url_env: <ENV_VAR_NAME>` and export the URL in that variable. Credentials are redacted (`rtsp://***@host`) in every report and console line.
- Both prompts are required and limited to 4096 characters.

#### RTSP Source Requirement

Serve a looping test video that contains the target event from a separate RTSP server; the MediaMTX RTSP port of RTSU is not published to the host. If the alert prompt never produces a Yes verdict, no deep analysis runs and the run ends with `timeout`. Run with no other streams active and the dashboard closed.

#### Example Commands

```bash
python profile-runner.py \
  --app=scene_understanding \
  --input=profiles/real-time-scene-understanding-config.yaml \
  --host_ip=<IP_ADDRESS> \
  --warmup_time=60
```

```bash
docker run --rm --network host \
  -v $(pwd)/profiles:/app/profiles:ro \
  -v $(pwd)/reports:/app/reports \
  genai-sizing-tool \
  --app=scene_understanding \
  --input=profiles/real-time-scene-understanding-config.yaml \
  --host_ip=<IP_ADDRESS> \
  --warmup_time=60
```

Add `-e <ENV_VAR_NAME>` to the Docker command for profiles that use `url_env`.

> [!NOTE]
> The run stops when `target_deep_analyses` deep analyses have completed (only `confirmed=true` ones count when `confirmed_only` is true), or after `max_timeout_seconds`. `--request_count` is ignored. One stream produces at most one deep analysis per 15-second segment, so 5 analyses on 1 stream take at least about 90 seconds. Press Ctrl+C to stop early: the streams are deleted and a partial report is written.

#### Warmup

RTSU loads its models on first use: the alert VLM when the first stream starts and the deep analyzer when the first segment finalizes. With `--warmup_time=<seconds>`, the tool starts the `rtsu_warmup_profile` streams, waits, deletes them, and then waits `warmup_settle_seconds`, because deleting a stream does not cancel deep analyses that are already queued. Use `--warmup_time=60` after an RTSU restart, with a source that produces Yes verdicts.

#### Output

```
reports/scene_understanding_<YYYYmmdd_HHMMSS>/
└── scene_understanding/
    ├── rtsu_run_metadata.json              # stop reason, clock offset, settings, models/devices, streams
    ├── rtsu_alert_check_indv_metrics.csv   # one row per alert check
    ├── rtsu_deep_analysis_indv_metrics.csv # one row per deep analysis
    ├── rtsu_startup_metrics.csv            # per stream: publishing, first caption, first deep analysis
    ├── rtsu_summary_metrics.json           # KPI statistics, rates, data quality
    ├── rtsu_summary_metrics.csv            # Section, KPI, Scope, Min, Max, Mean, p90
    ├── rtsu_metrics_wsf.csv                # key,value (WSF)
    └── responses/
        ├── api_responses.jsonl             # every API call, credentials redacted
        └── deep_analysis_<stream>_<frame_id>.json
```

The stop reason is one of `target_met`, `timeout`, `interrupted`, `preflight_failed`, or `no_streams_started`. Statistics (`min`, `max`, `mean`, `p90`) are reported overall and per stream; null values are excluded.

#### Key Metrics

Alert check (one row per caption):

| Metric | Description |
|--------|-------------|
| `TTFT (ms)`, `TPOT (ms)`, `Throughput (tok/s)` | GenAI values reported by RTSU. Values flagged `Possibly Stale` (identical to the previous caption) are excluded from the statistics |
| `Latency (ms)` | Frame age when its verdict is published: `caption_ts - (stream start + playback_seconds)`, including queue time in the shared alert VLM |
| `Cycle Interval (ms)` | Time between consecutive verdicts of the same stream |

Deep analysis (one row per alert):

| Metric | Description |
|--------|-------------|
| `TTFT (ms)`, `TPOT (ms)`, `Throughput (tok/s)`, `Output Tokens` | GenAI values of the multi-frame generation |
| `Generation Latency (ms)` | Duration of the deep-analysis generation |
| `E2E Latency (ms)` | From the trigger frame to the result being visible through the API |
| `Trigger Latency (ms)` | From the trigger frame to its Yes verdict |
| `Wait (ms)` | From the Yes verdict to the stored result, excluding generation: rest of the 15-second segment, queue time, segment decode, and frame sampling |
| `Publish Lag (ms)` | From the stored result to the result being visible to the client (includes up to one alert-poll interval) |

`E2E Latency = Trigger Latency + Wait + Generation Latency + Publish Lag`. System-level rates (alert checks and deep analyses per minute, Yes and confirmed ratios, aggregate deep decode throughput) and data-quality counters (missed or stale alert checks, unmatched triggers, evicted registry records) are in `rtsu_summary_metrics.json`. Resource metrics are not part of this benchmark; keep `--collect_resource_metrics=no`.

---

## Troubleshooting

### Common Issues

**Connection Refused**
```
Error: Connection refused to http://<ip>:12345
```
- Verify the target application is running and accessible
- Check firewall settings allow traffic on the required ports

**Video Upload Failed**
```
Warmup: Video upload failed, skipping this iteration
```
- Ensure test videos exist in the `data/` directory
- Verify the video file path in `profiles/profiles.yaml`

**Invalid IP Address**
```
Invalid IP address format: <ip>
```
- Use IPv4 format (e.g., `192.168.1.100`)
- Do not include protocol prefixes (http://)
- `localhost` and hostnames are not accepted; use the host IP or `127.0.0.1`

**Scene Understanding: `POST /api/streams` returns 409**
- The stream ID already exists, or RTSU reached its `MAX_STREAMS` limit. Delete leftover streams or use fewer streams in the profile.

**Scene Understanding: run ends with `timeout` and no deep analyses**
- The alert prompt never produced a Yes verdict, or the deep analyzer is disabled. Check the `Decision` column in `rtsu_alert_check_indv_metrics.csv` and use a test video that contains the event.

### Validation

Before running a full profiling session:

1. Verify application connectivity:
   ```bash
   curl http://<host_ip>:12345/manager/videos
   ```

2. Start with a single request:
   ```bash
   python profile-runner.py \
     --app=video_summary_search \
     --input=profiles/video-summary-config.yaml \
     --host_ip=<IP> \
     --request_count=1
   ```

## Supporting Resources

- **Video Search and Summarization**
  - [API Reference](../../sample-applications/video-search-and-summarization/docs/user-guide/api-reference.md)
  - [Get Started Guide](../../sample-applications/video-search-and-summarization/docs/user-guide/get-started.md)
- **Chat Question and Answer Core**
  - [Sample Application](../../sample-applications/chat-question-and-answer-core/README.md)
- **Live Video Captioning**
  - [Sample Application](https://github.com/open-edge-platform/metro-ai-suite/blob/main/live-video-analysis/live-video-captioning/README.md)
- **Real-Time Scene Understanding**
  - [Sample Application](https://github.com/open-edge-platform/metro-ai-suite/blob/main/live-video-analysis/real-time-scene-understanding/README.md)
- **Metrics Manager** [Microservice](../../microservices/metrics-manager/README.md)
- Customize input profiles in `profiles/profiles.yaml` for your use case
- Enable resource metrics collection for detailed hardware analysis

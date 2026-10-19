<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# ChatQnA-Core Application Benchmarking

## Overview

The benchmarking tool is a standalone, sequential Python HTTP client for an
**already deployed** Chat Question and Answer Core application. It sends a
repeatable document-and-question workload to the application's REST API and
records client-observed latency, request outcomes, and throughput. It can also
collect optional system resource metrics during the measured phase.

Use it to compare end-to-end application behavior across deployments, for
example CPU and GPU configurations, using the same documents, questions,
request mode, warm-up count, and network path. The tool does not deploy the
application, load models, instrument backend components, generate concurrent
traffic, score answer quality, or verify that a deployment's actual hardware
matches its label. It does not measure maximum service capacity.

### Why benchmark the application?

An LLM's inference speed is only one part of a retrieval-augmented ChatQnA
request. A request can also spend time in the API and network, document
retrieval, embedding or reranking, prompt construction, response streaming,
and client-side response handling. Measuring the deployed API shows the
latency and throughput an application client experiences across that whole
path. It also allows chat and document-ingestion behavior to be compared with
consistent inputs.

## Application KPIs and LLM KPIs

This tool's primary KPIs describe the **application/API request**, not an
isolated language model:

| Benchmark collects | What It Means |
|---|---|
| End-to-end request latency | Time from immediately before the HTTP request is dispatched until the response body is processed and validated by this client |
| First-visible-text latency | For streaming chat only, time to the first non-whitespace answer text observed by this client |
| Maximum inter-chunk gap | For streaming chat, the largest observed gap between chunks containing visible text |
| Sequential successful requests/s | Successful measured requests divided by that mode's complete sequential measurement window |
| Failure rate and error categories | The fraction of attempted measured requests that failed and the benchmark's classification of those failures |
| Document ingestion documents/s and MiB/s | Successful document uploads and their source bytes divided by the complete ingestion window |
| Input and answer character counts | Unicode code points in submitted prompts and reconstructed answers |
| Input and answer token counts (when available) | Offline retokenization statistics for submitted prompt text and visible answer text |
| Optional resource utilization | System CPU user utilization, GPU utilization, NPU utilization, memory-used percentage, and temperature samples |

LLM or inference-engine KPIs typically describe model execution more directly:
prompt-processing and generation tokens/s, per-generated-token latency
(inter-token latency), model queue time, batch utilization, or KV-cache use.
This tool does **not** collect those engine-level metrics. In particular:

- First-visible-text latency is an API-level observation, not model-engine
  time-to-first-token (TTFT). It includes everything between client dispatch
  and visible text arriving, and may be affected by network or proxy buffering.
- The tool's token counts are added by offline tokenization after measurement.
  They are not generated-token counts from the model and are not used to
  calculate model tokens/s.
- Sequential successful requests/s is application throughput for a
  single-client, one-request-at-a-time run. It is neither concurrent capacity
  nor model token throughput.
- Optional resource metrics are system utilization samples, not per-model or
  per-process attribution.

## Workload

From the ChatQnA Core sample-application root, prepare the following files:

```text
benchmarking/
├── benchmark.py
├── postprocess.py
├── resource_collection.py
├── requirements.txt
├── requirements-resource.txt
├── data/                 # Add source documents here
│   ├── manual.pdf
│   ├── faq.txt
│   └── subfolder/
│       └── guide.docx
├── questions.txt         # Chat prompts; blank-line separated
└── reports/              # One generated directory per run
```

The document directory is searched recursively for `.pdf`, `.docx`, and `.txt`
files. Hidden files and unsupported formats are ignored. Document filenames
must be unique, including across subdirectories, because the API identifies
documents by filename. Symbolic-link documents are not supported. At least one
supported document is required for both benchmark types.

The questions file must be UTF-8 text. Separate prompts with one or more blank
lines; line breaks within a prompt are preserved. Each prompt becomes one chat
request, and each request uses the complete document corpus:

```text
What troubleshooting steps are listed for the device?

Summarize the setup requirements in the guide.
```

The questions file is not uploaded as a document. It is not read for a
document-ingestion run.

The report records a workload fingerprint calculated from sorted document
filenames and content hashes plus the prompt text. This lets you compare runs
for input consistency without placing source document contents in the
fingerprint.

## Setup

Requires Python 3.11 or newer on Linux. Deploy ChatQnA Core separately and
ensure its API is reachable before running a benchmark. Commands below are
run from the sample-application root:

```bash
mkdir -p benchmarking/data
# Copy .pdf, .docx, and/or .txt files into benchmarking/data/
# Create benchmarking/questions.txt with blank-line-separated prompts
python3 -m venv benchmarking/.venv
benchmarking/.venv/bin/python -m pip install -r benchmarking/requirements.txt
```

The benchmark uses a single reused HTTPX client and sends requests
sequentially. It bypasses environment HTTP proxies for the selected API URL;
HTTPS certificate verification remains enabled. Health and model metadata are
checked before measurement. Docker is used on a best-effort basis to read
deployment details and, when possible, retrieve the deployed LLM tokenizer.
The client checks `/health` and `/model`, uses `/documents` for corpus
inventory and ingestion, and sends chat requests to `/chat`. Responses larger
than 8 MiB are treated as failures.

## Run the benchmark

### Chat API

```bash
benchmarking/.venv/bin/python benchmarking/benchmark.py \
  --api chat \
  --stream both \
  --warmup 3
```

For chat, `--stream` defaults to `true`. `--stream both` measures streaming
requests first and then non-streaming JSON requests, producing independent
summaries. Each mode receives its own warm-up requests before measurement. Each
question is sent once per measured mode. If the document store is empty, the
benchmark uploads the workload documents once as unmeasured corpus setup and
checks that the resulting document inventory matches.

### Document-ingestion API

```bash
benchmarking/.venv/bin/python benchmarking/benchmark.py --api documents
```

Each source document is uploaded separately and included in the measured
ingestion window. The store is emptied first by default; ingestion therefore
measures adding documents to a progressively growing store. This is not an
isolated embedding-model throughput measurement.

### Resource metrics (optional)

```bash
benchmarking/.venv/bin/python benchmarking/benchmark.py \
  --api chat \
  --stream both \
  --collect-resource-metrics
```

The collector starts after chat warm-up and stops after measurement. It
launches the Intel Metrics Manager container using the local Docker CLI and
requires Docker access, host device/system metrics access, and the collector's
ports to be available. Metrics are polled about once per second. Collection is
best-effort; if it cannot start, the benchmark reports a warning and
continues. Resource metrics are not per-process attribution.

## Command-line arguments

Run `benchmarking/.venv/bin/python benchmarking/benchmark.py --help` for the
CLI help. Arguments are:

| Argument | Default | Description |
|---|---|---|
| `--api` | Required | Benchmark `chat` or `documents` |
| `--base-url` | `http://localhost:8102/v1/chatqna` | Base URL of the deployed API; must be an HTTP(S) URL without credentials, query, or fragment |
| `--stream` | `true` for chat | Chat response mode: `true`, `false`, or `both`. Not valid with `--api documents` |
| `--warmup` | `3` | Number of excluded warm-up chat requests per mode; `0` disables warm-up |
| `--timeout` | `300` | HTTP operation/inactivity timeout in seconds; must be finite and greater than zero |
| `--data-dir` | `benchmarking/data` | Directory of source documents, searched recursively |
| `--questions-file` | `benchmarking/questions.txt` | UTF-8 `.txt` file of blank-line-separated prompts; chat only |
| `--reports-dir` | `benchmarking/reports` | Parent directory for run reports |
| `--reset-documents` / `--no-reset-documents` | `--reset-documents` | Delete existing embeddings before uploads, or require an empty store |
| `--collect-resource-metrics` | Off | Collect optional CPU/GPU/NPU/memory/temperature metrics during the measured phase |

`--warmup` must be zero or greater. `--timeout` must be finite and greater
than zero. The selected workload must contain at least one supported document;
chat also requires at least one non-empty prompt. Failed requests are not
retried. For repeatable comparisons, use the same workload, request mode,
warm-up count, timeout, and client-to-service network path.

## KPIs and interpretation

All durations use `time.perf_counter_ns()`, a monotonic high-resolution clock.
Per-request latency starts immediately before HTTP dispatch and includes
response delivery plus client-side response decoding and validation; it is
not backend-only processing time.

Latency distributions include successful requests only and report `count`,
`min`, `p50`, `p90`, `p95`, and `max`. Percentiles use linear interpolation.
There is no p99. Failed requests remain in `requests.jsonl` with their elapsed
time, HTTP status when available, error category, and partial-output status.

| KPI | Definition and scope |
|---|---|
| `successful_end_to_end_ms` | Distribution of successful chat responses or document uploads, measured from client dispatch through response validation |
| `successful_first_visible_text_ms` | Streaming chat only: distribution of time to the first non-whitespace text observed in a stream |
| `successful_max_inter_chunk_gap_ms` | Streaming chat only: distribution of each successful response's largest gap between text-bearing stream chunks; responses with fewer than two visible chunks have no value |
| `successful_answer_char_count` | Distribution of answer Unicode code-point counts; this is not a token or grapheme count |
| `failure_rate_pct` | Failed measured requests divided by all attempted measured requests, expressed as a percentage |
| `sequential_successful_requests_per_s` | Successful requests divided by that mode's measurement window; the window includes failed requests and client loop overhead |
| `documents_per_s` | Successful document uploads divided by the complete ingestion window |
| `input_mib_per_s` | Bytes of successfully uploaded source documents divided by 2^20 and by the ingestion window |
| `successful_input_tokens` | Offline token-count distribution for submitted question text, when the deployed tokenizer is available |
| `successful_answer_tokens` | Offline token-count distribution for reconstructed visible answer text, when the deployed tokenizer is available |

The chat measurement window excludes corpus setup, warm-up, tokenization, and
report writing. For `--stream both`, streaming and JSON each have a separate
window and summary. Document ingestion includes the entire sequential upload
loop. Health/model checks and document reset are setup operations, not
measured endpoints.

The application emits raw `data: {text}\n\n` streaming frames, not token IDs or
a completion marker. Therefore:

- Chunk timings are based on client-observed HTTP body reads. A single read
  may contain several frames with identical observed arrival times; these
  values are not inter-token latency.
- The stream parser handles fragmented UTF-8 and preserves payload
  whitespace/newlines. The raw format is ambiguous if answer text itself
  contains the exact delimiter `\n\ndata: `.
- A cleanly ended but semantically truncated answer cannot be detected
  reliably without a finish reason. Transport errors and missing final frame
  delimiters are detected; failed responses may retain received text as
  partial output.
- Proxy buffering can delay or group chunks. Check behavior through the actual
  deployment path and use the same path for comparisons.

The tool does not calculate SLO goodput, answer quality, concurrent load,
capacity, model tokens/s, or model-engine TTFT.

## Reports

Every run that passes local workload validation gets a new UTC timestamp and
unique-ID directory under `--reports-dir`. Existing reports are not
overwritten or deleted. A run that fails during setup or is interrupted may
have a partial report; check `manifest.json`'s `state` before treating the
results as complete.

```text
reports/<UTC-time>-<unique-id>/
├── manifest.json                 # Workload fingerprint, run settings, state, model metadata
├── configuration.yaml            # Best-effort container image, model/runtime, and device details
├── setup-ingestion.json          # Chat corpus setup results, when setup uploads are made
├── chat-stream-warmup.json       # Streaming warm-up results; excluded from KPIs
├── chat-json-warmup.json         # JSON warm-up results; only for JSON/both runs
├── requests.jsonl                # One JSON record per measured request
├── tokenized-requests.jsonl      # Chat text plus offline input/answer token counts, if available
├── tokenization.json             # Tokenizer identity/settings and successful token statistics
├── summary.json                  # Full per-mode KPI distributions and corpus document sizes
├── summary.csv                   # Condensed, tabular representation of summary.json
└── resource_metrics/             # Only when --collect-resource-metrics is requested
    └── perf_tool_logs/
        ├── metrics_stream.log
        ├── resource_utilization.png
        └── metrics_summary.csv
```

Files that do not apply to a run may be absent. `configuration.yaml` and
tokenization data depend on successfully finding the running backend container
through Docker; the benchmark warns and continues if that information is
unavailable. In particular, offline tokenization may be skipped for runtimes
or deployments from which the matching `tokenizer.json` cannot be copied.

`summary.json` contains full distributions and one KPI object per mode
(`chat-stream`, `chat-json`, or `documents`), plus `corpus_documents` with each
input filename and byte size. Each distribution has its applicable statistics
and count. `summary.csv` is a condensed view: it uses p90 for distributions,
`avg` for token-count statistics, reports document sizes as `size_mb`, and
keeps scalar metrics as-is.

The measured `requests.jsonl` records include submitted prompts and returned
answers; `tokenized-requests.jsonl` also includes that text. Treat reports as
potentially sensitive, restrict access, and review them before sharing.

## Offline tokenization

For chat runs, the tool attempts tokenization after measurement and never
during it. It copies the deployed LLM's `tokenizer.json` from
`/tmp/model_cache/<llm-model-id>/tokenizer.json` inside the running backend
container. It retokenizes submitted prompt text and reconstructed visible
answer text with special tokens, padding, and truncation disabled. These
counts do not include retrieved RAG context or the prompt template, and
retokenized answer counts are not guaranteed to match the model's actual
generated-token count.

To tokenize a saved run manually with a local tokenizer, use:

```bash
benchmarking/.venv/bin/python benchmarking/postprocess.py \
  --run-dir benchmarking/reports/<run-id> \
  --tokenizer /path/to/matching/tokenizer.json \
  --tokenizer-id microsoft/Phi-3.5-mini-instruct \
  --tokenizer-revision <exact-tokenizer-revision>
```

The postprocessor refuses to overwrite existing tokenization output.

## Tests

The synthetic unit tests are offline and do not contact a deployed service:

```bash
benchmarking/.venv/bin/python -m unittest discover \
  -s benchmarking -p 'test_*.py' -v
```

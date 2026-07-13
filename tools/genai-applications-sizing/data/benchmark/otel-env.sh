#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# otel-env.sh
# Standard OpenTelemetry environment variables that direct OpenClaw's
# observability plugin (and any OTel SDK) to export traces + metrics to the
# local SigNoz collector.
#
# These are the *vendor-neutral* OTEL_* variables defined by the OpenTelemetry
# spec, so they work regardless of the exact plugin config schema. Source this
# file before launching OpenClaw and before running the benchmark:
#
#   source ./benchmark/otel-env.sh
#   ./run-agent.sh            # OpenClaw now exports to SigNoz
#   ./benchmark/run-benchmark.sh
#
# Override any value by exporting it before sourcing (e.g. a remote endpoint).

# OTLP HTTP endpoint of the SigNoz collector (use 4317 for gRPC instead).
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://127.0.0.1:4318}"
export OTEL_EXPORTER_OTLP_PROTOCOL="${OTEL_EXPORTER_OTLP_PROTOCOL:-http/protobuf}"

# Enable trace + metric export; disable logs unless you want them too.
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-otlp}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-otlp}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"

# Service identity shown in SigNoz. Filter on service.name = this value.
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-teacher-assistant-openclaw}"

# Resource attributes let you separate benchmark runs in SigNoz.
# BENCH_SESSION is set per-run by run-benchmark.sh; default to 'manual' here.
export OTEL_RESOURCE_ATTRIBUTES="${OTEL_RESOURCE_ATTRIBUTES:-service.namespace=teacher-assistant,deployment.environment=benchmark,bench.session=${BENCH_SESSION:-manual}}"

# Export spans promptly so short benchmark runs are not lost on exit.
export OTEL_BSP_SCHEDULE_DELAY="${OTEL_BSP_SCHEDULE_DELAY:-1000}"

echo "OTel export configured -> ${OTEL_EXPORTER_OTLP_ENDPOINT} (service.name=${OTEL_SERVICE_NAME})"

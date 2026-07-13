#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# run-benchmark.sh
# Drives a fixed set of classroom_qa prompts through the OpenClaw agent and
# records per-run client-side KPIs (end-to-end latency and a time-to-first-
# output proxy). Detailed span-level KPIs (agent completion time, tool
# execution time, OVMS inference latency, tokens/sec, authoritative TTFT) are
# collected by the OpenClaw observability plugin and viewed in SigNoz.
#
# Prerequisites:
#   1. OVMS running and model loaded         (../setup-ovms.sh)
#   2. SigNoz running                         (./setup-signoz.sh)
#   3. Observability plugin enabled           (openclaw config patch --file ./openclaw-observability.json && openclaw gateway install)
#   4. OTel env exported                      (source ./otel-env.sh)
#
# Usage:
#   ./run-benchmark.sh [-n ITERATIONS] [-w WARMUP] [-p PROMPTS_FILE] [-i INVOKE]
#
#   -n  iterations per prompt   (default: 3)
#   -w  warmup runs (discarded) (default: 1)
#   -p  prompts file            (default: ./prompts.txt)
#   -i  invocation method       (default: $OPENCLAW_INVOKE or "run")
#
# Invocation methods (-i / OPENCLAW_INVOKE) — set to match your OpenClaw CLI:
#   run        -> openclaw run "<prompt>"
#   ask        -> openclaw ask "<prompt>"
#   chat-pipe  -> printf '<prompt>\n' | openclaw chat
#   custom     -> use OPENCLAW_CMD_TEMPLATE with a {{PROMPT}} placeholder
#   dry        -> no OpenClaw call; validates the harness plumbing only
#
# Confirm the correct method with:  openclaw --help   (see README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ITERATIONS=3
WARMUP=1
PROMPTS_FILE="${SCRIPT_DIR}/prompts.txt"
OPENCLAW_INVOKE="${OPENCLAW_INVOKE:-run}"

while getopts "n:w:p:i:h" opt; do
    case "${opt}" in
        n) ITERATIONS="${OPTARG}" ;;
        w) WARMUP="${OPTARG}" ;;
        p) PROMPTS_FILE="${OPTARG}" ;;
        i) OPENCLAW_INVOKE="${OPTARG}" ;;
        h) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Try: $0 -h"; exit 1 ;;
    esac
done

[ -f "${PROMPTS_FILE}" ] || { echo "ERROR: prompts file not found: ${PROMPTS_FILE}"; exit 1; }

# Per-run session id ties client CSV rows to spans in SigNoz (bench.session attr).
BENCH_SESSION="bench-$(date +%Y%m%d-%H%M%S)"
export BENCH_SESSION

# Load OTel env (also (re)exports resource attrs with this BENCH_SESSION).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/otel-env.sh"

RESULTS_DIR="${SCRIPT_DIR}/results/${BENCH_SESSION}"
OUT_DIR="${RESULTS_DIR}/outputs"
mkdir -p "${OUT_DIR}"
CSV="${RESULTS_DIR}/results.csv"
echo "session,phase,iter,prompt_index,invoke,e2e_ms,ttft_ms,exit_code,output_chars,prompt" > "${CSV}"

# Read non-comment, non-empty prompt lines into an array.
mapfile -t PROMPTS < <(grep -vE '^\s*(#|$)' "${PROMPTS_FILE}")
[ "${#PROMPTS[@]}" -gt 0 ] || { echo "ERROR: no prompts found in ${PROMPTS_FILE}"; exit 1; }

echo "=== OpenClaw classroom_qa benchmark ==="
echo "session     : ${BENCH_SESSION}"
echo "prompts     : ${#PROMPTS[@]} (from ${PROMPTS_FILE})"
echo "iterations  : ${ITERATIONS} (+${WARMUP} warmup) per prompt"
echo "invoke      : ${OPENCLAW_INVOKE}"
echo "results dir : ${RESULTS_DIR}"
echo ""

# Runs one prompt once. Writes timestamped output lines to a temp file so we can
# derive a client-side time-to-first-output proxy and the end-to-end latency.
# Echoes: "<e2e_ms> <ttft_ms> <exit_code> <output_chars> <outfile>"
run_once() {
    local prompt="$1" outfile="$2"
    local q; printf -v q '%q' "${prompt}"
    local cmd
    case "${OPENCLAW_INVOKE}" in
        run)       cmd="openclaw run ${q}" ;;
        ask)       cmd="openclaw ask ${q}" ;;
        chat-pipe) cmd="printf '%s\n' ${q} | openclaw chat" ;;
        custom)    cmd="${OPENCLAW_CMD_TEMPLATE//\{\{PROMPT\}\}/${q}}" ;;
        dry)       cmd="printf 'DRY reply to: %s\n' ${q}" ;;
        *) echo "ERROR: unknown invoke method '${OPENCLAW_INVOKE}'" >&2; return 2 ;;
    esac

    local tsfile; tsfile="$(mktemp)"
    local start_ns end_ns
    start_ns="$(date +%s%N)"
    # Timestamp each output line as it arrives (line-buffered proxy for TTFT).
    set +e
    bash -o pipefail -c "${cmd}" 2>&1 \
        | while IFS= read -r line; do printf '%s\t%s\n' "$(date +%s%N)" "${line}"; done > "${tsfile}"
    local rc="${PIPESTATUS[0]}"
    set -e
    end_ns="$(date +%s%N)"

    # Strip timestamps into the clean output file; keep first-line ns for TTFT.
    local first_ns=""
    if [ -s "${tsfile}" ]; then
        first_ns="$(head -n1 "${tsfile}" | cut -f1)"
    fi
    cut -f2- "${tsfile}" > "${outfile}"
    local chars; chars="$(wc -c < "${outfile}" | tr -d ' ')"
    rm -f "${tsfile}"

    local e2e_ms ttft_ms
    e2e_ms="$(( (end_ns - start_ns) / 1000000 ))"
    if [ -n "${first_ns}" ]; then
        ttft_ms="$(( (first_ns - start_ns) / 1000000 ))"
    else
        ttft_ms=""
    fi
    echo "${e2e_ms} ${ttft_ms} ${rc} ${chars} ${outfile}"
}

record() {
    local phase="$1" iter="$2" pidx="$3" prompt="$4" res="$5"
    local e2e ttft rc chars outfile
    read -r e2e ttft rc chars outfile <<< "${res}"
    # CSV-escape the prompt (wrap in quotes, double any internal quotes).
    local esc="${prompt//\"/\"\"}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
        "${BENCH_SESSION}" "${phase}" "${iter}" "${pidx}" "${OPENCLAW_INVOKE}" \
        "${e2e}" "${ttft}" "${rc}" "${chars}" "${esc}" >> "${CSV}"
    printf '  [%s] prompt %-2s iter %-2s  e2e=%sms  ttft=%sms  rc=%s\n' \
        "${phase}" "${pidx}" "${iter}" "${e2e}" "${ttft:-NA}" "${rc}"
}

for pidx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$pidx]}"
    echo "Prompt ${pidx}: ${prompt}"

    for ((w = 1; w <= WARMUP; w++)); do
        of="${OUT_DIR}/p${pidx}_warmup${w}.txt"
        res="$(run_once "${prompt}" "${of}")"
        record "warmup" "${w}" "${pidx}" "${prompt}" "${res}"
    done

    for ((i = 1; i <= ITERATIONS; i++)); do
        of="${OUT_DIR}/p${pidx}_iter${i}.txt"
        res="$(run_once "${prompt}" "${of}")"
        record "measure" "${i}" "${pidx}" "${prompt}" "${res}"
    done
    echo ""
done

echo "=== Done. Raw results: ${CSV} ==="
echo ""
if command -v python3 >/dev/null 2>&1; then
    python3 "${SCRIPT_DIR}/analyze.py" "${CSV}"
else
    echo "Install python3 to see the summary, or open ${CSV} directly."
fi

echo ""
echo "Span-level KPIs (agent completion, tool execution, OVMS inference,"
echo "tokens/sec, authoritative TTFT) are in SigNoz. Filter traces by:"
echo "  service.name = ${OTEL_SERVICE_NAME}   AND   bench.session = ${BENCH_SESSION}"

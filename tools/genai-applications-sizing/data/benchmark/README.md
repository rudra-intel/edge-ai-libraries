# Benchmarking the Teacher Assistant (OpenClaw) with OpenTelemetry + SigNoz

This harness measures performance KPIs for the OpenClaw agent by exporting
OpenTelemetry traces/metrics to a local **SigNoz** backend and driving a fixed
set of `classroom_qa` prompts.

## KPIs collected

| KPI | Where it comes from | How to read it |
|---|---|---|
| **End-to-end latency** | Client wall clock + root span | `results.csv` / `analyze.py`, and root span duration in SigNoz |
| **Agent completion time** | `agent.turn` span | SigNoz span duration |
| **Tool execution time** | `tool.execute` spans (bash / read) | SigNoz span duration, grouped by tool name |
| **OVMS inference latency** | `model.completion` span | SigNoz span duration |
| **Tokens/sec** | `model.completion` span attrs | `gen_ai.usage.output_tokens` ÷ span duration |
| **Time-to-first-token** | streaming start event / attr | authoritative in SigNoz; client-side proxy in `results.csv` |

> Span names above match `captureSpans` in `openclaw-observability.json`. If your
> OpenClaw build emits different span names, adjust filters accordingly (the
> OpenTelemetry GenAI convention uses `gen_ai.*` attributes).

## Files

| File | Purpose |
|---|---|
| `setup-signoz.sh` | Clone + start/stop the SigNoz OTLP backend (Docker) |
| `otel-env.sh` | Standard `OTEL_*` env vars pointing OpenClaw at SigNoz |
| `openclaw-observability.json` | Config patch enabling the observability plugin |
| `prompts.txt` | Representative `classroom_qa` prompts (edit freely) |
| `run-benchmark.sh` | Driver: runs prompts, records client KPIs to `results/` |
| `analyze.py` | Prints min/mean/p50/p90/p95/max from a results CSV |

## Two values you must confirm for YOUR OpenClaw build

Everything else is standard OpenTelemetry/SigNoz. Confirm these two first:

```bash
# 1) Exact observability plugin key + its config schema:
openclaw plugins list
openclaw plugins info observability   # or the name shown by the command above

# 2) The non-interactive way to send a prompt (pick one for -i / OPENCLAW_INVOKE):
openclaw --help                       # look for: run | ask | exec | a chat pipe
```

- Update the plugin key/fields in `openclaw-observability.json` if they differ
  from `plugins.entries.observability`.
- Set the driver's invocation method with `-i` (or `OPENCLAW_INVOKE`) to `run`,
  `ask`, `chat-pipe`, or `custom` (with `OPENCLAW_CMD_TEMPLATE="... {{PROMPT}} ..."`).

## Run order

```bash
cd benchmark

# 1. Start the OTLP backend (SigNoz UI at http://127.0.0.1:8080).
./setup-signoz.sh

# 2. Make sure OVMS is up and the model is loaded.
../setup-ovms.sh

# 3. Enable the observability plugin, then reinstall the gateway.
openclaw config patch --file ./openclaw-observability.json
openclaw gateway install

# 4. Point OpenClaw at SigNoz and (re)start the agent in another shell.
source ./otel-env.sh
../run-agent.sh          # OpenClaw now exports spans/metrics to SigNoz

# 5. Validate the harness end-to-end without OVMS (optional).
./run-benchmark.sh -i dry -n 1 -w 0

# 6. Run the real benchmark (3 measured iterations + 1 warmup per prompt).
./run-benchmark.sh -n 3 -w 1 -i run
```

Client-side latency stats print at the end and are saved under
`results/<session>/results.csv`. Raw agent outputs are in
`results/<session>/outputs/`.

## Reading span KPIs in SigNoz

Open http://127.0.0.1:8080 → **Traces**, then filter:

```
service.name = teacher-assistant-openclaw
bench.session = bench-<timestamp>     # printed at the end of each run
```

- Group/aggregate by span name to get p50/p90/p95 of `agent.turn`,
  `tool.execute`, and `model.completion` durations.
- Open a single trace to see the full agent → tool → model waterfall for one
  prompt (this shows where the end-to-end time is spent).
- For tokens/sec, add a column/aggregation on `gen_ai.usage.output_tokens` and
  divide by `model.completion` duration.
- Build a dashboard (Dashboards → New) with these as saved panels for repeatable
  comparisons across runs (compare by the `bench.session` attribute).

## Notes

- The client-side TTFT is a line-buffered proxy; treat SigNoz's streaming-start
  span/event as authoritative.
- Keep OVMS `--cache_size` and `--target_device` constant across runs you intend
  to compare.
- To wipe SigNoz data between comparison campaigns: `./setup-signoz.sh purge`.

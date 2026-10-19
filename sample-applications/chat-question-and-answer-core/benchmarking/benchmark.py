# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sequential black-box ChatQnA benchmark. No application imports or deployment."""

import argparse
import codecs
from collections import deque
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable
from urllib.parse import urlsplit
import uuid

import httpx

ROOT = Path(__file__).resolve().parent
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SUPPORTED_FORMATS = {".docx", ".pdf", ".txt"}

# Deployed ChatQnA Core backend container candidates (nginx/UI are excluded).
CHATQNA_CONTAINER_NAMES = (
    "chatqna-core-ov-gpu",
    "chatqna-core-ov-cpu",
    "chatqna-core-ollama",
)
# Model cache location inside the backend container (see docker/Dockerfile).
CONTAINER_MODEL_CACHE = "/tmp/model_cache"
# In-container snippet that prints the effective model/device configuration.
_CONTAINER_CONFIG_SNIPPET = (
    "import json,sys;"
    "sys.path[:0]=['/my-app','/app'];"
    "from app.config import config;"
    "print(json.dumps({"
    "'model_runtime':config.MODEL_RUNTIME,"
    "'llm_model':config.LLM_MODEL_ID,"
    "'embedding_model':config.EMBEDDING_MODEL_ID,"
    "'reranker_model':config.RERANKER_MODEL_ID,"
    "'llm_device':config.LLM_DEVICE,"
    "'embedding_device':config.EMBEDDING_DEVICE,"
    "'reranker_device':config.RERANKER_DEVICE}))"
)


@dataclass(frozen=True)
class Document:
    name: str
    content: bytes


@dataclass(frozen=True)
class Question:
    question_id: str
    documents: list[str]
    input: str


@dataclass(frozen=True)
class Workload:
    documents: list[Document]
    questions: list[Question]
    fingerprint: str


def load_workload(data_dir: Path, questions_file: Path | None) -> Workload:
    if not data_dir.is_dir():
        raise ValueError(f"Document folder does not exist: {data_dir}.")

    question_path = questions_file.resolve() if questions_file is not None else None
    if questions_file is not None:
        if not questions_file.is_file():
            raise ValueError(f"Questions file does not exist: {questions_file}.")
        if questions_file.suffix.lower() != ".txt":
            raise ValueError(f"Questions file must be a .txt file: {questions_file}.")

    document_paths = sorted(
        (
            path for path in data_dir.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
            and path.suffix.lower() in SUPPORTED_FORMATS
            and (question_path is None or path.resolve() != question_path)
        ),
        key=lambda path: path.relative_to(data_dir).as_posix(),
    )
    if not document_paths:
        raise ValueError(
            f"No supported documents found in {data_dir}; add .pdf, .txt, or .docx files."
        )
    names = [path.name for path in document_paths]
    if len(names) != len(set(names)):
        raise ValueError("Document filenames must be unique, including across subfolders.")
    if any(path.is_symlink() for path in document_paths):
        raise ValueError("Symbolic-link documents are not supported.")

    documents = [Document(path.name, path.read_bytes()) for path in document_paths]
    prompt_texts: list[str] = []
    if questions_file is not None:
        contents = questions_file.read_text(encoding="utf-8")
        prompt_texts = [
            prompt.strip() for prompt in re.split(r"\n\s*\n", contents.strip()) if prompt.strip()
        ]
        if not prompt_texts:
            raise ValueError(f"No prompts found in {questions_file}.")

    document_names = [document.name for document in documents]
    questions = [
        Question(f"q{index:04d}", document_names, prompt)
        for index, prompt in enumerate(prompt_texts, start=1)
    ]
    fingerprint_input = {
        "documents": [
            {"name": document.name, "sha256": hashlib.sha256(document.content).hexdigest()}
            for document in documents
        ],
        "questions": [question.input for question in questions],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return Workload(documents, questions, fingerprint)


class ResponseError(ValueError):
    """The response is not a valid successful application response."""


class ChatStream:
    """Parse this app's raw `data: {text}\\n\\n` frames, including raw newlines.

    Frame boundaries use the next data prefix, or EOF, to preserve newlines in
    payloads. Arrival offsets retain original read times despite delayed parsing.
    """

    PREFIX = "data: "
    BOUNDARY = "\n\ndata: "

    def __init__(self) -> None:
        self.buffer = ""
        self.offset = 0
        self.total = 0
        self.arrivals: deque[tuple[int, int]] = deque()
        self.started = False

    def _frame(self, payload: str) -> tuple[str, int | None]:
        visible = re.search(r"\S", payload)
        if visible is None:
            return payload, None
        position = self.offset + visible.start()
        arrival = next(timestamp for end, timestamp in self.arrivals if end > position)
        return payload, arrival

    def _consume(self, count: int) -> None:
        self.buffer = self.buffer[count:]
        self.offset += count
        while self.arrivals and self.arrivals[0][0] <= self.offset:
            self.arrivals.popleft()

    def feed(self, text: str, timestamp_ns: int) -> list[tuple[str, int | None]]:
        self.buffer += text
        self.total += len(text)
        self.arrivals.append((self.total, timestamp_ns))
        if not self.started:
            if len(self.buffer) < len(self.PREFIX):
                return []
            if not self.buffer.startswith(self.PREFIX):
                raise ResponseError("Unexpected ChatQnA stream prefix.")
            self._consume(len(self.PREFIX))
            self.started = True
        frames: list[tuple[str, int | None]] = []
        while (boundary := self.buffer.find(self.BOUNDARY)) >= 0:
            frames.append(self._frame(self.buffer[:boundary]))
            self._consume(boundary + len(self.BOUNDARY))
        return frames

    def finish(self) -> list[tuple[str, int | None]]:
        if not self.started or not self.buffer.endswith("\n\n"):
            raise ResponseError("Missing final ChatQnA stream frame terminator.")
        frame = self._frame(self.buffer[:-2])
        self._consume(len(self.buffer))
        return [frame]

    def partial(self) -> str:
        if not self.started:
            return ""
        return self.buffer[:-2] if self.buffer.endswith("\n\n") else self.buffer

    def partial_frame(self) -> list[tuple[str, int | None]]:
        payload = self.partial()
        return [self._frame(payload)] if payload else []


@dataclass
class Result:
    request_id: str
    api: str
    stream: bool | None
    input: str = ""
    documents: list[str] = field(default_factory=list)
    outcome: str = "success"
    error_category: str | None = None
    error_detail: str | None = None
    http_status: int | None = None
    elapsed_ms: float = 0
    first_visible_text_ms: float | None = None
    inter_chunk_gaps_ms: list[float] = field(default_factory=list)
    max_inter_chunk_gap_ms: float | None = None
    visible_chunks: int = 0
    answer: str = ""
    answer_char_count: int = 0
    input_unicode_codepoints: int = 0
    document_bytes: int = 0
    output_is_partial: bool = False


def read_body(response: httpx.Response) -> bytes:
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ResponseError("Response exceeds the 8 MiB limit.")
    return bytes(body)


def success_json(body: bytes) -> dict:
    data = json.loads(body)
    if not isinstance(data, dict) or data.get("status") != "Success":
        raise ResponseError("Expected application status Success.")
    return data


def error_detail(error: Exception) -> str:
    if isinstance(error, ResponseError):
        return str(error)
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP status {error.response.status_code}."
    if isinstance(error, httpx.TimeoutException):
        return "HTTP operation/inactivity timeout."
    if isinstance(error, json.JSONDecodeError):
        return "Malformed JSON response."
    if isinstance(error, UnicodeError):
        return "Invalid UTF-8 response."
    return "HTTP transport/protocol failure."


def measure_chat(
    client: httpx.Client,
    question: Question,
    stream: bool,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Result:
    result = Result(question.question_id, "chat", stream, question.input, list(question.documents))
    result.input_unicode_codepoints = len(question.input)
    parser = ChatStream()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts: list[str] = []
    arrivals: list[int] = []

    def collect(frames: list[tuple[str, int | None]]) -> None:
        for text, arrival in frames:
            parts.append(text)
            if arrival is not None:
                arrivals.append(arrival)

    start = clock()
    try:
        with client.stream("POST", "chat", json={"input": question.input, "stream": stream}) as response:
            result.http_status = response.status_code
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if stream:
                if content_type != "text/event-stream":
                    raise ResponseError("Expected text/event-stream.")
                size = 0
                for chunk in response.iter_bytes():
                    arrival = clock()
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ResponseError("Response exceeds the 8 MiB limit.")
                    collect(parser.feed(decoder.decode(chunk), arrival))
                collect(parser.feed(decoder.decode(b"", final=True), clock()))
                collect(parser.finish())
            else:
                if content_type != "application/json":
                    raise ResponseError("Expected application/json.")
                data = success_json(read_body(response))
                answer = data.get("metadata")
                if not isinstance(answer, str):
                    raise ResponseError("Expected a string answer in metadata.")
                parts.append(answer)
            end = clock()
            if not "".join(parts).strip():
                raise ResponseError("Empty answer.")
    except (httpx.HTTPError, ResponseError, UnicodeError, json.JSONDecodeError) as error:
        end = clock()
        result.outcome = "failure"
        result.error_category = type(error).__name__
        result.error_detail = error_detail(error)
        result.output_is_partial = True
        if stream:
            collect(parser.partial_frame())
    result.elapsed_ms = (end - start) / 1_000_000
    result.answer = "".join(parts)
    result.answer_char_count = len(result.answer)
    if arrivals:
        result.first_visible_text_ms = (arrivals[0] - start) / 1_000_000
    result.visible_chunks = len(arrivals)
    result.inter_chunk_gaps_ms = [
        (right - left) / 1_000_000 for left, right in zip(arrivals, arrivals[1:])
    ]
    result.max_inter_chunk_gap_ms = max(result.inter_chunk_gaps_ms, default=None)
    return result


def measure_document(
    client: httpx.Client,
    document: Document,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Result:
    body = document.content
    result = Result(document.name, "documents", None, documents=[document.name], document_bytes=len(body))
    start = clock()
    try:
        with client.stream(
            "POST", "documents",
            files={"files": (document.name, body, mimetypes.guess_type(document.name)[0] or "application/octet-stream")},
        ) as response:
            result.http_status = response.status_code
            response.raise_for_status()
            data = success_json(read_body(response))
            metadata = data.get("metadata")
            names = metadata.get("documents") if isinstance(metadata, dict) else None
            if not isinstance(names, list) or document.name not in names:
                raise ResponseError("Upload response does not confirm the document.")
            end = clock()
    except (httpx.HTTPError, ResponseError, UnicodeError, json.JSONDecodeError) as error:
        end = clock()
        result.outcome = "failure"
        result.error_category = type(error).__name__
        result.error_detail = error_detail(error)
    result.elapsed_ms = (end - start) / 1_000_000
    return result


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * fraction
        low = math.floor(index)
        high = math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {
        "count": len(values),
        "min": ordered[0] if ordered else None,
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": ordered[-1] if ordered else None,
    }


def summarize(
    results: list[Result], window_s: float
) -> dict[str, object]:
    successful = [result for result in results if result.outcome == "success"]
    streaming = [result for result in successful if result.stream]
    gap_results = [result for result in streaming if result.max_inter_chunk_gap_ms is not None]
    errors: dict[str, int] = {}
    for result in results:
        if result.error_category:
            errors[result.error_category] = errors.get(result.error_category, 0) + 1
    summary: dict[str, object] = {
        "attempted": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "failure_rate_pct": 100 * (len(results) - len(successful)) / len(results) if results else None,
        "total_duration_s": window_s,
        "sequential_successful_requests_per_s": len(successful) / window_s if window_s > 0 else None,
        "successful_end_to_end_ms": distribution([result.elapsed_ms for result in successful]),
        "successful_first_visible_text_ms": distribution([
            result.first_visible_text_ms for result in streaming
            if result.first_visible_text_ms is not None
        ]),
        "successful_max_inter_chunk_gap_ms": distribution([
            result.max_inter_chunk_gap_ms for result in gap_results
            if result.max_inter_chunk_gap_ms is not None
        ]),
        "successful_answer_char_count": distribution([
            float(result.answer_char_count) for result in successful if result.api == "chat"
        ]),
        "errors_by_category": errors,
    }
    if results and results[0].api == "documents":
        summary["documents_per_s"] = len(successful) / window_s if window_s > 0 else None
        summary["input_mib_per_s"] = (
            sum(result.document_bytes for result in successful) / (1024 ** 2) / window_s
            if window_s > 0 else None
        )
    return summary


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_json(client: httpx.Client, endpoint: str) -> dict:
    with client.stream("GET", endpoint) as response:
        response.raise_for_status()
        data = json.loads(read_body(response))
    if not isinstance(data, dict):
        raise ResponseError(f"Expected JSON object for {endpoint}.")
    return data


def inventory(client: httpx.Client) -> set[str]:
    data = get_json(client, "documents")
    metadata = data.get("metadata")
    if data.get("status") != "Success" or not isinstance(metadata, dict):
        raise ResponseError("Invalid document inventory response.")
    names = metadata.get("documents")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ResponseError("Invalid document names.")
    return set(names)


def prepare_corpus(client: httpx.Client, workload: Workload, reset: bool) -> bool:
    existing = inventory(client)
    if reset and existing:
        response = client.delete("documents", params={"delete_all": "true"})
        response.raise_for_status()
        if response.status_code != 204 or inventory(client):
            raise ResponseError("Document reset was not confirmed.")
        existing = set()
    if existing:
        raise ResponseError(
            "The vector store is not empty. Use a dedicated deployment "
            "or keep --reset-documents enabled to clear existing embeddings."
        )
    return True


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be greater than zero.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be zero or greater.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and greater than zero.")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, choices=("chat", "documents"))
    parser.add_argument("--base-url", default="http://localhost:8102/v1/chatqna")
    parser.add_argument("--stream", choices=("true", "false", "both"), default=None)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--timeout", type=positive_float, default=300, help="HTTP operation/inactivity timeout, seconds.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--questions-file", type=Path, default=ROOT / "questions.txt",
        help="UTF-8 .txt prompts file; separate prompts with blank lines (chat only).",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--reset-documents", action=argparse.BooleanOptionalAction, default=True,
        help="DELETE ALL existing document embeddings before the run (on by default).",
    )
    parser.add_argument(
        "--collect-resource-metrics", action="store_true", default=False,
        help="Collect CPU/GPU/memory utilization during the measured phase only.",
    )
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
        or url.query or url.fragment
    ):
        parser.error("--base-url must be an HTTP(S) URL without credentials, query, or fragment.")
    if args.api == "documents" and args.stream is not None:
        parser.error("--stream applies only to --api chat.")
    return args


def _docker(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess | None:
    """Run a docker subcommand, returning the completed process or None on failure."""
    try:
        return subprocess.run(
            ["docker", *command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def find_chatqna_container() -> str | None:
    """Locate the running ChatQnA Core backend container, if any."""
    listing = _docker(["ps", "--format", "{{.Names}}\t{{.Image}}"])
    if listing is None:
        return None
    candidates = []
    for line in listing.stdout.splitlines():
        name, _, image = line.partition("\t")
        repo = image.split("@", 1)[0].rsplit(":", 1)[0].rsplit("/", 1)[-1]
        if name in CHATQNA_CONTAINER_NAMES or repo == "chatqna":
            candidates.append(name)
    for preferred in CHATQNA_CONTAINER_NAMES:
        if preferred in candidates:
            return preferred
    return candidates[0] if candidates else None


def container_image_tag(container: str) -> str | None:
    """Return the deployed image tag (e.g. core_gpu_2026.2.0)."""
    inspected = _docker(["inspect", "--format", "{{.Config.Image}}", container])
    if inspected is None:
        return None
    image = inspected.stdout.strip()
    if not image:
        return None
    reference = image.split("@", 1)[0]
    return reference.rsplit(":", 1)[-1] if ":" in reference else reference


def container_model_config(container: str) -> dict | None:
    """Read the effective model/device configuration from the running container."""
    result = _docker(["exec", container, "python", "-c", _CONTAINER_CONFIG_SNIPPET])
    if result is None:
        return None
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def collect_configuration(output: Path, container: str | None) -> dict:
    """Collect deployment details and persist them as configuration.yaml."""
    configuration: dict[str, object] = {}
    if container is not None:
        tag = container_image_tag(container)
        model_config = container_model_config(container)
        configuration["container"] = container
        if tag is not None:
            configuration["tag"] = tag
        if model_config is not None:
            configuration["models"] = {
                "llm_model": model_config.get("llm_model"),
                "embedding_model": model_config.get("embedding_model"),
                "reranker_model": model_config.get("reranker_model"),
                "model_runtime": model_config.get("model_runtime"),
            }
            configuration["devices"] = {
                "llm_device": model_config.get("llm_device"),
                "embedding_device": model_config.get("embedding_device"),
                "reranker_device": model_config.get("reranker_device"),
            }
    else:
        print(
            "WARNING - No running ChatQnA Core backend container found; "
            "no deployment configuration will be recorded.",
            file=sys.stderr,
        )
    _write_yaml(output / "configuration.yaml", configuration)
    return configuration


def _write_yaml(path: Path, value: object) -> None:
    """Serialize a small, flat configuration mapping as YAML without extra dependencies."""
    lines: list[str] = []

    def emit(data: object, indent: int) -> None:
        pad = "  " * indent
        if isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    emit(item, indent + 1)
                else:
                    lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    first = True
                    for key, sub in item.items():
                        prefix = f"{pad}- " if first else f"{pad}  "
                        lines.append(f"{prefix}{key}: {_yaml_scalar(sub)}")
                        first = False
                else:
                    lines.append(f"{pad}- {_yaml_scalar(item)}")

    emit(value, 0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or re.search(r"[:#\n]|^\s|\s$", text):
        return json.dumps(text, ensure_ascii=False)
    return text


def fetch_container_tokenizer(container: str, llm_model_id: str, destination: Path) -> Path | None:
    """Copy the deployed LLM tokenizer.json out of the running container."""
    source = f"{container}:{CONTAINER_MODEL_CACHE}/{llm_model_id}/tokenizer.json"
    copied = _docker(["cp", source, str(destination)])
    if copied is None or not destination.is_file():
        return None
    return destination


def tokenize_run(
    output: Path, container: str | None, llm_model_id: str | None
) -> dict | None:
    """Retokenize saved requests using the deployed tokenizer and return token stats.

    This runs after measurement completes, never during it, and is best-effort:
    if the deployed tokenizer cannot be obtained it is skipped with a warning.
    """
    if container is None or not llm_model_id:
        print(
            "WARNING - Skipping tokenization; no running container or LLM model id available.",
            file=sys.stderr,
        )
        return None
    try:
        from postprocess import postprocess
    except ImportError as error:
        print(f"WARNING - Skipping tokenization; {error}.", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory() as workdir:
        tokenizer_path = fetch_container_tokenizer(
            container, llm_model_id, Path(workdir) / "tokenizer.json"
        )
        if tokenizer_path is None:
            print(
                "WARNING - Skipping tokenization; deployed tokenizer.json was not found "
                f"in {container}:{CONTAINER_MODEL_CACHE}/{llm_model_id}/.",
                file=sys.stderr,
            )
            return None
        try:
            postprocess(output, tokenizer_path, llm_model_id, "deployed")
        except (OSError, ValueError, ImportError) as error:
            print(f"WARNING - Tokenization failed ({type(error).__name__}): {error}.", file=sys.stderr)
            return None
    metadata_path = output / "tokenization.json"
    if not metadata_path.is_file():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def merge_token_kpis(summaries: dict[str, object], tokenization: dict | None) -> None:
    """Fold retokenized input/answer token statistics into each chat mode summary."""
    if not tokenization:
        return
    counts = tokenization.get("successful_token_counts")
    if not isinstance(counts, dict):
        return
    field_map = {
        "submitted_input_retokenized_tokens": "successful_input_tokens",
        "answer_retokenized_tokens": "successful_answer_tokens",
    }
    for key, stats in counts.items():
        mode, _, field_name = key.partition(".")
        summary_key = "chat-stream" if mode == "stream" else "chat-json"
        summary = summaries.get(summary_key)
        if not isinstance(summary, dict) or field_name not in field_map:
            continue
        summary[field_map[field_name]] = stats


def _start_resource_metrics(output: Path) -> tuple[str | None, str | None]:
    """Start the Metrics Manager collector after warm-up; returns (container, log_dir)."""
    try:
        from resource_collection import start_perf_tool
    except ImportError as error:
        print(f"WARNING - Resource metrics collection unavailable; {error}.", file=sys.stderr)
        return None, None
    image = os.environ.get("METRICS_MANAGER_IMAGE", "")
    try:
        log_dir, container = start_perf_tool(image, str(output / "resource_metrics"))
    except Exception as error:  # noqa: BLE001 - best-effort, never fail the benchmark
        print(f"WARNING - Could not start resource metrics collection ({type(error).__name__}): {error}.", file=sys.stderr)
        return None, None
    return container, log_dir


def _stop_resource_metrics(container: str | None, log_dir: str | None) -> None:
    """Stop the collector and render the utilization summary and graph."""
    if container is None or log_dir is None:
        return
    try:
        from resource_collection import plot_graphs, stop_perf_tool
    except ImportError as error:
        print(f"WARNING - Could not finalize resource metrics; {error}.", file=sys.stderr)
        return
    try:
        stop_perf_tool(container, log_dir)
        plot_graphs(log_dir)
    except Exception as error:  # noqa: BLE001 - best-effort, never fail the benchmark
        print(f"WARNING - Could not finalize resource metrics ({type(error).__name__}): {error}.", file=sys.stderr)


def run(args: argparse.Namespace) -> int:
    workload = load_workload(
        args.data_dir, args.questions_file if args.api == "chat" else None
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.reports_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "state": "preparing",
        "base_url": args.base_url.rstrip("/"),
        "api": args.api,
        "stream": (args.stream or "true") if args.api == "chat" else None,
        "workload_sha256": workload.fingerprint,
        "questions_file": args.questions_file.name if args.api == "chat" else None,
        "questions": len(workload.questions),
        "documents": len(workload.documents),
        "corpus_bytes": sum(len(doc.content) for doc in workload.documents),
        "warmup_requests_per_chat_mode": args.warmup if args.api == "chat" else 0,
        "http_operation_timeout_s": args.timeout,
        "reset_documents": args.reset_documents,
        "collect_resource_metrics": args.collect_resource_metrics,
        "python_version": platform.python_version(),
        "httpx_version": httpx.__version__,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("benchmark.py",)
        },
    }
    write_json(output / "manifest.json", manifest)
    all_results: list[Result] = []
    summaries: dict[str, object] = {}
    container = find_chatqna_container()
    resource_log_dir: str | None = None
    resource_container: str | None = None
    try:
        # Bypass environment HTTP proxies for the user-selected local API.
        with httpx.Client(
            base_url=args.base_url.rstrip("/") + "/",
            timeout=args.timeout,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            health = get_json(client, "health")
            if health.get("status") != "Success":
                raise ResponseError("Application health check failed.")
            model = get_json(client, "model")
            if model.get("status") != "Success" or not isinstance(model.get("llm_model"), str):
                raise ResponseError("Invalid model metadata.")
            manifest["model"] = model["llm_model"]
            needs_upload = prepare_corpus(client, workload, args.reset_documents)
            if args.api == "chat" and needs_upload:
                setup_results = [measure_document(client, document) for document in workload.documents]
                write_json(output / "setup-ingestion.json", [asdict(result) for result in setup_results])
                if any(result.outcome != "success" for result in setup_results):
                    raise ResponseError("Corpus ingestion failed; see setup-ingestion.json.")
            if args.api == "chat" and inventory(client) != {doc.name for doc in workload.documents}:
                raise ResponseError("Prepared corpus inventory does not match.")
            modes: list[bool | None] = (
                [None] if args.api == "documents"
                else [True, False] if args.stream == "both"
                else [args.stream != "false"]
            )
            # Warm up every chat mode first so resource metrics exclude warm-ups.
            for stream in modes:
                if stream is None:
                    continue
                key = "chat-stream" if stream else "chat-json"
                warmups = [
                    measure_chat(client, workload.questions[index % len(workload.questions)], stream)
                    for index in range(args.warmup)
                ]
                write_json(output / f"{key}-warmup.json", [asdict(result) for result in warmups])
                if any(result.outcome != "success" for result in warmups):
                    raise ResponseError(f"{key} warm-up failed; measured phase not started.")
            # Resource collection starts after warm-up and stops when measurement ends.
            if args.collect_resource_metrics:
                resource_container, resource_log_dir = _start_resource_metrics(output)
            manifest["state"] = "measuring"
            write_json(output / "manifest.json", manifest)
            for stream in modes:
                key = "documents" if stream is None else "chat-stream" if stream else "chat-json"
                results: list[Result] = []
                window_start = time.perf_counter_ns()
                if stream is None:
                    for document in workload.documents:
                        result = measure_document(client, document)
                        results.append(result)
                        all_results.append(result)
                else:
                    for question in workload.questions:
                        result = measure_chat(client, question, stream)
                        results.append(result)
                        all_results.append(result)
                window_s = (time.perf_counter_ns() - window_start) / 1_000_000_000
                summaries[key] = summarize(results, window_s)
            manifest["state"] = "completed"
    except (httpx.HTTPError, ValueError, OSError, KeyboardInterrupt) as error:
        manifest["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error_category"] = type(error).__name__
        raise
    finally:
        if args.collect_resource_metrics:
            _stop_resource_metrics(resource_container, resource_log_dir)
        write_json(output / "manifest.json", manifest)
        with (output / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for result in all_results:
                handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        if args.api == "chat" and all_results:
            tokenization = tokenize_run(output, container, manifest.get("model"))
            merge_token_kpis(summaries, tokenization)
        collect_configuration(output, container)
        summaries["corpus_documents"] = [
            {"name": document.name, "size_bytes": len(document.content)}
            for document in workload.documents
        ]
        write_json(output / "summary.json", summaries)
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mode", "metric", "statistic", "value"])
            for mode, summary in summaries.items():
                if isinstance(summary, dict):
                    for metric, value in summary.items():
                        if isinstance(value, dict):
                            if "mean" in value:
                                if value["mean"] is not None:
                                    writer.writerow([mode, metric, "avg", value["mean"]])
                            elif value.get("p90") is not None:
                                writer.writerow([mode, metric, "p90", value["p90"]])
                        else:
                            writer.writerow([mode, metric, "", value])
                elif isinstance(summary, list):
                    for item in summary:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name", "")
                        for statistic, scalar in item.items():
                            if statistic == "name":
                                continue
                            if statistic == "size_bytes":
                                writer.writerow(
                                    [mode, name, "size_mb", round(scalar / (1024 * 1024), 4)]
                                )
                            else:
                                writer.writerow([mode, name, statistic, scalar])
        print(f"Reports: {output}", file=sys.stderr)
    return 1 if any(result.outcome != "success" for result in all_results) else 0


def main() -> int:
    args = arguments()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130
    except (httpx.HTTPError, ValueError, OSError) as error:
        print(f"Benchmark failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

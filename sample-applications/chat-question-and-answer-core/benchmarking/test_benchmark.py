# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for local workloads and API benchmarking."""

import argparse
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

import benchmark
from benchmark import (
    ChatStream, Document, Question, ResponseError, Result, Workload, distribution,
    load_workload, measure_chat, measure_document, prepare_corpus, summarize,
)
from postprocess import postprocess


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail

    def __iter__(self):
        yield from self.chunks
        if self.fail:
            raise httpx.ReadError("Synthetic interrupted response")


SHARED = "Synthetic shared note: every station roof is blue.\n"


class WorkloadTests(unittest.TestCase):
    def test_reads_documents_and_blank_line_separated_prompts_reproducibly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            (data_dir / "manuals").mkdir(parents=True)
            (data_dir / "manuals" / "station.txt").write_text(SHARED, encoding="utf-8")
            (data_dir / "guide.pdf").write_bytes(b"synthetic PDF bytes")
            (data_dir / ".hidden.txt").write_text("ignored", encoding="utf-8")
            (data_dir / "ignored.csv").write_text("ignored", encoding="utf-8")
            questions_file = root / "questions.txt"
            questions_file.write_text(
                "  What color is the roof?\n\nDescribe the station\nusing its manual.\n",
                encoding="utf-8",
            )

            workload = load_workload(data_dir, questions_file)

            self.assertEqual([document.name for document in workload.documents], ["guide.pdf", "station.txt"])
            self.assertEqual(
                [question.input for question in workload.questions],
                ["What color is the roof?", "Describe the station\nusing its manual."],
            )
            self.assertEqual([question.question_id for question in workload.questions], ["q0001", "q0002"])
            self.assertEqual(
                [question.documents for question in workload.questions],
                [["guide.pdf", "station.txt"], ["guide.pdf", "station.txt"]],
            )
            self.assertNotIn("questions.txt", {document.name for document in workload.documents})
            self.assertEqual(workload, load_workload(data_dir, questions_file))

    def test_invalid_or_ambiguous_workloads_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            questions_file = root / "questions.txt"
            questions_file.write_text("Question?", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No supported documents"):
                load_workload(data_dir, questions_file)

            (data_dir / "one.txt").write_text("one", encoding="utf-8")
            (data_dir / "nested").mkdir()
            (data_dir / "nested" / "one.txt").write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "filenames must be unique"):
                load_workload(data_dir, questions_file)

            (data_dir / "nested" / "one.txt").unlink()
            questions_file.write_text(" \n\n ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No prompts"):
                load_workload(data_dir, questions_file)

            unsupported_questions_file = root / "questions.csv"
            unsupported_questions_file.write_text("Question?", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a .txt"):
                load_workload(data_dir, unsupported_questions_file)


class StreamTests(unittest.TestCase):
    def test_fragmentation_preserves_whitespace_and_original_arrival(self):
        parser = ChatStream()
        frames = parser.feed("da", 10)
        frames += parser.feed("ta: Hello\n\n", 20)
        frames += parser.feed("data: \n\n\n\n", 30)
        frames += parser.feed("data:  world\n\n", 40)
        frames += parser.finish()
        self.assertEqual("".join(text for text, _ in frames), "Hello\n\n world")
        self.assertEqual([arrival for _, arrival in frames], [20, None, 40])

    def test_raw_paragraphs_inside_one_payload(self):
        parser = ChatStream()
        frames = parser.feed("data: first\n\nsecond\n\n", 100)
        frames += parser.finish()
        self.assertEqual(frames, [("first\n\nsecond", 100)])

    def test_bad_and_unterminated_frames_fail(self):
        with self.assertRaises(ResponseError):
            ChatStream().feed('{"error": "wrong protocol"}', 1)
        parser = ChatStream()
        parser.feed("data: incomplete", 1)
        with self.assertRaises(ResponseError):
            parser.finish()
        self.assertEqual(parser.partial(), "incomplete")


class MeasurementTests(unittest.TestCase):
    question = Question("q1", ["doc.txt"], "Question?")

    def client(self, response: httpx.Response) -> httpx.Client:
        return httpx.Client(
            base_url="http://example.test/v1/chatqna/",
            transport=httpx.MockTransport(lambda request: response),
        )

    def test_stream_unicode_and_timing(self):
        encoded = "data: Hi 🌍\n\ndata: !\n\n".encode()
        split = encoded.index("🌍".encode()) + 1
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=Chunks([encoded[:split], encoded[split:]]),
        )
        ticks = iter([0, 10_000_000, 20_000_000, 30_000_000, 40_000_000])
        with self.client(response) as client:
            result = measure_chat(client, self.question, True, lambda: next(ticks))
        self.assertEqual(result.outcome, "success")
        self.assertEqual(result.answer, "Hi 🌍!")
        self.assertEqual(result.answer_char_count, 5)
        self.assertEqual(result.first_visible_text_ms, 10)
        self.assertEqual(result.inter_chunk_gaps_ms, [10])
        self.assertEqual(result.elapsed_ms, 40)

    def test_json_measures_only_metadata(self):
        with self.client(httpx.Response(200, json={"status": "Success", "metadata": "Hello!"})) as client:
            result = measure_chat(client, self.question, False)
        self.assertEqual(result.answer_char_count, 6)
        self.assertIsNone(result.first_visible_text_ms)
        self.assertIsNone(result.max_inter_chunk_gap_ms)

    def test_empty_wrong_shape_http_and_invalid_utf8_fail(self):
        responses = [
            httpx.Response(200, json={"status": "Success", "metadata": ""}),
            httpx.Response(200, json={"status": "Failure", "metadata": "x"}),
            httpx.Response(200, json={"status": "Success", "metadata": {}}),
            httpx.Response(500, text="server detail must not enter reports"),
        ]
        for response in responses:
            with self.subTest(response=response), self.client(response) as client:
                result = measure_chat(client, self.question, False)
                self.assertEqual(result.outcome, "failure")
                self.assertNotIn("server detail", json.dumps(asdict(result)))
        with self.client(httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=Chunks([b"data: \xff\n\n"]),
        )) as client:
            self.assertEqual(measure_chat(client, self.question, True).error_category, "UnicodeDecodeError")

    def test_partial_output_is_retained_on_transport_failure(self):
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=Chunks([b"data: partial"], fail=True),
        )
        with self.client(response) as client:
            result = measure_chat(client, self.question, True)
        self.assertEqual(result.outcome, "failure")
        self.assertEqual(result.answer, "partial")
        self.assertIsNotNone(result.first_visible_text_ms)
        self.assertTrue(result.output_is_partial)

    def test_response_limit_and_timeout_fail_explicitly(self):
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            stream=Chunks([b"data: too long\n\n"]),
        )
        with self.client(response) as client, patch("benchmark.MAX_RESPONSE_BYTES", 4):
            self.assertEqual(measure_chat(client, self.question, True).error_category, "ResponseError")
        def timeout(request):
            raise httpx.ReadTimeout("synthetic")
        with httpx.Client(
            base_url="http://example.test/", transport=httpx.MockTransport(timeout)
        ) as client:
            self.assertEqual(measure_chat(client, self.question, False).error_category, "ReadTimeout")

    def test_ingestion_requires_filename_confirmation(self):
        document = Document("doc.txt", b"Synthetic content")
        with self.client(httpx.Response(200, json={
            "status": "Success", "metadata": {"documents": ["doc.txt"]}
        })) as client:
            result = measure_document(client, document)
        self.assertEqual(result.outcome, "success")
        self.assertEqual(result.document_bytes, len(document.content))
        with self.client(httpx.Response(200, json={
            "status": "Success", "metadata": {"documents": None}
        })) as client:
            self.assertEqual(measure_document(client, document).outcome, "failure")

    def test_statistics_count_failures_and_single_chunk_gap_handling(self):
        results = [
            Result("1", "chat", True, elapsed_ms=100, first_visible_text_ms=10,
                   answer_char_count=5, max_inter_chunk_gap_ms=40),
            Result("2", "chat", True, elapsed_ms=200, first_visible_text_ms=20,
                   answer_char_count=10),
            Result("3", "chat", True, outcome="failure", error_category="ReadTimeout", elapsed_ms=1000),
        ]
        summary = summarize(results, 2)
        self.assertEqual(summary["sequential_successful_requests_per_s"], 1)
        self.assertAlmostEqual(summary["failure_rate_pct"], 100 / 3)
        self.assertEqual(distribution([100, 200])["p95"], 195)
        self.assertIsNone(distribution([])["p50"])


class WorkflowTests(unittest.TestCase):
    def test_both_modes_run_and_reports_have_reproducibility_metadata(self):
        self.check_workflow("chat", "both")

    def test_document_ingestion_reports_rates(self):
        self.check_workflow("documents", None)

    def test_resource_metrics_bracket_the_measured_phase_only(self):
        with patch("benchmark._start_resource_metrics", return_value=("c", "log")) as start, \
             patch("benchmark._stop_resource_metrics") as stop:
            self.check_workflow("chat", "true", collect_resource_metrics=True)
        start.assert_called_once()
        stop.assert_called_once_with("c", "log")

    def check_workflow(self, api: str, stream: str | None, collect_resource_metrics: bool = False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "manual.txt").write_text(SHARED, encoding="utf-8")
            questions_file = root / "questions.txt"
            questions_file.write_text("What color is the roof?\n\nWhat is in the manual?\n", encoding="utf-8")
            expected = load_workload(data_dir, questions_file)
            names: set[str] = set()
            chat_requests: list[bool] = []
            def handler(request):
                path = request.url.path
                if path.endswith("/health"):
                    return httpx.Response(200, json={"status": "Success"})
                if path.endswith("/model"):
                    return httpx.Response(200, json={"status": "Success", "llm_model": "synthetic"})
                if path.endswith("/documents"):
                    if request.method == "GET":
                        return httpx.Response(200, json={
                            "status": "Success", "metadata": {"documents": sorted(names)}
                        })
                    body = request.read().decode()
                    name = body.split('filename="', 1)[1].split('"', 1)[0]
                    names.add(name)
                    return httpx.Response(200, json={
                        "status": "Success", "metadata": {"documents": [name]}
                    })
                data = json.loads(request.read())
                chat_requests.append(data["stream"])
                if data["stream"]:
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                          content=b"data: blue\n\n")
                return httpx.Response(200, json={"status": "Success", "metadata": "blue"})
            client = httpx.Client(base_url="http://example.test/v1/chatqna/",
                                 transport=httpx.MockTransport(handler))
            args = argparse.Namespace(
                data_dir=data_dir, questions_file=questions_file, reports_dir=root / "reports",
                api=api,
                stream=stream, base_url="http://example.test/v1/chatqna", warmup=1,
                timeout=10, reset_documents=True, collect_resource_metrics=collect_resource_metrics,
            )
            with patch("benchmark.httpx.Client", return_value=client), \
                 patch("benchmark.find_chatqna_container", return_value=None):
                self.assertEqual(benchmark.run(args), 0)
            run_dir = next((root / "reports").iterdir())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["state"], "completed")
            self.assertEqual(manifest["model"], "synthetic")
            requests = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["documents"], len(expected.documents))
            if api == "chat":
                expected_measured = len(expected.questions) * (2 if stream == "both" else 1)
            else:
                expected_measured = len(expected.documents)
            self.assertEqual(len(requests), expected_measured)
            self.assertEqual(names, {d.name for d in expected.documents})
            self.assertTrue((run_dir / "configuration.yaml").is_file())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(
                summary["corpus_documents"],
                [{"name": d.name, "size_bytes": len(d.content)} for d in expected.documents],
            )
            if api == "chat":
                if stream == "both":
                    self.assertEqual(chat_requests, [True, False, True, True, False, False])
                    self.assertEqual(set(summary), {"chat-stream", "chat-json", "corpus_documents"})
                else:
                    self.assertEqual(set(summary), {"chat-stream", "corpus_documents"})
            else:
                self.assertGreater(summary["documents"]["input_mib_per_s"], 0)
            rows = list(csv.reader(io.StringIO((run_dir / "summary.csv").read_text())))
            self.assertEqual(rows[0], ["mode", "metric", "statistic", "value"])
            self.assertIn(
                ["corpus_documents", expected.documents[0].name, "size_mb",
                 str(round(len(expected.documents[0].content) / (1024 * 1024), 4))],
                rows,
            )
            statistics_seen = {row[2] for row in rows[1:] if row[1].startswith("successful_")}
            self.assertFalse(statistics_seen & {"min", "p50", "p95", "max"})

    def test_existing_corpus_requires_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "manual.txt").write_text("Synthetic content", encoding="utf-8")
            workload = load_workload(data_dir, None)
            calls = []
            def handler(request):
                calls.append(request.method)
                return httpx.Response(200, json={
                    "status": "Success", "metadata": {"documents": ["private.txt"]}
                })
            with httpx.Client(base_url="http://example.test/",
                              transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ResponseError):
                    prepare_corpus(client, workload, False)
            self.assertEqual(calls, ["GET"])

    def test_reset_is_explicit_and_clears_existing_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "manual.txt").write_text("Synthetic content", encoding="utf-8")
            workload = load_workload(data_dir, None)
            existing = {"private.txt"}
            methods = []
            def handler(request):
                methods.append(request.method)
                if request.method == "DELETE":
                    self.assertEqual(request.url.params["delete_all"], "true")
                    existing.clear()
                    return httpx.Response(204)
                return httpx.Response(200, json={
                    "status": "Success", "metadata": {"documents": sorted(existing)}
                })
            with httpx.Client(base_url="http://example.test/",
                              transport=httpx.MockTransport(handler)) as client:
                self.assertTrue(prepare_corpus(client, workload, True))
                self.assertEqual(methods, ["GET", "DELETE", "GET"])

    def test_invalid_cli_combinations_are_rejected(self):
        cases = [
            ["--api", "documents", "--stream", "false"],
            ["--api", "chat", "--timeout", "nan"],
            ["--api", "chat", "--base-url", "http://user:password@example.test"],
        ]
        for values in cases:
            with self.subTest(arguments=values), \
                 patch("sys.argv", ["benchmark.py"] + values), \
                 patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as context:
                    benchmark.arguments()
                self.assertEqual(context.exception.code, 2)


class TokenizationTests(unittest.TestCase):
    def test_local_tokenizer_counts_without_padding_truncation_or_special_tokens(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel(
                {"[UNK]": 0, "hello": 1, "world": 2, "[CLS]": 3}, unk_token="[UNK]"
            ))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A", special_tokens=[("[CLS]", 3)]
            )
            tokenizer.enable_truncation(max_length=1)
            tokenizer.enable_padding(length=8)
            tokenizer.save(str(root / "tokenizer.json"))
            result = Result("1", "chat", False, input="hello world", answer="hello")
            (root / "requests.jsonl").write_text(json.dumps(asdict(result)) + "\n")
            postprocess(root, root / "tokenizer.json", "synthetic", "revision-1")
            output = json.loads((root / "tokenized-requests.jsonl").read_text())
            self.assertEqual(output["submitted_input_retokenized_tokens"], 2)
            self.assertEqual(output["answer_retokenized_tokens"], 1)
            self.assertNotIn("answer_retokenized_tokens", json.loads((root / "requests.jsonl").read_text()))
            metadata = json.loads((root / "tokenization.json").read_text())
            self.assertEqual(metadata["state"], "completed")
            self.assertFalse(metadata["add_special_tokens"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                postprocess(root, root / "tokenizer.json", "synthetic", "revision-1")


class DeploymentIntrospectionTests(unittest.TestCase):
    def test_configuration_includes_tag_models_and_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("benchmark.container_image_tag", return_value="core_gpu_2026.2.0"), \
                 patch("benchmark.container_model_config", return_value={
                     "model_runtime": "openvino",
                     "llm_model": "Intel/neural-chat-7b-v3-3",
                     "embedding_model": "BAAI/bge-small-en-v1.5",
                     "reranker_model": "BAAI/bge-reranker-base",
                     "llm_device": "GPU", "embedding_device": "GPU", "reranker_device": "GPU",
                 }):
                configuration = benchmark.collect_configuration(
                    output, "chatqna-core-ov-gpu"
                )
            self.assertEqual(configuration["tag"], "core_gpu_2026.2.0")
            self.assertEqual(configuration["models"]["llm_model"], "Intel/neural-chat-7b-v3-3")
            self.assertEqual(configuration["devices"]["llm_device"], "GPU")
            self.assertNotIn("documents", configuration)
            text = (output / "configuration.yaml").read_text()
            self.assertIn("tag: core_gpu_2026.2.0", text)
            self.assertIn("llm_device: GPU", text)
            self.assertNotIn("name:", text)

    def test_configuration_without_container_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            configuration = benchmark.collect_configuration(output, None)
        self.assertNotIn("tag", configuration)
        self.assertNotIn("documents", configuration)
        self.assertEqual(configuration, {})

    def test_merge_token_kpis_folds_counts_into_chat_summaries(self):
        summaries: dict[str, object] = {"chat-stream": {}, "chat-json": {}}
        tokenization = {"successful_token_counts": {
            "stream.submitted_input_retokenized_tokens": {"count": 2, "min": 1, "mean": 1.5, "max": 2},
            "stream.answer_retokenized_tokens": {"count": 2, "min": 3, "mean": 4.0, "max": 5},
            "json.submitted_input_retokenized_tokens": {"count": 2, "min": 1, "mean": 1.5, "max": 2},
        }}
        benchmark.merge_token_kpis(summaries, tokenization)
        self.assertEqual(summaries["chat-stream"]["successful_input_tokens"]["max"], 2)
        self.assertEqual(summaries["chat-stream"]["successful_answer_tokens"]["max"], 5)
        self.assertIn("successful_input_tokens", summaries["chat-json"])
        self.assertNotIn("successful_answer_tokens", summaries["chat-json"])

    def test_tokenize_run_uses_container_tokenizer_and_reports_counts(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tokenizer = Tokenizer(models.WordLevel(
                {"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"
            ))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            source = output / "deployed-tokenizer.json"
            tokenizer.save(str(source))
            result = Result("1", "chat", True, input="hello world", answer="hello")
            (output / "requests.jsonl").write_text(json.dumps(asdict(result)) + "\n")

            def fake_fetch(container, model_id, destination):
                destination.write_bytes(source.read_bytes())
                return destination

            with patch("benchmark.fetch_container_tokenizer", side_effect=fake_fetch):
                metadata = benchmark.tokenize_run(output, "chatqna-core-ov-cpu", "synthetic/model")
            self.assertEqual(metadata["state"], "completed")
            counts = metadata["successful_token_counts"]
            self.assertEqual(counts["stream.submitted_input_retokenized_tokens"]["max"], 2)
            self.assertTrue((output / "tokenized-requests.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()

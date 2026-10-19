# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Retokenize saved input/answer text offline, never during measurement."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def postprocess(
    run_directory: Path, tokenizer_path: Path, tokenizer_id: str, tokenizer_revision: str
) -> None:
    from tokenizers import Tokenizer

    output_path = run_directory / "tokenized-requests.jsonl"
    metadata_path = run_directory / "tokenization.json"
    if output_path.exists() or metadata_path.exists():
        raise ValueError("Tokenization output already exists; refusing to overwrite it.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counts: dict[str, list[int]] = defaultdict(list)
    with (run_directory / "requests.jsonl").open(encoding="utf-8") as source:
        with output_path.open("x", encoding="utf-8") as output:
            for line in source:
                result = json.loads(line)
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object per request.")
                if result.get("api") == "chat":
                    for text_field, token_field in (
                        ("input", "submitted_input_retokenized_tokens"),
                        ("answer", "answer_retokenized_tokens"),
                    ):
                        text = result.get(text_field)
                        if not isinstance(text, str):
                            raise ValueError(f"Expected string field: {text_field}.")
                        count = len(tokenizer.encode(text, add_special_tokens=False).ids)
                        result[token_field] = count
                        if result.get("outcome") == "success":
                            mode = "stream" if result.get("stream") else "json"
                            counts[f"{mode}.{token_field}"].append(count)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
    metadata = {
        "state": "completed",
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "add_special_tokens": False,
        "truncation": False,
        "padding": False,
        "input_scope": "Submitted question only; excludes RAG context and prompt template.",
        "answer_scope": "Retokenized visible text; not guaranteed to equal generated tokens.",
        "successful_token_counts": {
            key: {
                "count": len(values),
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }
            for key, values in counts.items()
        },
    }
    with metadata_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Matching local tokenizer.json.")
    parser.add_argument("--tokenizer-id", required=True, help="Deployed LLM tokenizer identity.")
    parser.add_argument("--tokenizer-revision", required=True, help="Exact deployed tokenizer revision.")
    args = parser.parse_args()
    try:
        postprocess(args.run_dir, args.tokenizer, args.tokenizer_id, args.tokenizer_revision)
    except (OSError, ValueError, ImportError) as error:
        print(f"Tokenization failed ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    print(f"Tokenization reports: {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

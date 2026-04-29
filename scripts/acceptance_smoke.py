#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run server-side embedding acceptance smoke tests.")
    parser.add_argument("--base-url", default=os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "qwen3-embedding-8b"))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models/Qwen3-Embedding-8B"))
    parser.add_argument("--expected-dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "4096")))
    parser.add_argument("--api-key", default=os.getenv("GATEWAY_API_KEY"))
    parser.add_argument("--qdrant", action="store_true", help="Run Qdrant demo after API smoke tests.")
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path("reports") / "acceptance" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    requests_path = run_dir / "requests.jsonl"
    vectors_meta: dict[str, Any] = {"vectors": [], "similarities": []}
    results: list[dict[str, Any]] = []

    write_env(run_dir / "env.txt", args)

    with httpx.Client(timeout=args.timeout) as client, requests_path.open("w", encoding="utf-8") as log:
        results.append(run_check("healthz", log, lambda: check_healthz(client, args)))
        results.append(run_check("readyz", log, lambda: check_readyz(client, args)))
        single_result = run_check(
            "single_embedding",
            log,
            lambda: check_embedding(
                client,
                args,
                "公司报销流程需要哪些材料？",
                expected_count=1,
            ),
        )
        results.append(single_result)
        record_vectors(vectors_meta, "single_embedding", single_result)

        batch_result = run_check(
            "batch_embedding",
            log,
            lambda: check_embedding(
                client,
                args,
                ["公司报销流程需要哪些材料？", "员工年假如何申请？"],
                expected_count=2,
            ),
        )
        results.append(batch_result)
        record_vectors(vectors_meta, "batch_embedding", batch_result)

        query_result = run_check(
            "query_instruction_embedding",
            log,
            lambda: check_embedding(
                client,
                args,
                "怎么申请年假？",
                expected_count=1,
                headers={"X-Embedding-Input-Type": "query"},
            ),
        )
        results.append(query_result)
        record_vectors(vectors_meta, "query_instruction_embedding", query_result)

    if args.qdrant:
        results.append(run_subprocess_check("qdrant_demo", ["python", "examples/qdrant_demo.py"]))

    if len(vectors_meta["vectors"]) >= 2:
        vectors_meta["similarities"].append(
            {
                "name": "sample_cosine",
                "value": cosine(
                    vectors_meta["vectors"][0]["sample"],
                    vectors_meta["vectors"][1]["sample"],
                ),
            }
        )

    (run_dir / "vectors_meta.json").write_text(
        json.dumps(vectors_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(1 for item in results if item["passed"])
    summary = {
        "passed": passed,
        "failed": len(results) - passed,
        "results": [{k: v for k, v in item.items() if k != "vectors"} for item in results],
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "model_path": args.model_path,
            "expected_dim": args.expected_dim,
            "api_key_configured": bool(args.api_key),
            "qdrant": args.qdrant,
        },
        "output_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


def run_check(name: str, log: Any, func: Any) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {"name": name, "passed": False}
    try:
        record.update(func())
        record["passed"] = True
    except Exception as exc:  # noqa: BLE001 - acceptance records need raw failure context.
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    record["latency_seconds"] = round(time.perf_counter() - started, 4)
    log.write(json.dumps(sanitize_record(record), ensure_ascii=False) + "\n")
    log.flush()
    return record


def run_subprocess_check(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "name": name,
        "passed": completed.returncode == 0,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def check_healthz(client: httpx.Client, args: argparse.Namespace) -> dict[str, Any]:
    response = client.get(f"{args.base_url}/healthz", headers=auth_headers(args))
    assert response.status_code == 200, response.text
    return {"status_code": response.status_code, "response": response.json()}


def check_readyz(client: httpx.Client, args: argparse.Namespace) -> dict[str, Any]:
    response = client.get(f"{args.base_url}/readyz", headers=auth_headers(args))
    assert response.status_code == 200, response.text
    return {"status_code": response.status_code, "response": response.json()}


def check_embedding(
    client: httpx.Client,
    args: argparse.Namespace,
    input_value: str | list[str],
    expected_count: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = auth_headers(args)
    request_headers.update(headers or {})
    response = client.post(
        f"{args.base_url}/v1/embeddings",
        json={"model": args.model, "input": input_value},
        headers=request_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    vectors = [item["embedding"] for item in body["data"]]
    assert len(vectors) == expected_count, body
    dims = [len(vector) for vector in vectors]
    assert all(dim == args.expected_dim for dim in dims), dims
    norms = [l2_norm(vector) for vector in vectors]
    return {
        "status_code": response.status_code,
        "count": len(vectors),
        "dimensions": dims,
        "norms": norms,
        "vectors": vectors,
    }


def record_vectors(vectors_meta: dict[str, Any], name: str, result: dict[str, Any]) -> None:
    for idx, vector in enumerate(result.get("vectors", [])):
        vectors_meta["vectors"].append(
            {
                "name": name,
                "index": idx,
                "dimension": len(vector),
                "norm": l2_norm(vector),
                "sample": vector[:16],
            }
        )


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(record)
    if "vectors" in sanitized:
        sanitized["vectors"] = [
            {"dimension": len(vector), "norm": l2_norm(vector), "sample": vector[:8]}
            for vector in sanitized["vectors"]
        ]
    return sanitized


def write_env(path: Path, args: argparse.Namespace) -> None:
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"base_url={args.base_url}",
        f"model={args.model}",
        f"model_path={args.model_path}",
        f"expected_dim={args.expected_dim}",
        f"api_key_configured={bool(args.api_key)}",
        "",
        "$ pip freeze",
        run_command(["python", "-m", "pip", "freeze"]),
        "",
        "$ nvidia-smi",
        run_command(["nvidia-smi"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return f"not available: {exc}"
    return (completed.stdout + completed.stderr)[-12000:]


def auth_headers(args: argparse.Namespace) -> dict[str, str]:
    if not args.api_key:
        return {}
    return {"Authorization": f"Bearer {args.api_key}"}


def l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine(left: list[float], right: list[float]) -> float | None:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0 or right_norm == 0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


if __name__ == "__main__":
    raise SystemExit(main())

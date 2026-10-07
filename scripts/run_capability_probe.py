#!/usr/bin/env python3
"""Run small deterministic capability probes against an LM Studio model."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def image_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def normalized_number_text(text: str) -> str:
    return (
        text.lower()
        .replace("’", "")
        .replace("'", "")
        .replace(" ", "")
        .replace(",", ".")
    )


def content_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def completion_record(response: Any, elapsed_s: float) -> dict[str, Any]:
    message = response.choices[0].message
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
        )
    usage = response.usage
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    return {
        "text": content_text(message),
        "tool_calls": tool_calls,
        "finish_reason": response.choices[0].finish_reason,
        "elapsed_s": round(elapsed_s, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": completion_tokens,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "completion_tokens_per_s": (
            round(completion_tokens / elapsed_s, 3)
            if completion_tokens is not None and elapsed_s > 0
            else None
        ),
    }


def invoke(client: OpenAI, **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # Preserve unsupported modality/runtime paths as evidence.
        return {
            "text": "",
            "tool_calls": [],
            "finish_reason": None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "completion_tokens_per_s": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    record = completion_record(response, time.perf_counter() - started)
    record["error"] = None
    return record


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = OpenAI(api_key=os.environ[args.token_env], base_url=args.base_url)
    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    cases: list[dict[str, Any]] = []

    classification = invoke(
        client,
        **common,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify each Polish document. Return only JSON: "
                    '{"labels":["label1","label2","label3"]}. Allowed labels: '
                    "raport_kwartalny, espi_biezacy, inne."
                ),
            },
            {
                "role": "user",
                "content": (
                    "1. Skonsolidowane wyniki Grupy za III kwartał 2026: przychody, EBITDA, bilans.\n"
                    "2. Raport bieżący ESPI 41/2026: powołanie członka zarządu.\n"
                    "3. Komentarz makroekonomiczny o stopach NBP i inflacji, bez danych jednej spółki."
                ),
            },
        ],
    )
    parsed = parse_json_object(classification["text"])
    classification["score"] = int(
        parsed is not None
        and parsed.get("labels") == ["raport_kwartalny", "espi_biezacy", "inne"]
    )
    cases.append({"id": "classification_pl", **classification})

    abstention = invoke(
        client,
        **common,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use only the supplied source. If the requested fact is absent, "
                    "answer exactly INSUFFICIENT_SOURCE and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    "SOURCE: In Q2 the company sold 100 units and reported revenue of PLN 2.0m. "
                    "QUESTION: What was its EBITDA margin in Q2?"
                ),
            },
        ],
    )
    abstention["score"] = int(abstention["text"].strip() == "INSUFFICIENT_SOURCE")
    cases.append({"id": "grounded_abstention", **abstention})

    tool = invoke(
        client,
        **common,
        messages=[
            {
                "role": "system",
                "content": "Use the provided tool for the calculation. Do not answer in prose.",
            },
            {
                "role": "user",
                "content": "Calculate the portfolio weight of a PLN 25,000 position in a PLN 100,000 portfolio.",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate_position_weight",
                    "description": "Calculate a position's share of a portfolio.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "position_value": {"type": "number"},
                            "portfolio_value": {"type": "number"},
                        },
                        "required": ["position_value", "portfolio_value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    tool_ok = False
    if len(tool["tool_calls"]) == 1:
        call = tool["tool_calls"][0]
        try:
            call_args = json.loads(call["arguments"])
        except json.JSONDecodeError:
            call_args = None
        tool_ok = (
            call["name"] == "calculate_position_weight"
            and call_args == {"position_value": 25000, "portfolio_value": 100000}
        )
    tool["score"] = int(tool_ok)
    cases.append({"id": "tool_selection", **tool})

    ocr = invoke(
        client,
        **common,
        messages=[
            {
                "role": "system",
                "content": "Read the attached invoice image. Return only facts visible in the pixels.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract bill number, subtotal before VAT, VAT amount, final bill amount, "
                            "payer name, and payee company."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url(args.ocr_image)}},
                ],
            },
        ],
    )
    normalized = normalized_number_text(ocr["text"])
    ocr_expected = ["3139", "3667.35", "282.40", "3949.75", "piarutschmann", "robertschneider"]
    ocr["matched"] = [item for item in ocr_expected if item in normalized]
    ocr["score"] = len(ocr["matched"])
    ocr["score_max"] = len(ocr_expected)
    cases.append({"id": "invoice_ocr", **ocr})

    chart = invoke(
        client,
        **common,
        messages=[
            {
                "role": "system",
                "content": "Read the chart from pixels. Return only JSON with keys company_return, benchmark_return, winner.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare the two cumulative returns shown at the right edge of the chart."},
                    {"type": "image_url", "image_url": {"url": image_url(args.chart_image)}},
                ],
            },
        ],
    )
    chart_json = parse_json_object(chart["text"])
    chart_text = normalized_number_text(chart["text"])
    winner = str(chart_json.get("winner", "")).lower() if chart_json else ""
    chart["score"] = int(
        "103" in chart_text
        and "58" in chart_text
        and chart_json is not None
        and ("two" in winner or winner == "company")
    )
    cases.append({"id": "chart_reasoning", **chart})

    scored = [case for case in cases if "score" in case]
    return {
        "schema_version": 1,
        "model": args.model,
        "base_url": args.base_url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_tokens": args.max_tokens,
        "cases": cases,
        "summary": {
            "binary_passes": sum(
                case["score"] for case in scored if case["id"] != "invoice_ocr"
            ),
            "binary_total": sum(1 for case in scored if case["id"] != "invoice_ocr"),
            "ocr_fields": next(case["score"] for case in cases if case["id"] == "invoice_ocr"),
            "ocr_fields_total": next(
                case["score_max"] for case in cases if case["id"] == "invoice_ocr"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ocr-image", required=True, type=Path)
    parser.add_argument("--chart-image", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--token-env", default="LM_STUDIO_TOKEN")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

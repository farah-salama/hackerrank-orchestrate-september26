"""Buy or Wait? CLI: load data, forecast, ask Gemini, validate, write output.csv."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv  # pylint: disable=import-error

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gemini_agent import build_models, call_gemini, extract_image_amount
from loader import load_dataset
from preprocessor import build_context
from validator import validate


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

FLASH_INPUT_COST = 0.15 / 1_000_000
FLASH_OUTPUT_COST = 0.60 / 1_000_000
PRO_INPUT_COST = 1.25 / 1_000_000
PRO_OUTPUT_COST = 10.00 / 1_000_000


def main() -> int:
    _load_env()
    args = _parse_args()
    dataset = load_dataset(args.dataset)
    requests = dataset.sample_requests if args.sample else dataset.requests
    if not requests:
        raise SystemExit("No requests found in the selected dataset.")

    flash_model, pro_model, vision_model = build_models()
    usage_log: list[dict[str, Any]] = []
    request_ids = {row["request_id"] for row in requests}
    _run_vision_prepass(dataset, request_ids, vision_model, usage_log)

    rows: list[dict[str, Any]] = []
    if args.workers > 1:
        rows = _run_parallel(
            dataset,
            requests,
            flash_model,
            pro_model,
            usage_log,
            args.workers,
        )
    else:
        for index, request in enumerate(requests, start=1):
            rows.append(
                _process_request(dataset, request, flash_model, pro_model, usage_log)
            )
            print(
                f"[{index}/{len(requests)}] {request['request_id']}",
                flush=True,
            )
            if index < len(requests) and args.sleep > 0:
                time.sleep(args.sleep)

    rows.sort(key=lambda row: row["request_id"])
    _write_output(args.output, rows)
    report_path = HERE / "evaluation" / "usage_report.md"
    _write_usage_report(report_path, usage_log, len(requests), args.sample)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote usage report to {report_path}")
    return 0


def _process_request(
    dataset: Any,
    request: dict[str, Any],
    flash_model: Any,
    pro_model: Any,
    usage_log: list[dict[str, Any]],
) -> dict[str, Any]:
    user_id = request["user_id"]
    request_id = request["request_id"]
    profile = dataset.profiles[user_id]
    events = dataset.events_by_user.get(user_id, [])
    options = dataset.options_by_request.get(request_id, [])
    messages = dataset.messages_by_user.get(user_id, [])
    context = build_context(
        request=request,
        profile=profile,
        events=events,
        payment_options=options,
        messages=messages,
        fx_lookup=dataset.fx,
    )
    raw = call_gemini(context, flash_model, pro_model, usage_log)
    result = validate(raw, context)
    result["request_id"] = request_id
    return result


def _run_parallel(
    dataset: Any,
    requests: list[dict[str, Any]],
    flash_model: Any,
    pro_model: Any,
    usage_log: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_request,
                dataset,
                request,
                flash_model,
                pro_model,
                usage_log,
            ): request["request_id"]
            for request in requests
        }
        for index, future in enumerate(as_completed(futures), start=1):
            request_id = futures[future]
            rows.append(future.result())
            print(f"[{index}/{len(requests)}] {request_id}", flush=True)
    return rows


def _run_vision_prepass(
    dataset: Any,
    request_ids: set[str],
    vision_model: Any,
    usage_log: list[dict[str, Any]],
) -> None:
    pairs = dataset.missing_image_amounts(request_ids)
    if not pairs:
        return
    print(f"Vision pre-pass: {len(pairs)} image(s)", flush=True)
    image_dir = dataset.dataset_dir / "media" / "images"
    for image_id, event_id in pairs:
        image_path = image_dir / f"{image_id}.png"
        if not image_path.exists():
            print(f"  missing image file {image_path}", flush=True)
            continue
        request_id = next(
            (
                image["request_id"]
                for image in dataset.images
                if image["image_id"] == image_id
            ),
            "",
        )
        amount, currency = extract_image_amount(
            str(image_path),
            vision_model,
            usage_log,
            request_id=request_id,
        )
        if amount is None:
            print(f"  {image_id}: no amount extracted", flush=True)
            continue
        dataset.patch_event_amount(event_id, amount, currency)
        print(f"  {image_id} -> {event_id} = {amount} {currency or ''}", flush=True)


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "request_id": row["request_id"],
                    "amount_safe_to_pay": _csv_amount(row.get("amount_safe_to_pay")),
                    "affordability_status": row.get("affordability_status", ""),
                    "recommended_payment_method": row.get("recommended_payment_method", ""),
                    "payment_plan": row.get("payment_plan") or "none",
                    "earliest_date_for_full_payment": row.get("earliest_date_for_full_payment") or "",
                    "spending_changes_needed": row.get("spending_changes_needed") or "none",
                    "decision_explanation": row.get("decision_explanation") or "",
                }
            )


def _write_usage_report(
    path: Path,
    usage_log: list[dict[str, Any]],
    request_count: int,
    sample: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    )
    total_in = 0
    total_out = 0
    total_cost = 0.0
    for entry in usage_log:
        model = entry.get("model") or "unknown"
        inp = int(entry.get("input_tokens") or 0)
        out = int(entry.get("output_tokens") or 0)
        cost = _estimate_cost(model, inp, out)
        stats = by_model[model]
        stats["calls"] += 1
        stats["input_tokens"] += inp
        stats["output_tokens"] += out
        stats["cost"] += cost
        total_in += inp
        total_out += out
        total_cost += cost

    total_tokens = total_in + total_out
    denom = max(1, request_count)
    mode = "sample" if sample else "full dataset"
    lines = [
        "# Token usage report",
        "",
        f"Run type: {mode}",
        f"Requests processed: {request_count}",
        "Provider: Google",
        "",
        "## Models",
        "",
        "| Model | Calls | Input tokens | Output tokens | Estimated cost (USD) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, stats in sorted(by_model.items()):
        lines.append(
            f"| {model} | {int(stats['calls'])} | {int(stats['input_tokens'])} | "
            f"{int(stats['output_tokens'])} | {stats['cost']:.6f} |"
        )
    if not by_model:
        lines.append("| none | 0 | 0 | 0 | 0.000000 |")
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Model calls: {len(usage_log)}",
            f"- Input tokens: {total_in}",
            f"- Output tokens: {total_out}",
            f"- Total tokens: {total_tokens}",
            f"- Average tokens per request: {total_tokens / denom:.2f}",
            f"- Estimated total cost (USD): {total_cost:.6f}",
            f"- Estimated cost per request (USD): {total_cost / denom:.6f}",
            "",
            "## Pricing assumptions",
            "",
            "- Gemini 2.5 Flash: $0.15 / 1M input tokens, $0.60 / 1M output tokens",
            "- Gemini 2.5 Pro: $1.25 / 1M input tokens, $10.00 / 1M output tokens",
            "- Vision extraction uses Gemini 2.5 Flash and is included in the Flash totals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    name = (model or "").lower()
    if "pro" in name:
        return input_tokens * PRO_INPUT_COST + output_tokens * PRO_OUTPUT_COST
    return input_tokens * FLASH_INPUT_COST + output_tokens * FLASH_OUTPUT_COST


def _csv_amount(value: Any) -> str:
    try:
        amount = round(float(value), 2)
    except (TypeError, ValueError):
        return "0"
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def _load_env() -> None:
    load_dotenv(HERE / ".env")
    load_dotenv(HERE.parent / ".env")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buy or Wait? decision pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=HERE.parent / "dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "output.csv",
        help="Path to write output.csv",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run against sample_requests.csv instead of requests.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Gemini workers (default 1)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between sequential Gemini calls",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

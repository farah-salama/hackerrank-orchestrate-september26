"""Load Buy or Wait? CSVs and provide dated FX conversion."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return datetime.strptime(text[:10], "%Y-%m-%d").date()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_pipe_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def format_date(value: date | None) -> str:
    return value.isoformat() if value else ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FxLookup:
    """Dated FX conversion with inverse rates and multi-hop paths."""

    def __init__(self, rates: dict[tuple[str, str], list[tuple[date, float]]]):
        self.rates = rates
        currencies: set[str] = set()
        for src, dst in rates:
            currencies.add(src)
            currencies.add(dst)
        self.currencies = currencies

    @classmethod
    def from_rows(cls, rows: list[dict[str, str]]) -> "FxLookup":
        grouped: dict[tuple[str, str], list[tuple[date, float]]] = defaultdict(list)
        for row in rows:
            rate_date = parse_date(row.get("rate_date"))
            rate = parse_float(row.get("rate"))
            src = (row.get("from_currency") or "").strip()
            dst = (row.get("to_currency") or "").strip()
            if rate_date is None or rate is None or not src or not dst:
                continue
            grouped[(src, dst)].append((rate_date, rate))
        for pair in grouped:
            grouped[pair].sort()
        return cls(dict(grouped))

    def convert(
        self,
        amount: float | None,
        from_curr: str,
        to_curr: str,
        on_date: date | None,
    ) -> float:
        if amount is None:
            return 0.0
        src = (from_curr or "").strip()
        dst = (to_curr or "").strip()
        if src == dst or not src or not dst:
            return float(amount)
        if on_date is None:
            raise ValueError(f"No settlement date for FX conversion {src}->{dst}")
        rate = self._path_rate(src, dst, on_date)
        if rate is None:
            raise ValueError(f"No rate path {src}->{dst} on {on_date}")
        return float(amount) * rate

    def _path_rate(self, src: str, dst: str, on_date: date) -> float | None:
        direct = self._direct(src, dst, on_date)
        if direct is not None:
            return direct
        queue = deque([(src, 1.0)])
        seen = {src}
        while queue:
            current, acc = queue.popleft()
            if current == dst:
                return acc
            for nxt in self.currencies:
                if nxt in seen:
                    continue
                hop = self._direct(current, nxt, on_date)
                if hop is None:
                    continue
                seen.add(nxt)
                queue.append((nxt, acc * hop))
        return None

    def _direct(self, from_curr: str, to_curr: str, on_date: date) -> float | None:
        pairs = self.rates.get((from_curr, to_curr))
        if pairs:
            return self._rate_on_or_before(pairs, on_date)
        inverse = self.rates.get((to_curr, from_curr))
        if inverse:
            inv = self._rate_on_or_before(inverse, on_date)
            return (1.0 / inv) if inv else None
        return None

    @staticmethod
    def _rate_on_or_before(pairs: list[tuple[date, float]], on_date: date) -> float | None:
        if not pairs:
            return None
        lo, hi = 0, len(pairs) - 1
        found = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if pairs[mid][0] <= on_date:
                found = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if found >= 0:
            return pairs[found][1]
        return pairs[0][1]


def _parse_profile(row: dict[str, str]) -> dict[str, Any]:
    max_months = parse_float(row.get("max_installment_months"))
    return {
        "user_id": row["user_id"],
        "home_currency": row["home_currency"].strip(),
        "current_available_balance": parse_float(row.get("current_available_balance")) or 0.0,
        "minimum_balance_to_keep": parse_float(row.get("minimum_balance_to_keep")) or 0.0,
        "financial_priorities": parse_pipe_list(row.get("financial_priorities")),
        "expense_categories_to_protect": parse_pipe_list(row.get("expense_categories_to_protect")),
        "expense_categories_user_is_willing_to_reduce": parse_pipe_list(
            row.get("expense_categories_user_is_willing_to_reduce")
        ),
        "expense_categories_user_is_willing_to_stop": parse_pipe_list(
            row.get("expense_categories_user_is_willing_to_stop")
        ),
        "payment_methods_user_will_consider": parse_pipe_list(
            row.get("payment_methods_user_will_consider")
        ),
        "max_installment_months": int(max_months) if max_months is not None else None,
    }


def _parse_event(row: dict[str, str]) -> dict[str, Any]:
    event_date = parse_date(row.get("event_date"))
    settlement_date = parse_date(row.get("settlement_date")) or event_date
    flexibility = (row.get("flexibility") or "").strip() or None
    return {
        "event_id": row["event_id"],
        "user_id": row["user_id"],
        "event_type": row.get("event_type", ""),
        "description": row.get("description", ""),
        "category": row.get("category", ""),
        "direction": row.get("direction", ""),
        "amount": parse_float(row.get("amount")),
        "currency": (row.get("currency") or "").strip(),
        "event_date": event_date,
        "settlement_date": settlement_date,
        "status": (row.get("status") or "").strip(),
        "linked_event_id": (row.get("linked_event_id") or "").strip() or None,
        "flexibility": flexibility,
        "minimum_allowed_amount": parse_float(row.get("minimum_allowed_amount")),
        "amount_missing": parse_float(row.get("amount")) is None,
    }


def _parse_option(row: dict[str, str]) -> dict[str, Any]:
    return {
        "payment_option_id": row["payment_option_id"],
        "request_id": row["request_id"],
        "payment_method": row.get("payment_method", ""),
        "payment_amount": parse_float(row.get("payment_amount")),
        "number_of_payments": int(parse_float(row.get("number_of_payments")) or 0),
        "first_payment_date": parse_date(row.get("first_payment_date")),
        "payment_frequency_days": parse_float(row.get("payment_frequency_days")),
        "financing_fee": parse_float(row.get("financing_fee")) or 0.0,
        "total_payable_amount": parse_float(row.get("total_payable_amount")),
    }


def _parse_message(row: dict[str, str]) -> dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "user_id": row["user_id"],
        "request_id": (row.get("request_id") or "").strip() or None,
        "related_event_id": (row.get("related_event_id") or "").strip() or None,
        "sent_at": row.get("sent_at", ""),
        "source_type": row.get("source_type", ""),
        "message_text": row.get("message_text", ""),
    }


def _parse_image(row: dict[str, str]) -> dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "user_id": row["user_id"],
        "request_id": row["request_id"],
        "related_event_id": (row.get("related_event_id") or "").strip() or None,
    }


def _parse_request(row: dict[str, str], include_labels: bool = False) -> dict[str, Any]:
    parsed = {
        "request_id": row["request_id"],
        "user_id": row["user_id"],
        "request_date": parse_date(row.get("request_date")),
        "request_type": row.get("request_type", ""),
        "requested_amount": parse_float(row.get("requested_amount")) or 0.0,
        "desired_completion_date": parse_date(row.get("desired_completion_date")),
        "allows_partial_payment": parse_bool(row.get("allows_partial_payment")),
        "request_text": row.get("request_text", ""),
    }
    if include_labels:
        for key in (
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
            "decision_explanation",
        ):
            if key in row:
                parsed[key] = row[key]
    return parsed


@dataclass
class Dataset:
    dataset_dir: Path
    profiles: dict[str, dict[str, Any]]
    events_by_user: dict[str, list[dict[str, Any]]]
    events_by_id: dict[str, dict[str, Any]]
    options_by_request: dict[str, list[dict[str, Any]]]
    messages_by_user: dict[str, list[dict[str, Any]]]
    messages_by_request: dict[str, list[dict[str, Any]]]
    images_by_request: dict[str, list[dict[str, Any]]]
    images: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    sample_requests: list[dict[str, Any]]
    fx: FxLookup
    exchange_rates: list[dict[str, str]] = field(default_factory=list)

    def patch_event_amount(
        self,
        event_id: str,
        amount: float,
        currency: str | None = None,
    ) -> None:
        event = self.events_by_id.get(event_id)
        if event is None:
            return
        event["amount"] = float(amount)
        event["amount_missing"] = False
        if currency:
            event["currency"] = currency.strip()

    def missing_image_amounts(
        self,
        request_ids: set[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Return (image_id, related_event_id) pairs that still need vision extraction."""
        pairs: list[tuple[str, str]] = []
        for image in self.images:
            request_id = image["request_id"]
            if request_ids is not None and request_id not in request_ids:
                continue
            event_id = image.get("related_event_id")
            if not event_id:
                continue
            event = self.events_by_id.get(event_id)
            if event is None or event.get("amount") is not None:
                continue
            pairs.append((image["image_id"], event_id))
        return pairs


def load_dataset(dataset_dir: str | Path) -> Dataset:
    root = Path(dataset_dir)
    profiles = {
        row["user_id"]: _parse_profile(row)
        for row in _read_csv(root / "financial_profiles.csv")
    }

    events_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_id: dict[str, dict[str, Any]] = {}
    for row in _read_csv(root / "financial_events.csv"):
        event = _parse_event(row)
        events_by_id[event["event_id"]] = event
        events_by_user[event["user_id"]].append(event)

    options_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(root / "request_payment_options.csv"):
        option = _parse_option(row)
        options_by_request[option["request_id"]].append(option)

    messages_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    messages_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(root / "messages.csv"):
        message = _parse_message(row)
        messages_by_user[message["user_id"]].append(message)
        if message["request_id"]:
            messages_by_request[message["request_id"]].append(message)

    images = [_parse_image(row) for row in _read_csv(root / "images.csv")]
    images_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        images_by_request[image["request_id"]].append(image)

    exchange_rows = _read_csv(root / "exchange_rates.csv")
    return Dataset(
        dataset_dir=root,
        profiles=profiles,
        events_by_user=dict(events_by_user),
        events_by_id=events_by_id,
        options_by_request=dict(options_by_request),
        messages_by_user=dict(messages_by_user),
        messages_by_request=dict(messages_by_request),
        images_by_request=dict(images_by_request),
        images=images,
        requests=[_parse_request(row) for row in _read_csv(root / "requests.csv")],
        sample_requests=[
            _parse_request(row, include_labels=True)
            for row in _read_csv(root / "sample_requests.csv")
        ],
        fx=FxLookup.from_rows(exchange_rows),
        exchange_rates=exchange_rows,
    )

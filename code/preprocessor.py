"""Stage 1: reconstruct a user's 90-day financial position for one request."""

from __future__ import annotations

import calendar
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from loader import format_date, parse_date, parse_float


FORECAST_DAYS = 90
IGNORED_STATUSES = {"cancelled", "failed"}
ADJUSTABLE_FLEX = {"stoppable", "reducible", "reducible_or_stoppable"}
OVERLAP_DAYS = 3


def build_context(
    request: dict[str, Any],
    profile: dict[str, Any],
    events: list[dict[str, Any]],
    payment_options: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    fx_lookup: Any,
) -> dict[str, Any]:
    d0 = _as_date(request.get("request_date"))
    if d0 is None:
        raise ValueError(f"Request {request.get('request_id')} is missing request_date")

    home = profile["home_currency"]
    converted, blank_event_ids = _apply_fx(events, home, fx_lookup)
    buckets = classify_events(converted, d0)
    recurring = detect_all_recurring(buckets["settled_debits"])
    recurring_income = detect_recurring_income(
        events=converted,
        d0=d0,
        confirmed_income=buckets["confirmed_income"],
    )
    adjustable = build_adjustable_events(buckets["adjustable"], profile, recurring)
    cashflows = build_cashflow_timeline(
        d0=d0,
        pending_debits=buckets["pending_debits"],
        scheduled_debits=buckets["scheduled_debits"],
        confirmed_income=buckets["confirmed_income"],
        recurring=recurring,
        recurring_income=recurring_income,
    )
    running = build_running_balance(
        d0, profile["current_available_balance"], cashflows
    )
    amount_safe = compute_amount_safe_to_pay(
        running, d0, request["requested_amount"], profile["minimum_balance_to_keep"]
    )
    earliest = compute_earliest_full_payment_date(
        running, d0, request["requested_amount"], profile["minimum_balance_to_keep"]
    )
    return assemble_context(
        request=request,
        profile=profile,
        d0=d0,
        running=running,
        amount_safe=amount_safe,
        earliest=earliest,
        payment_options=payment_options,
        adjustable=adjustable,
        pending_debits=buckets["pending_debits"],
        confirmed_income=buckets["confirmed_income"],
        messages=messages,
        blank_event_ids=blank_event_ids,
    )


def classify_events(events: list[dict[str, Any]], d0: date) -> dict[str, list[dict[str, Any]]]:
    settled_debits: list[dict[str, Any]] = []
    pending_debits: list[dict[str, Any]] = []
    scheduled_debits: list[dict[str, Any]] = []
    confirmed_income: list[dict[str, Any]] = []
    adjustable: list[dict[str, Any]] = []

    for event in events:
        if _should_ignore(event):
            continue
        status = event.get("status")
        direction = event.get("direction")
        category = event.get("category")
        settlement = event.get("settlement_date")

        if direction == "debit" and status == "pending":
            pending_debits.append(event)
        elif direction == "debit" and status == "scheduled":
            scheduled_debits.append(event)
        elif (
            direction == "credit"
            and category == "salary"
            and status in {"scheduled", "pending"}
        ):
            confirmed_income.append(event)

        if direction == "debit" and status == "settled" and settlement is not None and settlement < d0:
            settled_debits.append(event)
            if event.get("flexibility") in ADJUSTABLE_FLEX:
                adjustable.append(event)

    return {
        "settled_debits": settled_debits,
        "pending_debits": pending_debits,
        "scheduled_debits": scheduled_debits,
        "confirmed_income": confirmed_income,
        "adjustable": adjustable,
    }


def detect_recurring(events_in_group: list[dict[str, Any]]) -> dict[str, Any] | None:
    sorted_events = sorted(
        events_in_group,
        key=lambda event: event.get("settlement_date") or date.min,
    )
    dates = [event["settlement_date"] for event in sorted_events if event.get("settlement_date")]
    if len(dates) < 3:
        return None

    intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not intervals:
        return None
    rough = statistics.median(intervals)
    if rough <= 0:
        return None
    clean = [gap for gap in intervals if 0.5 * rough <= gap <= 1.5 * rough]
    median_gap = statistics.median(clean) if clean else rough
    if median_gap <= 0:
        return None

    if 27 <= median_gap <= 33:
        period = 30
    elif 12 <= median_gap <= 17:
        period = 14
    elif 6 <= median_gap <= 8:
        period = 7
    else:
        period = max(1, int(round(median_gap)))

    recent = sorted_events[-6:]
    amounts = [float(event.get("amount_home") or 0.0) for event in recent]
    projected_amount = _percentile(amounts, 70)

    last_date = dates[-1]
    return {
        "period": period,
        "last_date": last_date,
        "amount": _money(projected_amount),
        "next": _advance(last_date, period),
        "event_id": sorted_events[-1]["event_id"],
        "category": sorted_events[-1].get("category"),
        "flexibility": sorted_events[-1].get("flexibility"),
        "minimum_allowed_amount": sorted_events[-1].get("minimum_allowed_amount_home"),
    }


def detect_all_recurring(settled_debits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in settled_debits:
        flexibility = event.get("flexibility")
        category = event.get("category")
        if not flexibility or not category:
            continue
        groups[(category, flexibility)].append(event)

    detected: dict[str, dict[str, Any]] = {}
    for (category, flexibility), group in groups.items():
        pattern = detect_recurring(group)
        if pattern is None:
            continue
        detected[f"{category}|{flexibility}"] = pattern
    return detected


def detect_recurring_income(
    events: list[dict[str, Any]],
    d0: date,
    confirmed_income: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Project later salary dates when history already supports a cycle.

    Only salary is projected. The latest scheduled/pending amount is used so a
    prorated first paycheck does not drag the forecast down.
    """
    series: list[dict[str, Any]] = []
    for event in events:
        if _should_ignore(event):
            continue
        if event.get("category") != "salary" or event.get("direction") != "credit":
            continue
        status = event.get("status")
        settlement = event.get("settlement_date")
        if settlement is None:
            continue
        if status == "settled" and settlement < d0:
            series.append(event)
        elif status in {"scheduled", "pending"}:
            series.append(event)
    if len(series) < 2:
        return None
    pattern = detect_recurring(series)
    if pattern is None:
        # Two monthly salaries are enough to confirm the cycle.
        ordered = sorted(series, key=lambda item: item.get("settlement_date") or date.min)
        dates = [item["settlement_date"] for item in ordered if item.get("settlement_date")]
        if len(dates) < 2:
            return None
        gap = (dates[-1] - dates[-2]).days
        if not (27 <= gap <= 33):
            return None
        latest = next(
            (
                item
                for item in reversed(ordered)
                if item.get("status") in {"scheduled", "pending"}
            ),
            ordered[-1],
        )
        period = 30
        last_date = dates[-1]
        pattern = {
            "period": period,
            "last_date": last_date,
            "amount": _money(latest.get("amount_home") or 0.0),
            "next": _advance(last_date, period),
            "event_id": latest["event_id"],
            "category": "salary",
            "flexibility": latest.get("flexibility"),
            "minimum_allowed_amount": None,
        }
    dated_confirmed = [
        event
        for event in confirmed_income
        if event.get("amount_home") is not None and event.get("settlement_date") is not None
    ]
    if dated_confirmed:
        latest_confirmed = max(
            dated_confirmed,
            key=lambda event: event["settlement_date"],
        )
        pattern["amount"] = _money(latest_confirmed.get("amount_home") or 0.0)
    return pattern


def build_adjustable_events(
    adjustable: list[dict[str, Any]],
    profile: dict[str, Any],
    recurring: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    stop_cats = set(profile.get("expense_categories_user_is_willing_to_stop") or [])
    reduce_cats = set(profile.get("expense_categories_user_is_willing_to_reduce") or [])
    latest_by_category: dict[str, dict[str, Any]] = {}
    for event in sorted(
        adjustable,
        key=lambda item: item.get("settlement_date") or date.min,
    ):
        category = event.get("category")
        flexibility = event.get("flexibility")
        if not category or flexibility not in ADJUSTABLE_FLEX:
            continue
        eligible = False
        if flexibility in {"stoppable", "reducible_or_stoppable"} and category in stop_cats:
            eligible = True
        if flexibility in {"reducible", "reducible_or_stoppable"} and category in reduce_cats:
            eligible = True
        if not eligible:
            continue
        latest_by_category[category] = event

    rows: list[dict[str, Any]] = []
    for category, event in latest_by_category.items():
        pattern = recurring.get(f"{category}|{event.get('flexibility')}")
        next_date = pattern["next"] if pattern else None
        projected = (
            pattern["amount"]
            if pattern
            else _money(event.get("amount_home") or 0.0)
        )
        min_allowed = event.get("minimum_allowed_amount_home")
        rows.append(
            {
                "event_id": event["event_id"],
                "category": category,
                "flexibility": event.get("flexibility"),
                "projected_amount": projected,
                "minimum_allowed_amount": _money(min_allowed) if min_allowed is not None else None,
                "next_expected_date": format_date(next_date),
            }
        )
    return rows


def build_cashflow_timeline(
    d0: date,
    pending_debits: list[dict[str, Any]],
    scheduled_debits: list[dict[str, Any]],
    confirmed_income: list[dict[str, Any]],
    recurring: dict[str, dict[str, Any]],
    recurring_income: dict[str, Any] | None = None,
) -> list[tuple[date, float]]:
    end = d0 + timedelta(days=FORECAST_DAYS)
    cashflows: list[tuple[date, float]] = []
    covered: list[tuple[str, date]] = []

    for event in pending_debits:
        amount = float(event.get("amount_home") or 0.0)
        cashflows.append((d0, -amount))
        covered.append((event.get("category") or "", event.get("settlement_date") or d0))

    for event in scheduled_debits:
        when = event.get("settlement_date")
        if when is None:
            continue
        amount = float(event.get("amount_home") or 0.0)
        if d0 <= when <= end:
            cashflows.append((when, -amount))
        elif when < d0:
            cashflows.append((d0, -amount))
        covered.append((event.get("category") or "", when))

    income_covered: list[tuple[str, date]] = []
    for event in confirmed_income:
        when = event.get("settlement_date")
        if when is None or not (d0 <= when <= end):
            continue
        cashflows.append((when, float(event.get("amount_home") or 0.0)))
        income_covered.append(("salary", when))

    if recurring_income:
        occurrence = recurring_income["next"]
        period = max(1, int(recurring_income["period"]))
        amount = float(recurring_income["amount"])
        while occurrence <= end:
            if occurrence >= d0 and not _is_covered("salary", occurrence, income_covered):
                cashflows.append((occurrence, amount))
                income_covered.append(("salary", occurrence))
            occurrence = _advance(occurrence, period)

    for pattern in recurring.values():
        category = pattern.get("category") or ""
        occurrence = pattern["next"]
        period = max(1, int(pattern["period"]))
        amount = float(pattern["amount"])
        while occurrence <= end:
            if occurrence >= d0 and not _is_covered(category, occurrence, covered):
                cashflows.append((occurrence, -amount))
                covered.append((category, occurrence))
            occurrence = _advance(occurrence, period)

    return cashflows


def build_running_balance(
    d0: date,
    starting_balance: float,
    cashflows: list[tuple[date, float]],
) -> dict[date, float]:
    daily_delta: dict[date, float] = defaultdict(float)
    for when, delta in cashflows:
        daily_delta[when] += delta

    balance = float(starting_balance)
    running: dict[date, float] = {}
    for offset in range(FORECAST_DAYS + 1):
        day = d0 + timedelta(days=offset)
        balance += daily_delta.get(day, 0.0)
        running[day] = balance
    return running


def compute_amount_safe_to_pay(
    running: dict[date, float],
    d0: date,
    requested_amount: float,
    minimum_balance: float,
) -> float:
    future_min = min(running[day] for day in running if day >= d0)
    headroom = future_min - minimum_balance
    return _money(min(requested_amount, max(0.0, headroom)))


def compute_earliest_full_payment_date(
    running: dict[date, float],
    d0: date,
    requested_amount: float,
    minimum_balance: float,
) -> date | None:
    target = float(requested_amount)
    for offset in range(FORECAST_DAYS + 1):
        day = d0 + timedelta(days=offset)
        if running[day] - target < minimum_balance:
            continue
        post_payment_min = min(running[later] - target for later in running if later >= day)
        if post_payment_min >= minimum_balance:
            return day
    return None


def assemble_context(
    request: dict[str, Any],
    profile: dict[str, Any],
    d0: date,
    running: dict[date, float],
    amount_safe: float,
    earliest: date | None,
    payment_options: list[dict[str, Any]],
    adjustable: list[dict[str, Any]],
    pending_debits: list[dict[str, Any]],
    confirmed_income: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    blank_event_ids: list[str],
) -> dict[str, Any]:
    context = {
        "request": {
            "request_id": request["request_id"],
            "user_id": request["user_id"],
            "request_date": format_date(d0),
            "request_type": request.get("request_type", ""),
            "requested_amount": _money(request.get("requested_amount") or 0.0),
            "desired_completion_date": format_date(_as_date(request.get("desired_completion_date"))),
            "allows_partial_payment": bool(request.get("allows_partial_payment")),
            "request_text": request.get("request_text", ""),
        },
        "profile": {
            "home_currency": profile["home_currency"],
            "current_available_balance": _money(profile["current_available_balance"]),
            "minimum_balance_to_keep": _money(profile["minimum_balance_to_keep"]),
            "financial_priorities": list(profile.get("financial_priorities") or []),
            "expense_categories_to_protect": list(profile.get("expense_categories_to_protect") or []),
            "expense_categories_user_is_willing_to_reduce": list(
                profile.get("expense_categories_user_is_willing_to_reduce") or []
            ),
            "expense_categories_user_is_willing_to_stop": list(
                profile.get("expense_categories_user_is_willing_to_stop") or []
            ),
            "payment_methods_user_will_consider": list(
                profile.get("payment_methods_user_will_consider") or []
            ),
            "max_installment_months": profile.get("max_installment_months"),
        },
        "precomputed": {
            "amount_safe_to_pay_baseline": amount_safe,
            "earliest_date_for_full_payment": format_date(earliest),
            "min_balance_90days": _money(min(running.values()) if running else 0.0),
            "balance_on_request_date": _money(running[d0]),
        },
        "payment_options": [_serialize_option(option) for option in payment_options],
        "adjustable_events": adjustable,
        "pending_debits": [
            {
                "event_id": event["event_id"],
                "description": event.get("description", ""),
                "amount_home": _money(event.get("amount_home") or 0.0),
                "settlement_date": format_date(event.get("settlement_date")),
            }
            for event in pending_debits
        ],
        "future_income": [
            {
                "event_id": event["event_id"],
                "description": event.get("description", ""),
                "amount_home": _money(event.get("amount_home") or 0.0),
                "settlement_date": format_date(event.get("settlement_date")),
                "status": event.get("status"),
            }
            for event in confirmed_income
            if event.get("settlement_date") is not None
            and d0 <= event["settlement_date"] <= d0 + timedelta(days=FORECAST_DAYS)
        ],
        "messages": [
            {
                "message_id": message.get("message_id"),
                "sent_at": message.get("sent_at", ""),
                "source_type": message.get("source_type", ""),
                "message_text": message.get("message_text", ""),
                "related_event_id": message.get("related_event_id"),
            }
            for message in messages
        ],
    }
    if blank_event_ids:
        context["precomputed"]["blank_amount_event_ids"] = blank_event_ids
    return context


def _apply_fx(
    events: list[dict[str, Any]],
    home_currency: str,
    fx_lookup: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    converted: list[dict[str, Any]] = []
    blank_event_ids: list[str] = []
    for original in events:
        event = dict(original)
        amount = parse_float(event.get("amount"))
        if amount is None:
            blank_event_ids.append(event["event_id"])
            event["amount_home"] = 0.0
            event["minimum_allowed_amount_home"] = None
            converted.append(event)
            continue
        on_date = event.get("settlement_date") or event.get("event_date")
        event["amount_home"] = fx_lookup.convert(
            amount, event.get("currency") or home_currency, home_currency, on_date
        )
        min_allowed = parse_float(event.get("minimum_allowed_amount"))
        event["minimum_allowed_amount_home"] = (
            fx_lookup.convert(
                min_allowed,
                event.get("currency") or home_currency,
                home_currency,
                on_date,
            )
            if min_allowed is not None
            else None
        )
        converted.append(event)
    return converted, blank_event_ids


def _should_ignore(event: dict[str, Any]) -> bool:
    if event.get("status") in IGNORED_STATUSES:
        return True
    if event.get("status") == "unrealized":
        return True
    if event.get("direction") == "non_cash":
        return True
    return False


def _is_covered(category: str, occurrence: date, covered: list[tuple[str, date]]) -> bool:
    for covered_category, covered_date in covered:
        if covered_category != category:
            continue
        if abs((occurrence - covered_date).days) <= OVERLAP_DAYS:
            return True
    return False


def _serialize_option(option: dict[str, Any]) -> dict[str, Any]:
    frequency = option.get("payment_frequency_days")
    return {
        "payment_option_id": option.get("payment_option_id"),
        "payment_method": option.get("payment_method"),
        "payment_amount": _money(option.get("payment_amount")) if option.get("payment_amount") is not None else None,
        "number_of_payments": option.get("number_of_payments"),
        "first_payment_date": format_date(_as_date(option.get("first_payment_date"))),
        "payment_frequency_days": int(frequency) if frequency is not None else None,
        "financing_fee": _money(option.get("financing_fee") or 0.0),
        "total_payable_amount": _money(option.get("total_payable_amount"))
        if option.get("total_payable_amount") is not None
        else None,
    }


def _advance(start: date, period: int) -> date:
    if period == 30:
        return _add_months(start, 1)
    return start + timedelta(days=period)


def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return parse_date(value)


def _money(value: float | None) -> float:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0.0
    return round(float(value), 2)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction

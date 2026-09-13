"""Hard-clamp Gemini output so every row satisfies the challenge contract."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from loader import format_date, parse_date, parse_float


USER_METHODS = {"full_payment", "partial_payment", "installments"}
VALID_STATUS = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}
VALID_METHODS = USER_METHODS | {"wait", "not_recommended"}


def validate(raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    request = context["request"]
    profile = context["profile"]
    precomputed = context.get("precomputed") or {}
    options = context.get("payment_options") or []
    adjustable = context.get("adjustable_events") or []

    requested = float(request.get("requested_amount") or 0.0)
    request_date = request.get("request_date") or ""
    allowed_methods = set(profile.get("payment_methods_user_will_consider") or [])
    max_months = profile.get("max_installment_months")

    result = {
        "amount_safe_to_pay": raw.get("amount_safe_to_pay"),
        "affordability_status": raw.get("affordability_status") or "not_affordable",
        "recommended_payment_method": raw.get("recommended_payment_method") or "not_recommended",
        "payment_plan": raw.get("payment_plan") or "none",
        "earliest_date_for_full_payment": raw.get("earliest_date_for_full_payment")
        if raw.get("earliest_date_for_full_payment") is not None
        else (precomputed.get("earliest_date_for_full_payment") or ""),
        "spending_changes_needed": raw.get("spending_changes_needed") or "none",
        "decision_explanation": raw.get("decision_explanation") or "",
    }

    # 1. Clamp amount_safe_to_pay into [0, requested_amount].
    amount_safe = parse_float(result["amount_safe_to_pay"])
    if amount_safe is None:
        amount_safe = parse_float(precomputed.get("amount_safe_to_pay_baseline")) or 0.0
    result["amount_safe_to_pay"] = _money(max(0.0, min(requested, amount_safe)))

    if result["affordability_status"] not in VALID_STATUS:
        result["affordability_status"] = "not_affordable"
    if result["recommended_payment_method"] not in VALID_METHODS:
        result["recommended_payment_method"] = "not_recommended"

    # 2. affordable_now always pays in full on the request date.
    if result["affordability_status"] == "affordable_now":
        result["earliest_date_for_full_payment"] = request_date

    method = result["recommended_payment_method"]

    # 3. Reject user-facing methods the profile does not allow.
    if method in USER_METHODS and method not in allowed_methods:
        method = _reject_method(result)

    # 4 + 5. Installment duration and exact option match.
    if method == "installments":
        method = _validate_installments(result, options, allowed_methods, max_months)

    # 6. Partial payment must be two amounts that sum to the request.
    if method == "partial_payment":
        method = _validate_partial_payment(result, request, requested)

    # 7. Spending changes may only reference adjustable events.
    result["spending_changes_needed"] = _validate_spending_changes(
        result.get("spending_changes_needed") or "none",
        adjustable,
    )

    if not result["decision_explanation"]:
        result["decision_explanation"] = "Unable to determine recommendation."
    if not result["payment_plan"]:
        result["payment_plan"] = "none"
    if result["earliest_date_for_full_payment"] is None:
        result["earliest_date_for_full_payment"] = ""
    return result


def installment_plan_from_option(option: dict[str, Any]) -> str:
    count = int(option.get("number_of_payments") or 0)
    first = _as_date(option.get("first_payment_date"))
    amount = option.get("payment_amount")
    if count <= 0 or first is None or amount is None:
        return "none"
    frequency = int(option.get("payment_frequency_days") or 0)
    parts = []
    for index in range(count):
        day = first + timedelta(days=frequency * index)
        parts.append(f"{format_date(day)}:{_format_amount(amount)}")
    return "|".join(parts)


def parse_payment_plan(plan: str) -> list[tuple[str, float]]:
    if not plan or plan.strip().lower() == "none":
        return []
    entries: list[tuple[str, float]] = []
    for part in plan.split("|"):
        text = part.strip()
        if not text or ":" not in text:
            continue
        date_text, amount_text = text.rsplit(":", 1)
        amount = parse_float(amount_text)
        if amount is None:
            continue
        entries.append((date_text.strip(), _money(amount)))
    return entries


def _validate_installments(
    result: dict[str, Any],
    options: list[dict[str, Any]],
    allowed_methods: set[str],
    max_months: int | None,
) -> str:
    if "installments" not in allowed_methods or max_months is None:
        return _reject_method(result)

    matched = _matching_installment_option(result.get("payment_plan") or "", options)
    if matched is not None and _installment_allowed(matched, max_months):
        result["payment_plan"] = installment_plan_from_option(matched)
        return "installments"

    substitute = _best_installment_option(options, max_months)
    if substitute is None:
        return _reject_method(result)
    result["payment_plan"] = installment_plan_from_option(substitute)
    return "installments"


def _validate_partial_payment(
    result: dict[str, Any],
    request: dict[str, Any],
    requested: float,
) -> str:
    amount_safe = float(result["amount_safe_to_pay"])
    if not request.get("allows_partial_payment"):
        return _reject_method(result)
    if not (0.0 < amount_safe < requested):
        return _reject_method(result)

    earliest = result.get("earliest_date_for_full_payment") or ""
    deadline = request.get("desired_completion_date") or ""
    if not earliest:
        return _reject_method(result)
    if deadline and earliest > deadline:
        return _reject_method(result)

    remainder = _money(requested - amount_safe)
    request_date = request.get("request_date") or ""
    entries = parse_payment_plan(result.get("payment_plan") or "")
    valid = (
        len(entries) == 2
        and entries[0][0] == request_date
        and _almost_equal(entries[0][1], amount_safe)
        and entries[1][0] == earliest
        and _almost_equal(entries[0][1] + entries[1][1], requested)
    )
    if not valid:
        result["payment_plan"] = (
            f"{request_date}:{_format_amount(amount_safe)}|"
            f"{earliest}:{_format_amount(remainder)}"
        )
    return "partial_payment"


def _validate_spending_changes(
    raw: str,
    adjustable: list[dict[str, Any]],
) -> str:
    if not raw or raw.strip().lower() == "none":
        return "none"
    by_id = {item.get("event_id"): item for item in adjustable if item.get("event_id")}
    kept: list[str] = []
    for part in raw.split("|"):
        token = part.strip()
        if not token:
            continue
        pieces = token.split(":")
        kind = pieces[0]
        if kind == "stop" and len(pieces) == 2:
            event_id = pieces[1]
            event = by_id.get(event_id)
            if event is None:
                continue
            if event.get("flexibility") not in {"stoppable", "reducible_or_stoppable"}:
                continue
            kept.append(f"stop:{event_id}")
        elif kind == "reduce_to" and len(pieces) == 3:
            event_id = pieces[1]
            amount = parse_float(pieces[2])
            event = by_id.get(event_id)
            if event is None or amount is None:
                continue
            if event.get("flexibility") not in {"reducible", "reducible_or_stoppable"}:
                continue
            minimum = event.get("minimum_allowed_amount")
            if minimum is not None and amount < float(minimum):
                amount = float(minimum)
            kept.append(f"reduce_to:{event_id}:{_format_amount(amount)}")
        if len(kept) == 3:
            break
    return "|".join(kept) if kept else "none"


def _matching_installment_option(
    plan: str,
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    entries = parse_payment_plan(plan)
    if not entries:
        return None
    for option in options:
        if option.get("payment_method") != "installments":
            continue
        expected = parse_payment_plan(installment_plan_from_option(option))
        if _plans_match(entries, expected):
            return option
    return None


def _best_installment_option(
    options: list[dict[str, Any]],
    max_months: int | None,
) -> dict[str, Any] | None:
    eligible = [
        option
        for option in options
        if option.get("payment_method") == "installments"
        and _installment_allowed(option, max_months)
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda option: (
            float(option.get("total_payable_amount") or option.get("payment_amount") or 0.0),
            option.get("first_payment_date") or "9999-12-31",
            int(option.get("number_of_payments") or 99),
            option.get("payment_option_id") or "",
        ),
    )


def _installment_allowed(option: dict[str, Any], max_months: int | None) -> bool:
    if max_months is None:
        return False
    count = int(option.get("number_of_payments") or 0)
    frequency = option.get("payment_frequency_days")
    if count <= 0 or frequency is None:
        return False
    frequency = float(frequency)
    # Monthly-style cadences are one month per payment, not freq/30.
    if 27 <= frequency <= 33:
        months = float(count)
    else:
        months = (count * frequency) / 30.0
    return months <= float(max_months) + 1e-9


def _plans_match(
    left: list[tuple[str, float]],
    right: list[tuple[str, float]],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left_date == right_date and _almost_equal(left_amount, right_amount)
        for (left_date, left_amount), (right_date, right_amount) in zip(left, right)
    )


def _reject_method(result: dict[str, Any]) -> str:
    result["recommended_payment_method"] = "not_recommended"
    result["payment_plan"] = "none"
    return "not_recommended"


def _almost_equal(left: float, right: float, tol: float = 0.02) -> bool:
    return abs(left - right) <= tol


def _money(value: float) -> float:
    return round(float(value), 2)


def _format_amount(value: float) -> str:
    amount = _money(value)
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}"


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return parse_date(value)

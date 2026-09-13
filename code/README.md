# Buy or Wait? solution

A two-stage deterministic forecast plus Gemini decision pipeline. Stage 1 reconstructs a 90-day cash position from the dataset. Stage 2 asks Gemini for a structured recommendation. A validator then clamps every row to the challenge contract.

## Approach

The agent never invents income, expenses, payment options, or exchange rates. Numbers come from `dataset/` CSVs, dated FX rows, and optional image amounts. Messages are supporting evidence only.

### 1. Load and convert

`loader.py` indexes profiles, events, payment options, messages, images, and requests. Foreign-currency amounts convert with the settlement-date rate from `exchange_rates.csv` (including inverse and multi-hop paths). Events with a blank amount are flagged for the vision pre-pass.

### 2. Vision pre-pass

Before any request is scored, `main.py` asks Gemini to read `dataset/media/images/<image_id>.png` when an image is linked to an event that still has a missing amount. Extracted `{amount, currency}` values are patched onto that event so the forecast can include them.

### 3. Stage 1 — 90-day forecast (`preprocessor.py`)

For each request the preprocessor:

- Classifies events into pending/scheduled debits, confirmed salary credits, and adjustable historical spend. Cancelled, failed, and non-cash rows are ignored. Pending credits, bonuses, commissions, refunds, and unrealized investments are not treated as cash.
- Detects recurring expenses from settled debit history (weekly / biweekly / monthly when the interval is stable) and projects them forward for 90 days without double-counting a scheduled row on the same date.
- Projects later salary dates when history already supports a monthly cycle. The latest scheduled or pending paycheck amount is used so a prorated first credit does not understate later pay.
- Builds a daily running balance from the current available balance plus those cash flows.
- Computes `amount_safe_to_pay` as the headroom above `minimum_balance_to_keep` after the worst remaining day in the 90-day window, capped at the requested amount.
- Computes `earliest_date_for_full_payment` as the first day a full payment still leaves every later day at or above the minimum.

The assembled context also includes the request, profile preferences, seller payment options, adjustable events, pending debits, confirmed income, and relevant messages.

### 4. Stage 2 — Gemini decision (`gemini_agent.py`)

Gemini `gemini-3.1-flash-lite` receives that context and a JSON schema. Temperature is 0. The system prompt tells the model to:

- Trust the precomputed baseline and only override it downward when an institutional message (employer, bank, utility) states a concrete fact.
- Walk a decision tree: full payment today, then installments that fit `max_installment_months` and the deadline, then a two-payment partial plan, then spending changes, then wait, otherwise not recommended.
- Rank valid plans by completing the request by the deadline, avoiding spending changes, minimizing total cost, starting earlier, fewer payments, then lowest `payment_option_id`.

Rate-limited calls retry up to five times using the API-suggested delay. If the model still fails, a conservative not-recommended fallback is used.

### 5. Validator (`validator.py`)

The validator hard-clamps the model output so the row is evaluable:

- `amount_safe_to_pay` stays in `[0, requested_amount]`.
- Methods the user will not consider are rejected.
- Installment plans must match a supplied option and the month cap.
- Partial plans must be exactly two dated amounts that sum to the request.
- Spending changes may only reference adjustable events, with at most three actions.
- `earliest_date_for_full_payment` is the request date when the status is `affordable_now`.

### 6. Output and usage

Rows are written as `output.csv` with the eight required columns. After the run, `evaluation/usage_report.md` records provider, model name, call counts, tokens, and estimated cost.

## Setup

```bash
cd code
python -m pip install -r requirements.txt
copy .env.example .env
```

Set `GOOGLE_API_KEY` in `code/.env`.

## Run

Full evaluation set (250 requests), writing `output.csv` at the repo root:

```bash
python main.py --dataset ../dataset --output ../output.csv
```

Sample set (25 labeled rows):

```bash
python main.py --dataset ../dataset --output ../sample_output.csv --sample
```

Optional flags: `--sleep` (seconds between sequential Gemini calls, default `0.5`) and `--workers` (parallel calls; keep `1` if you are hitting rate limits).

After a full run, `evaluation/usage_report.md` is overwritten with token and cost totals.

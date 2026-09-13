# Buy or Wait? solution

Python pipeline: Stage 1 reconstructs a 90-day forecast; Stage 2 asks Gemini 2.5 Flash for a structured decision; a validator then clamps the row to the challenge contract.

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

After a full run, `evaluation/usage_report.md` is overwritten with token and cost totals.

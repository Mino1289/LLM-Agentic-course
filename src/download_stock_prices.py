import argparse
import csv
from datetime import date
from pathlib import Path

import yfinance as yf

from env_config import load_env_file


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRADING_DAYS = 23

COMPANIES = [
    {
        "slug": "NVDA",
        "company": "NVIDIA",
        "ticker": "NVDA",
        "market_symbol": "NVDA",
        "currency": "USD",
    },
    {
        "slug": "AMD",
        "company": "Advanced Micro Devices",
        "ticker": "AMD",
        "market_symbol": "AMD",
        "currency": "USD",
    },
    {
        "slug": "INTC",
        "company": "Intel",
        "ticker": "INTC",
        "market_symbol": "INTC",
        "currency": "USD",
    },
    {
        "slug": "TSMC",
        "company": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSMC",
        "market_symbol": "TSM",
        "currency": "USD",
    },
    {
        "slug": "ASML",
        "company": "ASML",
        "ticker": "ASML",
        "market_symbol": "ASML",
        "currency": "USD",
    },
]

CSV_COLUMNS = [
    "company",
    "ticker",
    "market_symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "currency",
    "provider",
    "source_url",
]


def parse_date(value: str):
    return date.fromisoformat(value)


def fetch_yfinance_rows(
    company: dict,
    start_date: date | None,
    end_date: date | None,
    limit: int | None,
):
    ticker = yf.Ticker(company["market_symbol"])
    history = ticker.history(start=start_date, end=end_date)

    if history.empty:
        raise RuntimeError(f"No price data returned for {company['ticker']}")

    if start_date is None:
        history = history.tail(limit)

    source_url = f"https://finance.yahoo.com/quote/{company['market_symbol']}"
    rows = []

    for day_idx, (day, values) in enumerate(history.iterrows()):
        rows.append({
            "company": company["company"],
            "ticker": company["ticker"],
            "market_symbol": company["market_symbol"],
            "date": day.strftime("%Y-%m-%d"),
            "open": values.get("Open", ""),
            "high": values.get("High", ""),
            "low": values.get("Low", ""),
            "close": values.get("Close", ""),
            "volume": values.get("Volume", ""),
            "currency": company["currency"],
            "provider": "yfinance",
            "source_url": source_url,
        })

    return rows


def write_csv(output_path: Path, rows: list[dict]):
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(output_path)


def read_csv(input_path: Path):
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def output_path_for_company(company: dict, suffix: str):
    return RAW_DIR / f"{company['slug']}_stock_prices_{suffix}.csv"


def download_stock_prices(
    start_date: date | None,
    end_date: date | None,
    limit: int | None,
    suffix: str,
    force: bool,
):
    all_rows = []

    for company in COMPANIES:
        output_path = output_path_for_company(company, suffix)

        if output_path.exists() and not force:
            rows = read_csv(output_path)
            print(f"[SKIP] {output_path} already exists ({len(rows)} rows)")
            all_rows.extend(rows)
            continue

        print(f"[DOWNLOAD] {company['ticker']} daily stock prices")
        rows = fetch_yfinance_rows(
            company=company,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        write_csv(output_path, rows)
        print(f"[OK] saved {len(rows)} rows to {output_path}")

        all_rows.extend(rows)

    combined_path = RAW_DIR / f"semiconductor_stock_prices_{suffix}.csv"
    write_csv(
        combined_path,
        sorted(all_rows, key=lambda row: (row["ticker"], row["date"])),
    )
    print(f"[OK] saved {len(all_rows)} rows to {combined_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download daily stock prices as CSV files for the RAG corpus using yfinance."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_TRADING_DAYS,
        help="Number of latest trading days to keep when --start-date is not provided.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="First date to request, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="Last date to request, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--suffix",
        default="last_month",
        help="Filename suffix for generated CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if per-company CSV files already exist.",
    )
    args = parser.parse_args()

    if args.days <= 0:
        parser.error("--days must be greater than 0")

    if args.start_date is not None and args.end_date is None:
        args.end_date = date.today()

    if (
        args.start_date is not None
        and args.end_date is not None
        and args.start_date > args.end_date
    ):
        parser.error("--start-date must be before or equal to --end-date")

    args.limit = args.days if args.start_date is None else None

    return args


def main():
    load_env_file()
    args = parse_args()
    try:
        download_stock_prices(
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            suffix=args.suffix,
            force=args.force,
        )
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from None


if __name__ == "__main__":
    main()

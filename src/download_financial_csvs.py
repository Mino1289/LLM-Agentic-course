import argparse
import csv
import os
import time
from datetime import date
from pathlib import Path

import requests


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_SEC_USER_AGENT = "GENAI-RAG classroom project contact@example.com"

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

COMPANIES = [
    {
        "slug": "NVDA",
        "company": "NVIDIA",
        "rag_ticker": "NVDA",
        "sec_ticker": "NVDA",
    },
    {
        "slug": "AMD",
        "company": "Advanced Micro Devices",
        "rag_ticker": "AMD",
        "sec_ticker": "AMD",
    },
    {
        "slug": "INTC",
        "company": "Intel",
        "rag_ticker": "INTC",
        "sec_ticker": "INTC",
    },
    {
        "slug": "TSMC",
        "company": "Taiwan Semiconductor Manufacturing Company",
        "rag_ticker": "TSMC",
        "sec_ticker": "TSM",
    },
    {
        "slug": "ASML",
        "company": "ASML",
        "rag_ticker": "ASML",
        "sec_ticker": "ASML",
    },
]

METRICS = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
        ("ifrs-full", "RevenueFromContractsWithCustomers"),
    ],
    "gross_profit": [
        ("us-gaap", "GrossProfit"),
        ("ifrs-full", "GrossProfit"),
    ],
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
        ("ifrs-full", "ProfitLossFromOperatingActivities"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("ifrs-full", "ProfitLoss"),
        ("ifrs-full", "ProfitLossAttributableToOwnersOfParent"),
    ],
    "diluted_eps": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("ifrs-full", "DilutedEarningsLossPerShare"),
    ],
    "total_assets": [
        ("us-gaap", "Assets"),
        ("ifrs-full", "Assets"),
    ],
    "total_liabilities": [
        ("us-gaap", "Liabilities"),
        ("ifrs-full", "Liabilities"),
    ],
    "shareholders_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("ifrs-full", "Equity"),
    ],
    "cash_and_equivalents": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ("ifrs-full", "CashAndCashEquivalents"),
    ],
    "operating_cash_flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
    ],
    "capital_expenditures": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("ifrs-full", "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
    ],
    "research_and_development": [
        ("us-gaap", "ResearchAndDevelopmentExpense"),
        ("ifrs-full", "ResearchAndDevelopmentExpense"),
    ],
}

CSV_COLUMNS = [
    "company",
    "ticker",
    "sec_ticker",
    "cik",
    "fiscal_year",
    "fiscal_period",
    "metric",
    "value",
    "unit",
    "taxonomy",
    "concept",
    "form",
    "filed",
    "start",
    "end",
    "source_url",
]


def make_session(user_agent: str):
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    })
    return session


def fetch_json(session: requests.Session, url: str):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_cik_by_ticker(session: requests.Session):
    ticker_data = fetch_json(session, CIK_LOOKUP_URL)
    cik_by_ticker = {}

    for item in ticker_data.values():
        ticker = item["ticker"].upper()
        cik_by_ticker[ticker] = f"{int(item['cik_str']):010d}"

    return cik_by_ticker


def parse_date(value: str | None):
    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def period_year(fact: dict):
    end_date = parse_date(fact.get("end"))

    if end_date:
        return end_date.year

    return fact.get("fy")


def is_annual_duration(fact: dict):
    start_date = parse_date(fact.get("start"))
    end_date = parse_date(fact.get("end"))

    if not start_date or not end_date:
        return True

    return (end_date - start_date).days >= 250


def is_annual_fact(fact: dict, start_year: int, end_year: int | None):
    fact_year = period_year(fact)

    if fact_year is None:
        return False

    if fact_year < start_year:
        return False

    if end_year is not None and fact_year > end_year:
        return False

    if fact.get("form") not in ANNUAL_FORMS:
        return False

    if not is_annual_duration(fact):
        return False

    return fact.get("fp") in {"FY", "CY"} or fact.get("form") in {"20-F", "20-F/A"}


def should_replace(existing: dict | None, candidate: dict):
    if existing is None:
        return True

    existing_rank = (
        existing["_concept_rank"],
        existing["_unit_rank"],
        existing.get("filed") or "",
    )
    candidate_rank = (
        candidate["_concept_rank"],
        candidate["_unit_rank"],
        candidate.get("filed") or "",
    )

    if candidate_rank[:2] < existing_rank[:2]:
        return True

    return candidate_rank[:2] == existing_rank[:2] and candidate_rank[2] > existing_rank[2]


def unit_rank(unit: str):
    if "/" in unit:
        return 1

    return 0


def extract_metric_rows(
    facts: dict,
    company: dict,
    cik: str,
    metric: str,
    candidates: list[tuple[str, str]],
    start_year: int,
    end_year: int | None,
):
    rows_by_year = {}
    source_url = COMPANY_FACTS_URL.format(cik=cik)

    for concept_rank, (taxonomy, concept) in enumerate(candidates):
        concept_data = facts.get("facts", {}).get(taxonomy, {}).get(concept)

        if not concept_data:
            continue

        for unit, unit_facts in concept_data.get("units", {}).items():
            for fact in unit_facts:
                if not is_annual_fact(fact, start_year, end_year):
                    continue

                row = {
                    "company": company["company"],
                    "ticker": company["rag_ticker"],
                    "sec_ticker": company["sec_ticker"],
                    "cik": cik,
                    "fiscal_year": period_year(fact),
                    "fiscal_period": fact.get("fp"),
                    "metric": metric,
                    "value": fact.get("val"),
                    "unit": unit,
                    "taxonomy": taxonomy,
                    "concept": concept,
                    "form": fact.get("form"),
                    "filed": fact.get("filed"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "source_url": source_url,
                    "_concept_rank": concept_rank,
                    "_unit_rank": unit_rank(unit),
                }

                year = row["fiscal_year"]

                if should_replace(rows_by_year.get(year), row):
                    rows_by_year[year] = row

    return [
        clean_output_row(row)
        for row in sorted(rows_by_year.values(), key=lambda item: item["fiscal_year"])
    ]


def clean_output_row(row: dict):
    return {
        column: row.get(column, "")
        for column in CSV_COLUMNS
    }


def extract_company_rows(
    facts: dict,
    company: dict,
    cik: str,
    start_year: int,
    end_year: int | None,
):
    rows = []

    for metric, candidates in METRICS.items():
        rows.extend(
            extract_metric_rows(
                facts=facts,
                company=company,
                cik=cik,
                metric=metric,
                candidates=candidates,
                start_year=start_year,
                end_year=end_year,
            )
        )

    return sorted(
        rows,
        key=lambda row: (row["fiscal_year"], row["metric"]),
    )


def write_csv(output_path: Path, rows: list[dict]):
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(output_path)


def download_company_csvs(
    session: requests.Session,
    start_year: int,
    end_year: int | None,
):
    cik_by_ticker = fetch_cik_by_ticker(session)
    all_rows = []

    for company in COMPANIES:
        cik = cik_by_ticker[company["sec_ticker"]]
        facts_url = COMPANY_FACTS_URL.format(cik=cik)
        print(f"[DOWNLOAD] {company['slug']} SEC company facts")
        facts = fetch_json(session, facts_url)

        rows = extract_company_rows(
            facts=facts,
            company=company,
            cik=cik,
            start_year=start_year,
            end_year=end_year,
        )

        output_path = RAW_DIR / f"{company['slug']}_financial_facts_sec.csv"
        write_csv(output_path, rows)
        print(f"[OK] saved {len(rows)} rows to {output_path}")

        all_rows.extend(rows)
        time.sleep(0.1)

    combined_path = RAW_DIR / "semiconductor_financial_facts_sec.csv"
    write_csv(
        combined_path,
        sorted(all_rows, key=lambda row: (row["ticker"], row["fiscal_year"], row["metric"])),
    )
    print(f"[OK] saved {len(all_rows)} rows to {combined_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download SEC financial facts as CSV files for the semiconductor RAG corpus."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="First fiscal year to keep in the CSV files.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last fiscal year to keep in the CSV files.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
        help="SEC User-Agent header. You can also set SEC_USER_AGENT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    session = make_session(args.user_agent)
    download_company_csvs(
        session=session,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()

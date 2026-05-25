import csv
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from chunking import chunk_text


RAW_DIR = Path("data/raw")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".tsv"}

DOCUMENT_METADATA = {
    "NVDA_2025_10K.pdf": {
        "company": "NVIDIA",
        "ticker": "NVDA",
        "year": 2025,
        "document_type": "10-K",
    },
    "AMD_2024_10K.pdf": {
        "company": "Advanced Micro Devices",
        "ticker": "AMD",
        "year": 2024,
        "document_type": "10-K",
    },
    "INTC_2024_annual_report.pdf": {
        "company": "Intel",
        "ticker": "INTC",
        "year": 2024,
        "document_type": "annual_report",
    },
    "TSMC_2024_annual_report.pdf": {
        "company": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSMC",
        "year": 2024,
        "document_type": "annual_report",
    },
    "ASML_2024_annual_report.pdf": {
        "company": "ASML",
        "ticker": "ASML",
        "year": 2024,
        "document_type": "annual_report",
    },
    "NVDA_2025_Q4_earnings_call.txt": {
        "company": "NVIDIA",
        "ticker": "NVDA",
        "year": 2025,
        "quarter": "Q4",
        "document_type": "earnings_call",
    },
    "AMD_2024_Q4_earnings_call.txt": {
        "company": "Advanced Micro Devices",
        "ticker": "AMD",
        "year": 2024,
        "quarter": "Q4",
        "document_type": "earnings_call",
    },
    "INTC_2024_Q4_earnings_call.txt": {
        "company": "Intel",
        "ticker": "INTC",
        "year": 2024,
        "quarter": "Q4",
        "document_type": "earnings_call",
    },
    "TSMC_2024_Q4_earnings_call.txt": {
        "company": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSMC",
        "year": 2024,
        "quarter": "Q4",
        "document_type": "earnings_call",
    },
    "ASML_2024_Q4_earnings_call.pdf": {
        "company": "ASML",
        "ticker": "ASML",
        "year": 2024,
        "quarter": "Q4",
        "document_type": "earnings_call",
    },
    "NVDA_financial_facts_sec.csv": {
        "company": "NVIDIA",
        "ticker": "NVDA",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "AMD_financial_facts_sec.csv": {
        "company": "Advanced Micro Devices",
        "ticker": "AMD",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "INTC_financial_facts_sec.csv": {
        "company": "Intel",
        "ticker": "INTC",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "TSMC_financial_facts_sec.csv": {
        "company": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSMC",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "ASML_financial_facts_sec.csv": {
        "company": "ASML",
        "ticker": "ASML",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "semiconductor_financial_facts_sec.csv": {
        "company": "Semiconductor peer group",
        "ticker": "MULTI",
        "document_type": "financial_facts_csv",
        "source": "SEC Company Facts",
    },
    "NVDA_stock_prices_last_month.csv": {
        "company": "NVIDIA",
        "ticker": "NVDA",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
    "AMD_stock_prices_last_month.csv": {
        "company": "Advanced Micro Devices",
        "ticker": "AMD",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
    "INTC_stock_prices_last_month.csv": {
        "company": "Intel",
        "ticker": "INTC",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
    "TSMC_stock_prices_last_month.csv": {
        "company": "Taiwan Semiconductor Manufacturing Company",
        "ticker": "TSMC",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
    "ASML_stock_prices_last_month.csv": {
        "company": "ASML",
        "ticker": "ASML",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
    "semiconductor_stock_prices_last_month.csv": {
        "company": "Semiconductor peer group",
        "ticker": "MULTI",
        "document_type": "stock_prices_csv",
        "source": "yfinance",
    },
}


def extract_pages_from_pdf(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_idx + 1,
                "text": text.strip(),
                "content_type": "text",
            })

    return pages


def extract_text_file(text_path: Path):
    text = text_path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{
        "page": 1,
        "text": text,
        "content_type": "text",
    }]


def clean_table_cell(cell):
    if cell is None:
        return ""

    return " ".join(str(cell).replace("\n", " ").split())


def table_to_markdown(rows):
    cleaned_rows = [
        [clean_table_cell(cell) for cell in row]
        for row in rows
        if row and any(clean_table_cell(cell) for cell in row)
    ]

    if not cleaned_rows:
        return ""

    max_columns = max(len(row) for row in cleaned_rows)
    normalized_rows = [
        row + [""] * (max_columns - len(row))
        for row in cleaned_rows
    ]

    header = normalized_rows[0]
    body = normalized_rows[1:]

    if not any(header):
        header = [f"Column {index + 1}" for index in range(max_columns)]
        body = normalized_rows

    separator = ["---"] * max_columns
    markdown_rows = [header, separator, *body]

    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in markdown_rows
    )


def extract_table_file(table_path: Path):
    delimiter = "\t" if table_path.suffix.lower() == ".tsv" else ","

    with table_path.open("r", encoding="utf-8", errors="ignore", newline="") as table_file:
        rows = list(csv.reader(table_file, delimiter=delimiter))

    text = table_to_markdown(rows)

    if not text:
        return []

    return [{
        "page": 1,
        "text": text,
        "content_type": "table",
        "table_id": 0,
    }]


def extract_tables_from_pdf(pdf_path: Path):
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for PDF table extraction. "
            "Install dependencies with: .venv/bin/pip install -r requirements.txt"
        ) from exc

    tables = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            for table_idx, table in enumerate(page.extract_tables() or []):
                text = table_to_markdown(table)

                if text:
                    tables.append({
                        "page": page_idx + 1,
                        "text": text,
                        "content_type": "table",
                        "table_id": table_idx,
                    })

    return tables


def extract_items(source_path: Path, include_tables: bool = False):
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        items = extract_pages_from_pdf(source_path)

        if include_tables:
            items.extend(extract_tables_from_pdf(source_path))

        return items

    if suffix in {".txt", ".md"}:
        return extract_text_file(source_path)

    if suffix in {".csv", ".tsv"}:
        return extract_table_file(source_path)

    raise ValueError(f"Unsupported file type: {source_path}")


def build_documents(
    raw_dir: Path = RAW_DIR,
    chunk_method: str = "simple",
    chunk_size: int = 1000,
    overlap: int = 200,
    include_tables: bool = False,
):
    documents = []

    source_files = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for source_path in tqdm(source_files, desc="Reading source files"):
        filename = source_path.name
        source_type = source_path.suffix.lower().lstrip(".")
        base_metadata = DOCUMENT_METADATA.get(filename, {})

        items = extract_items(source_path, include_tables=include_tables)

        for item in items:
            chunks = chunk_text(
                item["text"],
                chunk_size=chunk_size,
                overlap=overlap,
                method=chunk_method,
            )

            for chunk_id, chunk in enumerate(chunks):
                metadata = {
                    **base_metadata,
                    "source_file": filename,
                    "source_type": source_type,
                    "content_type": item["content_type"],
                    "page": item["page"],
                    "chunk_id": chunk_id,
                    "chunking_method": chunk_method,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                }

                if "table_id" in item:
                    metadata["table_id"] = item["table_id"]

                documents.append({
                    "text": chunk,
                    "metadata": metadata,
                })

    return documents

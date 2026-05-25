from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream,*/*",
}

REPORTS = {
    "NVDA_2025_10K.pdf": [
        "https://s201.q4cdn.com/141608511/files/doc_financials/2025/q4/177440d5-3b32-4185-8cc8-95500a9dc783.pdf",
    ],
    "AMD_2024_10K.pdf": [
        "https://ir.amd.com/financial-information/sec-filings/content/0001193125-25-067185/0001193125-25-067185.pdf",
    ],
    "INTC_2024_annual_report.pdf": [
        "https://www.intc.com/filings-reports/all-sec-filings/content/0000050863-25-000052/a2024arsform10-k.pdf",
    ],
    "TSMC_2024_annual_report.pdf": [
        "https://investor.tsmc.com/sites/ir/annual-report/2024/2024%20Annual%20Report-E.pdf",
        "https://www.annualreports.com/HostedData/AnnualReports/PDF/NYSE_TSM_2024.pdf",
    ],
    "ASML_2024_annual_report.pdf": [
        "https://ourbrand.asml.com/m/79d325b168e0fd7e/original/2024-Annual-Report-based-on-US-GAAP.pdf",
    ],
}


def download_file(session: requests.Session, filename: str, urls: list[str]):
    output_path = RAW_DIR / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists():
        print(f"[SKIP] {filename} already exists")
        return

    last_error = None

    for url in urls:
        print(f"[DOWNLOAD] {filename}")
        try:
            with session.get(url, timeout=60, stream=True) as response:
                response.raise_for_status()

                with temp_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)

            if filename.endswith(".pdf") and not temp_path.read_bytes().startswith(b"%PDF-"):
                raise ValueError("downloaded content is not a PDF")

            temp_path.replace(output_path)
            print(f"[OK] saved to {output_path}")
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            print(f"[WARN] failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {filename}") from last_error

def main():
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    for filename, urls in REPORTS.items():
        download_file(session, filename, urls)

if __name__ == "__main__":
    main()

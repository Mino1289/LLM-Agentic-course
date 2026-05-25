from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlparse

import requests


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,application/octet-stream,*/*",
}

DEFAULT_END_MARKERS = [
    "This article is a transcript",
    "Stocks Mentioned",
    "Premium Investing Services",
]

EARNINGS_CALLS = {
    "NVDA_2025_Q4_earnings_call.txt": {
        "format": "html",
        "title": "NVIDIA Q4 FY2025 earnings call",
        "source": "The Motley Fool",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2025/02/26/nvidia-nvda-q4-2025-earnings-call-transcript/",
        ],
        "start_marker": "Prepared Remarks:",
    },
    "AMD_2024_Q4_earnings_call.txt": {
        "format": "html",
        "title": "AMD Q4 2024 earnings call",
        "source": "The Motley Fool",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2025/02/05/advanced-micro-devices-amd-q4-2024-earnings-call-t/",
        ],
        "start_marker": "Prepared Remarks:",
    },
    "INTC_2024_Q4_earnings_call.txt": {
        "format": "html",
        "title": "Intel Q4 2024 earnings call",
        "source": "The Motley Fool",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2025/01/30/intel-intc-q4-2024-earnings-call-transcript/",
        ],
        "start_marker": "Prepared Remarks:",
    },
    "TSMC_2024_Q4_earnings_call.txt": {
        "format": "html",
        "title": "TSMC Q4 2024 earnings call",
        "source": "The Motley Fool",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2025/01/16/taiwan-semiconductor-manufacturing-tsm-q4-2024-ear/",
        ],
        "start_marker": "Prepared Remarks:",
    },
    "ASML_2024_Q4_earnings_call.pdf": {
        "format": "pdf",
        "title": "ASML Q4 and FY 2024 investor call transcript",
        "source": "ASML",
        "urls": [
            "https://edge.sitecorecloud.io/asmlnetherlaaea-asmlcom-prd-5369/media/project/asmlcom/asmlcom/asml/files/investors/financial-results/q-results/2024/q4/20250129-asml-transcript-investor-call-q4--fy-2024.pdf",
        ],
    },
}


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "main",
        "p",
        "section",
        "ul",
    }
    SKIP_TAGS = {"script", "style", "svg", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
            return

        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str):
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def crop_transcript(text: str, start_marker: str | None, end_markers: list[str]):
    if start_marker:
        start = text.find(start_marker)
        next_start = text.find(start_marker, start + len(start_marker)) if start != -1 else -1

        if next_start != -1:
            start = next_start

        if start != -1:
            text = text[start:]

    end_candidates = [
        text.find(marker)
        for marker in end_markers
        if text.find(marker) != -1
    ]

    if end_candidates:
        text = text[:min(end_candidates)]

    return text.strip()


def build_header(config: dict, url: str):
    return "\n".join([
        f"Title: {config['title']}",
        f"Source: {config['source']}",
        f"Source URL: {url}",
        "",
    ])


def download_pdf(session: requests.Session, filename: str, config: dict):
    output_path = RAW_DIR / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists():
        print(f"[SKIP] {filename} already exists")
        return

    last_error = None

    for url in config["urls"]:
        print(f"[DOWNLOAD] {filename}")
        try:
            with session.get(url, timeout=60, stream=True) as response:
                response.raise_for_status()

                with temp_path.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)

            if not temp_path.read_bytes().startswith(b"%PDF-"):
                raise ValueError("downloaded content is not a PDF")

            temp_path.replace(output_path)
            print(f"[OK] saved to {output_path}")
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            print(f"[WARN] failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {filename}") from last_error


def download_html_transcript(session: requests.Session, filename: str, config: dict):
    output_path = RAW_DIR / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists():
        print(f"[SKIP] {filename} already exists")
        return

    last_error = None

    for url in config["urls"]:
        print(f"[DOWNLOAD] {filename}")
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()

            text = html_to_text(response.text)
            text = crop_transcript(
                text,
                start_marker=config.get("start_marker"),
                end_markers=config.get("end_markers", DEFAULT_END_MARKERS),
            )

            if len(text) < 1000:
                domain = urlparse(url).netloc
                raise ValueError(f"extracted transcript from {domain} is too short")

            temp_path.write_text(build_header(config, url) + text + "\n", encoding="utf-8")
            temp_path.replace(output_path)
            print(f"[OK] saved to {output_path}")
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            print(f"[WARN] failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {filename}") from last_error


def download_earnings_call(session: requests.Session, filename: str, config: dict):
    if config["format"] == "pdf":
        download_pdf(session, filename, config)
        return

    if config["format"] == "html":
        download_html_transcript(session, filename, config)
        return

    raise ValueError(f"Unsupported earnings call format: {config['format']}")


def main():
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    for filename, config in EARNINGS_CALLS.items():
        download_earnings_call(session, filename, config)


if __name__ == "__main__":
    main()

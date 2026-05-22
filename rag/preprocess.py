import os
import re
from bs4 import BeautifulSoup
import glob
import json

# Get current directory of this file to define absolute paths relative to it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")


def clean_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(text):
    # Very basic regex to try to find Item 1A, Item 7, and Item 8
    # 10-K texts are notoriously messy, so this tries a general heuristic

    sections = {}

    # Regex to find Item 1A. Risk Factors until Item 1B
    item_1a_match = re.search(
        r"ITEM\s+1A\.\s+RISK\s+FACTORS(.*?)(?=ITEM\s+1B\.)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if item_1a_match:
        sections["Item_1A"] = item_1a_match.group(1)

    # Regex to find Item 7. Management's Discussion and Analysis until Item 7A
    item_7_match = re.search(
        r"ITEM\s+7\.\s+MANAGEMENT[\'’]S\s+DISCUSSION(.*?)(?=ITEM\s+7A\.)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if item_7_match:
        sections["Item_7"] = item_7_match.group(1)

    # Regex to find Item 8. Financial Statements until Item 9
    item_8_match = re.search(
        r"ITEM\s+8\.\s+FINANCIAL\s+STATEMENTS(.*?)(?=ITEM\s+9\.)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if item_8_match:
        sections["Item_8"] = item_8_match.group(1)

    return sections


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(DATA_DIR, "*.html")) + glob.glob(
        os.path.join(DATA_DIR, "*.htm")
    )

    print(f"Found {len(files)} 10-K files.")

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        try:
            # Try utf-8 first, fallback to latin-1
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()

            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = clean_text(text)

            sections = extract_sections(text)

            if not sections:
                print(
                    f"  Warning: No matched sections found for {filename} with basic regex. May need specific parsing."
                )

            out_file = os.path.join(OUTPUT_DIR, filename + ".json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")


if __name__ == "__main__":
    main()

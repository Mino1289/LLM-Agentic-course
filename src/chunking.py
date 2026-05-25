import re


CHUNKING_METHODS = ("simple", "paragraph")


def validate_chunking_args(chunk_size: int, overlap: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def chunk_text_simple(text: str, chunk_size: int = 1000, overlap: int = 200):
    validate_chunking_args(chunk_size, overlap)

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def split_paragraphs(text: str):
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", normalized_text)
        if paragraph.strip()
    ]


def overlap_tail(paragraphs: list[str], overlap: int):
    if overlap == 0:
        return []

    tail = []
    tail_length = 0

    for paragraph in reversed(paragraphs):
        separator_length = 2 if tail else 0
        next_length = tail_length + separator_length + len(paragraph)

        if next_length > overlap:
            break

        tail.insert(0, paragraph)
        tail_length = next_length

        if tail_length >= overlap:
            break

    return tail


def chunk_text_by_paragraph(text: str, chunk_size: int = 1000, overlap: int = 200):
    validate_chunking_args(chunk_size, overlap)

    paragraphs = split_paragraphs(text)

    if len(paragraphs) <= 1:
        return chunk_text_simple(text, chunk_size=chunk_size, overlap=overlap)

    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs).strip())
                current_paragraphs = overlap_tail(current_paragraphs, overlap)
                current_length = len("\n\n".join(current_paragraphs))

            chunks.extend(chunk_text_simple(paragraph, chunk_size=chunk_size, overlap=overlap))
            current_paragraphs = []
            current_length = 0
            continue

        separator_length = 2 if current_paragraphs else 0
        next_length = current_length + separator_length + len(paragraph)

        if current_paragraphs and next_length > chunk_size:
            chunks.append("\n\n".join(current_paragraphs).strip())
            current_paragraphs = overlap_tail(current_paragraphs, overlap)
            current_length = len("\n\n".join(current_paragraphs))
            separator_length = 2 if current_paragraphs else 0

            if current_length + separator_length + len(paragraph) > chunk_size:
                current_paragraphs = []
                current_length = 0
                separator_length = 0

        current_paragraphs.append(paragraph)
        current_length += separator_length + len(paragraph)

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs).strip())

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    method: str = "simple",
):
    if method == "simple":
        return chunk_text_simple(text, chunk_size=chunk_size, overlap=overlap)

    if method == "paragraph":
        return chunk_text_by_paragraph(text, chunk_size=chunk_size, overlap=overlap)

    raise ValueError(f"Unsupported chunking method: {method}")

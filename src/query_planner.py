import json
import os
import re

import requests

from env_config import load_env_file


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_QUERY_PLANNER_MODEL = "gpt-4o-mini"
DEFAULT_QUERY_PLANNER_TEMPERATURE = 0.0
DEFAULT_QUERY_PLANNER_TIMEOUT = 60

COMPANY_CONTEXT = (
    "Companies in the corpus: NVIDIA (NVDA), Advanced Micro Devices (AMD), "
    "Intel (INTC), Taiwan Semiconductor Manufacturing Company (TSMC/TSM), "
    "and ASML. Document types include annual reports, 10-K filings, earnings "
    "call transcripts, SEC financial facts CSV files, and stock price CSV files."
)


class QueryPlanningError(RuntimeError):
    pass


def _extract_json_object(text: str):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise QueryPlanningError("The query planner did not return JSON.")

    return text[start:end + 1]


def _normalize_query_item(item):
    if isinstance(item, str):
        query = item
        purpose = ""
    elif isinstance(item, dict):
        query = item.get("query", "")
        purpose = item.get("purpose", "")
    else:
        return None

    query = " ".join(str(query).split())
    purpose = " ".join(str(purpose).split())

    if not query:
        return None

    return {
        "query": query,
        "purpose": purpose,
    }


def parse_query_plan(content: str, max_queries: int):
    try:
        payload = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise QueryPlanningError("The query planner returned invalid JSON.") from exc

    raw_queries = payload.get("queries")

    if not isinstance(raw_queries, list):
        raise QueryPlanningError("The query planner JSON must contain a queries list.")

    queries = []
    seen = set()

    for item in raw_queries:
        normalized = _normalize_query_item(item)

        if not normalized:
            continue

        query_key = normalized["query"].lower()

        if query_key in seen:
            continue

        seen.add(query_key)
        queries.append(normalized)

        if len(queries) >= max_queries:
            break

    if not queries:
        raise QueryPlanningError("The query planner returned no usable queries.")

    return queries


def build_planner_messages(question: str, max_queries: int):
    system_prompt = (
        "You are a retrieval query planner for a RAG system. "
        "Your only job is to rewrite a user question into focused search queries. "
        "Do not answer the question. Do not add facts that are not implied by the question. "
        "Prefer English because the corpus is mostly English. "
        "Make each query standalone and precise. "
        f"Return at most {max_queries} queries. "
        "Return JSON only with this shape: "
        '{"queries":[{"query":"...","purpose":"..."}]}. '
        f"{COMPANY_CONTEXT}"
    )
    user_prompt = (
        "Rewrite this user question into focused retrieval queries for the RAG corpus.\n\n"
        f"Question: {question}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _float_from_env(name: str, default: float):
    value = os.environ.get(name)

    if value is None or value == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise QueryPlanningError(f"{name} must be a float.") from exc


def plan_queries_with_openai(
    question: str,
    max_queries: int,
    model: str | None = None,
    temperature: float | None = None,
):
    load_env_file()

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise QueryPlanningError(
            "OPENAI_API_KEY is required for --decompose. "
            "Add it to .env or export it in the shell."
        )

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")
    model = model or os.environ.get(
        "OPENAI_QUERY_PLANNER_MODEL",
        DEFAULT_QUERY_PLANNER_MODEL,
    )

    if temperature is None:
        temperature = _float_from_env(
            "OPENAI_QUERY_PLANNER_TEMPERATURE",
            DEFAULT_QUERY_PLANNER_TEMPERATURE,
        )

    payload = {
        "model": model,
        "messages": build_planner_messages(question, max_queries),
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=DEFAULT_QUERY_PLANNER_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise QueryPlanningError(f"Query planner request failed: {exc}") from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise QueryPlanningError("Query planner returned a non-JSON response.") from exc

    if response.status_code >= 400:
        error = response_payload.get("error", {})
        message = error.get("message") or response.text
        raise QueryPlanningError(f"Query planner API error: {message}")

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QueryPlanningError("Unexpected query planner API response.") from exc

    return parse_query_plan(content, max_queries=max_queries)

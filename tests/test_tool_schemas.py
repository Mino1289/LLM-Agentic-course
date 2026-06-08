"""Pydantic BaseModel validation for tool args (PRD etape 4 §3.1)."""

import unittest

from pydantic import ValidationError


class SecFilingsRAGArgsTests(unittest.TestCase):
    def test_rejects_missing_query(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        with self.assertRaises(ValidationError):
            SecFilingsRAGArgs.model_validate({})

    def test_accepts_string_list_for_tickers(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        args = SecFilingsRAGArgs.model_validate(
            {"query": "NVDA risk", "tickers": "MSFT,NVDA"}
        )
        self.assertEqual(args.tickers, ["MSFT", "NVDA"])

    def test_defaults_optional_fields(self) -> None:
        from rag.tool_schemas import SecFilingsRAGArgs

        args = SecFilingsRAGArgs.model_validate({"query": "MSFT risk"})
        self.assertIsNone(args.tickers)
        self.assertIsNone(args.years)
        self.assertIsNone(args.doc_types)


class MarketPriceArgsTests(unittest.TestCase):
    def test_validates_date_format(self) -> None:
        from rag.tool_schemas import MarketPriceArgs

        args = MarketPriceArgs.model_validate(
            {"tickers": ["NVDA"], "start_date": "2024-01-01", "end_date": "2024-12-31"}
        )
        self.assertEqual(args.start_date, "2024-01-01")

        with self.assertRaises(ValidationError):
            MarketPriceArgs.model_validate(
                {"tickers": ["NVDA"], "start_date": "01/01/2024", "end_date": "2024-12-31"}
            )

    def test_requires_at_least_one_ticker(self) -> None:
        from rag.tool_schemas import MarketPriceArgs

        with self.assertRaises(ValidationError):
            MarketPriceArgs.model_validate(
                {"tickers": [], "start_date": "2024-01-01", "end_date": "2024-12-31"}
            )


class ValidateClaimsLLMArgsTests(unittest.TestCase):
    def test_excludes_chunks_from_schema(self) -> None:
        from rag.tool_schemas import ValidateClaimsLLMArgs

        schema = ValidateClaimsLLMArgs.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        self.assertEqual(props, {"claims"})
        self.assertNotIn("chunks", props)
        self.assertNotIn("metadatas", props)

    def test_requires_non_empty_claims(self) -> None:
        from rag.tool_schemas import ValidateClaimsLLMArgs

        with self.assertRaises(ValidationError):
            ValidateClaimsLLMArgs.model_validate({"claims": []})


class ValidateClaimsArgsTests(unittest.TestCase):
    def test_accepts_explicit_chunks_and_metadatas(self) -> None:
        from rag.tool_schemas import ValidateClaimsArgs

        args = ValidateClaimsArgs.model_validate(
            {
                "claims": ["MSFT risk"],
                "chunks": ["Item 1A risk factors."],
                "metadatas": [{"ticker": "MSFT", "year": "2024"}],
            }
        )
        self.assertEqual(len(args.chunks), 1)
        self.assertEqual(args.metadatas[0]["ticker"], "MSFT")

    def test_defaults_chunks_and_metadatas_to_empty(self) -> None:
        from rag.tool_schemas import ValidateClaimsArgs

        args = ValidateClaimsArgs.model_validate({"claims": ["claim 1"]})
        self.assertEqual(args.chunks, [])
        self.assertEqual(args.metadatas, [])


class SimulatePortfolioArgsTests(unittest.TestCase):
    def test_caps_notional(self) -> None:
        from rag.tool_schemas import SimulatePortfolioArgs

        with self.assertRaises(ValidationError):
            SimulatePortfolioArgs.model_validate(
                {"allocations": {"MSFT": 100.0}, "notional_usd": 2_000_000}
            )

    def test_default_notional(self) -> None:
        from rag.tool_schemas import SimulatePortfolioArgs

        args = SimulatePortfolioArgs.model_validate({"allocations": {"MSFT": 100.0}})
        self.assertEqual(args.notional_usd, 100_000)


class ExportReportArgsTests(unittest.TestCase):
    def test_default_format_is_md(self) -> None:
        from rag.tool_schemas import ExportReportArgs

        args = ExportReportArgs.model_validate(
            {"title": "Report", "content": "Body"}
        )
        self.assertEqual(args.format, "md")

    def test_rejects_unknown_format(self) -> None:
        from rag.tool_schemas import ExportReportArgs

        with self.assertRaises(ValidationError):
            ExportReportArgs.model_validate(
                {"title": "Report", "content": "Body", "format": "html"}
            )

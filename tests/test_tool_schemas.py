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


class PortfolioInfoArgsTests(unittest.TestCase):
    def test_accepts_empty(self) -> None:
        from rag.tool_schemas import PortfolioInfoArgs

        args = PortfolioInfoArgs.model_validate({})
        self.assertIsInstance(args, PortfolioInfoArgs)


class PlaceTradeArgsTests(unittest.TestCase):
    def test_requires_ticker_side_qty(self) -> None:
        from rag.tool_schemas import PlaceTradeArgs

        args = PlaceTradeArgs.model_validate({"ticker": "NVDA", "side": "buy", "qty": 10})
        self.assertEqual(args.ticker, "NVDA")
        self.assertEqual(args.side, "buy")
        self.assertEqual(args.qty, 10)
        self.assertEqual(args.order_type, "market")

    def test_rejects_invalid_side(self) -> None:
        from rag.tool_schemas import PlaceTradeArgs

        with self.assertRaises(ValidationError):
            PlaceTradeArgs.model_validate({"ticker": "NVDA", "side": "hold", "qty": 10})

    def test_rejects_negative_qty(self) -> None:
        from rag.tool_schemas import PlaceTradeArgs

        with self.assertRaises(ValidationError):
            PlaceTradeArgs.model_validate({"ticker": "NVDA", "side": "buy", "qty": -1})


class ClosePositionArgsTests(unittest.TestCase):
    def test_either_ticker_or_all(self) -> None:
        from rag.tool_schemas import ClosePositionArgs

        by_ticker = ClosePositionArgs.model_validate({"ticker": "NVDA"})
        self.assertEqual(by_ticker.ticker, "NVDA")
        self.assertFalse(by_ticker.all)

        all_pos = ClosePositionArgs.model_validate({"all": True})
        self.assertTrue(all_pos.all)
        self.assertIsNone(all_pos.ticker)


class GetNewsArgsTests(unittest.TestCase):
    def test_requires_symbols(self) -> None:
        from rag.tool_schemas import GetNewsArgs

        with self.assertRaises(ValidationError):
            GetNewsArgs.model_validate({})

    def test_defaults_limit(self) -> None:
        from rag.tool_schemas import GetNewsArgs

        args = GetNewsArgs.model_validate({"symbols": ["NVDA"]})
        self.assertEqual(args.limit, 10)
        self.assertFalse(args.include_content)

    def test_rejects_limit_over_50(self) -> None:
        from rag.tool_schemas import GetNewsArgs

        with self.assertRaises(ValidationError):
            GetNewsArgs.model_validate({"symbols": ["NVDA"], "limit": 100})


class PortfolioHistoryArgsTests(unittest.TestCase):
    def test_default_period_is_1m(self) -> None:
        from rag.tool_schemas import PortfolioHistoryArgs

        args = PortfolioHistoryArgs.model_validate({})
        self.assertEqual(args.period, "1M")
        self.assertFalse(args.extended_hours)


class AccountActivityArgsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        from rag.tool_schemas import AccountActivityArgs

        args = AccountActivityArgs.model_validate({})
        self.assertEqual(args.page_size, 20)
        self.assertEqual(args.direction, "desc")

    def test_accepts_csv_string_for_types(self) -> None:
        from rag.tool_schemas import AccountActivityArgs

        args = AccountActivityArgs.model_validate({"activity_types": "FILL,DIV"})
        self.assertEqual(args.activity_types, ["FILL", "DIV"])

    def test_rejects_page_size_over_100(self) -> None:
        from rag.tool_schemas import AccountActivityArgs

        with self.assertRaises(ValidationError):
            AccountActivityArgs.model_validate({"page_size": 200})


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

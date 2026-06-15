import type { Locale } from "@/lib/types/chat";
import type { ToolDefinition } from "@/lib/types/chat";

const TOOLS_FR: ToolDefinition[] = [
  {
    name: "SEC_Filings",
    description:
      "Interroge les dépôts SEC 10-K, 10-Q et 8-K pour les tickers donnés. Extrait les sections Risk Factors, MD&A et Financial Statements.",
  },
  {
    name: "Price_History",
    description:
      "Récupère les prix historiques OHLCV avec fenêtre configurable. Calcule volatilité, beta et rendements sur période.",
  },
  {
    name: "Earnings_Transcripts",
    description:
      "Accède aux transcripts d'appels earnings avec extraction de sentiment et résumé par topic.",
  },
  {
    name: "Claim_Validator",
    description:
      "Vérifie une affirmation financière contre les données des filings, des prix et des sources externes.",
  },
  {
    name: "Allocation_Simulator",
    description:
      "Simule un portefeuille avec allocation personnalisée, calcule VaR, Sharpe ratio et drawdown maximum.",
  },
  {
    name: "Report_Generator",
    description:
      "Génère des rapports PDF ou Markdown à partir de l'analyse effectuée, avec table des matières et références.",
  },
];

const TOOLS_EN: ToolDefinition[] = [
  {
    name: "SEC_Filings",
    description:
      "Queries SEC 10-K, 10-Q, and 8-K filings for given tickers. Extracts Risk Factors, MD&A, and Financial Statements sections.",
  },
  {
    name: "Price_History",
    description:
      "Retrieves historical OHLCV prices with configurable window. Computes volatility, beta, and period returns.",
  },
  {
    name: "Earnings_Transcripts",
    description:
      "Accesses earnings call transcripts with sentiment extraction and topic summaries.",
  },
  {
    name: "Claim_Validator",
    description:
      "Verifies a financial claim against filing data, prices, and external sources.",
  },
  {
    name: "Allocation_Simulator",
    description:
      "Simulates a portfolio with custom allocation, computing VaR, Sharpe ratio, and maximum drawdown.",
  },
  {
    name: "Report_Generator",
    description:
      "Generates PDF or Markdown reports from completed analysis, with table of contents and references.",
  },
];

export function getTools(locale: Locale): ToolDefinition[] {
  return locale === "fr" ? TOOLS_FR : TOOLS_EN;
}

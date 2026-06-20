import type { ChatMessage, HistoryGroup, Locale } from "@/lib/types/chat";

export const DEMO_CONVERSATION_ID = "demo-msft-nvda";

export function getWelcomeMessage(locale: Locale): ChatMessage {
  if (locale === "fr") {
    return {
      id: "welcome",
      role: "assistant",
      content:
        "Bonjour. Je peux analyser **NVDA, ASML, AMD, ARM, MSFT, TSM, AVGO, INTC, QCOM, MU** — interroger les filings SEC, les transcripts earnings, valider des affirmations, simuler une allocation, récupérer les prix et générer des rapports.",
      timestamp: "10:23",
    };
  }

  return {
    id: "welcome",
    role: "assistant",
    content:
      "Hello. I can analyze **NVDA, ASML, AMD, ARM, MSFT, TSM, AVGO, INTC, QCOM, MU** — query SEC filings, earnings transcripts, validate claims, simulate allocation, fetch prices, and generate reports.",
    timestamp: "10:23",
  };
}

export function getDemoConversation(locale: Locale): ChatMessage[] {
  const welcome = getWelcomeMessage(locale);

  if (locale === "fr") {
    return [
      welcome,
      {
        id: "user-1",
        role: "user",
        content:
          "Compare MSFT et NVDA — risques SEC 2024, perf 6 mois, recommande un trade avec une allocation de 50 actions.",
        timestamp: "10:24",
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: [
          "Voici l'analyse comparative entre **MSFT** et **NVDA** basée sur les filings SEC 2024 et les performances des 6 derniers mois.",
          "Les risques identifiés dans les 10-K récents montrent que NVDA est plus exposé aux restrictions géopolitiques (export controls Chine), tandis que MSFT fait face à des risques réglementaires sur le cloud (antitrust EU). Côté performance, NVDA affiche +187% sur 6 mois vs +28% pour MSFT, mais avec une volatilité significativement plus élevée (beta 1.8 vs 0.9).",
          "Sur la base de cette analyse, je recommande un ordre **NVDA Buy** avec les paramètres suivants :",
        ].join("\n\n"),
        timestamp: "10:26",
        artifacts: {
          steps: [
            { id: "1", text: "Récupération des 10-K SEC pour MSFT et NVDA (fiscal year 2024)" },
            { id: "2", text: "Extraction des facteurs de risque — Risk Factors sections" },
            { id: "3", text: "Récupération des prix historiques via **Price_History** (6 mois)" },
            { id: "4", text: "Calcul de la volatilité et du beta relatif" },
            { id: "5", text: "Génération de la recommandation de trade" },
          ],
          sources: [
            {
              id: "s1",
              title: "NVDA 10-K FY2024 — Risk Factors",
              excerpt:
                '"Our business is subject to export control laws and regulations, including those administered by the Bureau of Industry and Security... Changes in U.S. export control policies, particularly with respect to China, could materially adversely affect our revenue..."',
              meta: "SEC EDGAR · Filed 2024-02-28 · Pages 15-18",
              ticker: "NVDA",
              section: "Risk Factors",
            },
            {
              id: "s2",
              title: "MSFT 10-K FY2024 — Risk Factors",
              excerpt:
                '"We face increasing scrutiny from regulatory authorities around the world regarding antitrust concerns, particularly with respect to our cloud computing services and practices in the European Union..."',
              meta: "SEC EDGAR · Filed 2024-07-30 · Pages 22-25",
              ticker: "MSFT",
              section: "Risk Factors",
            },
            {
              id: "s3",
              title: "NVDA Price History — 6 Month Range",
              excerpt:
                "Range: $475.18 — $1,024.00 · Change: +187.4% · Beta: 1.82 · Vol 30d: 42.6%",
              meta: "Price_History tool · Window: 180 days",
              ticker: "NVDA",
              section: "Financial Statements",
            },
          ],
          reports: [
            {
              id: "r1",
              name: "Analyse comparative MSFT-NVDA",
              size: "2.4 Mo",
              type: "pdf",
            },
            {
              id: "r2",
              name: "Rapport fiscal SEC 2024",
              size: "840 Ko",
              type: "md",
            },
          ],
          stats: [
            { id: "tokens", label: "Tokens utilisés", value: "8,241" },
            { id: "chunks", label: "Chunks RAG", value: "6" },
            { id: "iterations", label: "Itérations LLM", value: "3" },
            { id: "tools", label: "Outils appelés", value: "5" },
            { id: "latency", label: "Latence totale", value: "4.2s" },
            { id: "confidence", label: "Score confiance", value: "0.847" },
          ],
          trade: {
            ticker: "NVDA",
            side: "BUY",
            quantity: 50,
            orderType: "Limit",
            limitPrice: "$1,024.00",
            riskLevel: "medium",
            justification:
              "Le secteur des semi-conducteurs bénéficie d'un cycle d'investissement massif en infrastructure AI. NVDA domine le marché des GPU datacenter avec une part estimée à 80%. Les restrictions d'exportation vers la Chine représentent un risque mais sont partiellement compensées par la diversification géographique croissante. Le trailing P/E de 64x reste élevé mais justifié par la croissance attendue du revenu de +40% pour FY2025. La position limitée (50 actions) capte le potentiel de hausse tout en limitant l'exposition à ~$51K.",
          },
        },
      },
    ];
  }

  return [
    welcome,
    {
      id: "user-1",
      role: "user",
      content:
        "Compare MSFT and NVDA — 2024 SEC risks, 6-month perf, recommend a trade with a 50-share allocation.",
      timestamp: "10:24",
    },
    {
      id: "assistant-1",
      role: "assistant",
      content: [
        "Here is the comparative analysis between **MSFT** and **NVDA** based on 2024 SEC filings and the last 6 months of performance.",
        "Recent 10-K risk factors show NVDA is more exposed to geopolitical restrictions (China export controls), while MSFT faces regulatory cloud risks (EU antitrust). On performance, NVDA is up +187% over 6 months vs +28% for MSFT, but with significantly higher volatility (beta 1.8 vs 0.9).",
        "Based on this analysis, I recommend an **NVDA Buy** order with the following parameters:",
      ].join("\n\n"),
      timestamp: "10:26",
      artifacts: {
        steps: [
          { id: "1", text: "Retrieved SEC 10-K filings for MSFT and NVDA (fiscal year 2024)" },
          { id: "2", text: "Extracted risk factors — Risk Factors sections" },
          { id: "3", text: "Fetched historical prices via **Price_History** (6 months)" },
          { id: "4", text: "Computed relative volatility and beta" },
          { id: "5", text: "Generated trade recommendation" },
        ],
        sources: [
          {
            id: "s1",
            title: "NVDA 10-K FY2024 — Risk Factors",
            excerpt:
              '"Our business is subject to export control laws and regulations, including those administered by the Bureau of Industry and Security... Changes in U.S. export control policies, particularly with respect to China, could materially adversely affect our revenue..."',
            meta: "SEC EDGAR · Filed 2024-02-28 · Pages 15-18",
            ticker: "NVDA",
            section: "Risk Factors",
          },
          {
            id: "s2",
            title: "MSFT 10-K FY2024 — Risk Factors",
            excerpt:
              '"We face increasing scrutiny from regulatory authorities around the world regarding antitrust concerns, particularly with respect to our cloud computing services and practices in the European Union..."',
            meta: "SEC EDGAR · Filed 2024-07-30 · Pages 22-25",
            ticker: "MSFT",
            section: "Risk Factors",
          },
          {
            id: "s3",
            title: "NVDA Price History — 6 Month Range",
            excerpt:
              "Range: $475.18 — $1,024.00 · Change: +187.4% · Beta: 1.82 · Vol 30d: 42.6%",
            meta: "Price_History tool · Window: 180 days",
            ticker: "NVDA",
            section: "Financial Statements",
          },
        ],
        reports: [
          {
            id: "r1",
            name: "MSFT-NVDA Comparative Analysis",
            size: "2.4 MB",
            type: "pdf",
          },
          {
            id: "r2",
            name: "SEC Fiscal Report 2024",
            size: "840 KB",
            type: "md",
          },
        ],
        stats: [
          { id: "tokens", label: "Tokens used", value: "8,241" },
          { id: "chunks", label: "RAG chunks", value: "6" },
          { id: "iterations", label: "LLM iterations", value: "3" },
          { id: "tools", label: "Tools called", value: "5" },
          { id: "latency", label: "Total latency", value: "4.2s" },
          { id: "confidence", label: "Confidence score", value: "0.847" },
        ],
        trade: {
          ticker: "NVDA",
          side: "BUY",
          quantity: 50,
          orderType: "Limit",
          limitPrice: "$1,024.00",
          riskLevel: "medium",
          justification:
            "The semiconductor sector benefits from massive AI infrastructure investment. NVDA dominates the datacenter GPU market with an estimated 80% share. China export restrictions remain a risk but are partially offset by growing geographic diversification. A trailing P/E of 64x remains elevated but is justified by expected FY2025 revenue growth of +40%. The limited position (50 shares) captures upside potential while capping exposure at ~$51K.",
        },
      },
    },
  ];
}

export function getHistoryGroups(locale: Locale): HistoryGroup[] {
  if (locale === "fr") {
    return [
      {
        label: "Aujourd'hui",
        items: [
          { id: DEMO_CONVERSATION_ID, title: "Comparaison MSFT vs NVDA", active: true },
          { id: "sec-risks", title: "Risques SEC ARM 2024" },
          { id: "asml-perf", title: "Performance ASML 6 mois" },
        ],
      },
      {
        label: "Hier",
        items: [
          { id: "amd-earnings", title: "Rapport AMD earnings Q3" },
          { id: "nvda-claim", title: "Validation affirmation NVDA" },
        ],
      },
      {
        label: "Semaine dernière",
        items: [{ id: "nvda-alloc", title: "Allocation simulateur NVDA" }],
      },
    ];
  }

  return [
    {
      label: "Today",
      items: [
        { id: DEMO_CONVERSATION_ID, title: "MSFT vs NVDA Comparison", active: true },
        { id: "sec-risks", title: "ARM 2024 SEC Risks" },
        { id: "asml-perf", title: "ASML 6-Month Performance" },
      ],
    },
    {
      label: "Yesterday",
      items: [
        { id: "amd-earnings", title: "AMD Q3 Earnings Report" },
        { id: "nvda-claim", title: "NVDA Claim Validation" },
      ],
    },
    {
      label: "Last week",
      items: [{ id: "nvda-alloc", title: "NVDA Allocation Simulator" }],
    },
  ];
}

export function getConversationTitle(locale: Locale): string {
  return locale === "fr" ? "Comparaison MSFT vs NVDA" : "MSFT vs NVDA Comparison";
}

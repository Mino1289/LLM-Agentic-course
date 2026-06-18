"use client";

import type { MessageArtifacts } from "@/lib/types/chat";
import { ThoughtsRenderer } from "@/components/artifacts/ThoughtsRenderer";
import { SourcesRenderer } from "@/components/artifacts/SourcesRenderer";
import { ReportsRenderer } from "@/components/artifacts/ReportsRenderer";
import { StatsRenderer } from "@/components/artifacts/StatsRenderer";
import { PriceChartRenderer } from "@/components/artifacts/PriceChartRenderer";

interface ArtifactStackProps {
  artifacts: MessageArtifacts;
  labels: {
    thoughts: string;
    sources: string;
    reports: string;
    stats: string;
    priceCharts: string;
    pricePerformance: string;
    priceVolatility: string;
    priceDrawdown: string;
    priceCurrent: string;
    priceHigh: string;
    priceLow: string;
    pricePeriod: string;
    priceNoData: string;
    allTickers: string;
    allSections: string;
    download: string;
  };
}

export function ArtifactStack({ artifacts, labels }: ArtifactStackProps) {
  const hasArtifacts =
    artifacts.steps?.length ||
    artifacts.sources?.length ||
    artifacts.reports?.length ||
    artifacts.stats?.length ||
    artifacts.priceCharts?.length;

  if (!hasArtifacts) return null;

  const priceLabels = {
    performance: labels.pricePerformance,
    volatility: labels.priceVolatility,
    drawdown: labels.priceDrawdown,
    current: labels.priceCurrent,
    high: labels.priceHigh,
    low: labels.priceLow,
    period: labels.pricePeriod,
    noData: labels.priceNoData,
  };

  return (
    <div className="artifact-stack">
      {artifacts.priceCharts?.length ? (
        <PriceChartRenderer
          title={labels.priceCharts}
          charts={artifacts.priceCharts}
          labels={priceLabels}
        />
      ) : null}
      {artifacts.steps?.length ? (
        <ThoughtsRenderer title={labels.thoughts} steps={artifacts.steps} defaultOpen={false} />
      ) : null}
      {artifacts.sources?.length ? (
        <SourcesRenderer
          title={labels.sources}
          sources={artifacts.sources}
          allTickersLabel={labels.allTickers}
          allSectionsLabel={labels.allSections}
          defaultOpen={false}
        />
      ) : null}
      {artifacts.reports?.length ? (
        <ReportsRenderer
          title={labels.reports}
          reports={artifacts.reports}
          downloadLabel={labels.download}
        />
      ) : null}
      {artifacts.stats?.length ? (
        <StatsRenderer title={labels.stats} stats={artifacts.stats} />
      ) : null}
    </div>
  );
}

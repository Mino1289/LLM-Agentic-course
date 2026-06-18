"use client";

import type { PriceSeriesArtifact } from "@/lib/types/chat";
import { Accordion } from "@/components/ui/Accordion";

interface PriceChartRendererProps {
  title: string;
  charts: PriceSeriesArtifact[];
  labels: {
    performance: string;
    volatility: string;
    drawdown: string;
    current: string;
    high: string;
    low: string;
    period: string;
  };
  defaultOpen?: boolean;
}

function buildPath(
  points: { close: number }[],
  width: number,
  height: number,
  padding: number,
): string {
  if (points.length === 0) return "";
  const values = points.map((p) => p.close);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const innerW = width - padding * 2;
  const innerH = height - padding * 2;

  return points
    .map((point, index) => {
      const x = padding + (index / Math.max(points.length - 1, 1)) * innerW;
      const y = padding + innerH - ((point.close - min) / span) * innerH;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function formatPct(value: number | undefined, locale: string) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${value.toLocaleString(locale, { maximumFractionDigits: 2 })}%`;
}

function formatPrice(value: number | undefined, locale: string) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `$${value.toLocaleString(locale, { maximumFractionDigits: 2 })}`;
}

function PriceChartCard({
  chart,
  labels,
  locale,
}: {
  chart: PriceSeriesArtifact;
  labels: PriceChartRendererProps["labels"];
  locale: string;
}) {
  const stats = chart.stats;
  const path = buildPath(chart.points, 320, 120, 8);

  return (
    <div className="price-chart-card">
      <div className="price-chart-header">
        <div className="price-chart-ticker">{chart.ticker}</div>
        <div className="price-chart-period">
          {labels.period}: {chart.startDate} → {chart.endDate}
        </div>
      </div>
      <svg
        className="price-chart-svg"
        viewBox="0 0 320 120"
        role="img"
        aria-label={`${chart.ticker} price chart`}
      >
        <path d={path} className="price-chart-line" fill="none" />
      </svg>
      <div className="price-chart-stats">
        <div>
          <span className="price-chart-stat-label">{labels.performance}</span>
          <span>{formatPct(stats?.perfPct, locale)}</span>
        </div>
        <div>
          <span className="price-chart-stat-label">{labels.volatility}</span>
          <span>{formatPct(stats?.volAnnPct, locale)}</span>
        </div>
        <div>
          <span className="price-chart-stat-label">{labels.drawdown}</span>
          <span>{formatPct(stats?.maxDrawdownPct, locale)}</span>
        </div>
        <div>
          <span className="price-chart-stat-label">{labels.current}</span>
          <span>{formatPrice(stats?.closeLast, locale)}</span>
        </div>
        <div>
          <span className="price-chart-stat-label">{labels.high}</span>
          <span>{formatPrice(stats?.closeMax, locale)}</span>
        </div>
        <div>
          <span className="price-chart-stat-label">{labels.low}</span>
          <span>{formatPrice(stats?.closeMin, locale)}</span>
        </div>
      </div>
    </div>
  );
}

export function PriceChartRenderer({
  title,
  charts,
  labels,
  defaultOpen = true,
}: PriceChartRendererProps) {
  const locale = "fr-FR";

  return (
    <Accordion title={title} defaultOpen={defaultOpen}>
      <div className="price-chart-stack">
        {charts.map((chart) => (
          <PriceChartCard key={chart.id} chart={chart} labels={labels} locale={locale} />
        ))}
      </div>
    </Accordion>
  );
}

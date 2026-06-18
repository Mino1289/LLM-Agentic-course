"use client";

import type { PricePoint, PriceSeriesArtifact } from "@/lib/types/chat";
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
    noData?: string;
  };
  defaultOpen?: boolean;
}

const CHART_WIDTH = 320;
const CHART_HEIGHT = 120;
const CHART_PADDING = 8;

function normalizePoints(points: PricePoint[] | undefined): { date: string; close: number }[] {
  if (!points?.length) return [];
  return points
    .map((point, index) => ({
      date: point.date || `point-${index + 1}`,
      close: Number(point.close),
    }))
    .filter((point) => Number.isFinite(point.close));
}

function buildLinePath(
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

function buildAreaPath(
  points: { close: number }[],
  width: number,
  height: number,
  padding: number,
): string {
  const linePath = buildLinePath(points, width, height, padding);
  if (!linePath) return "";
  const innerW = width - padding * 2;
  const innerH = height - padding * 2;
  const baseline = padding + innerH;
  const lastX = padding + innerW;
  return `${linePath} L${lastX.toFixed(1)},${baseline.toFixed(1)} L${padding.toFixed(1)},${baseline.toFixed(1)} Z`;
}

function singlePointCoords(
  point: { close: number },
  width: number,
  height: number,
  padding: number,
): { x: number; y: number } {
  const innerW = width - padding * 2;
  const innerH = height - padding * 2;
  return {
    x: padding + innerW / 2,
    y: padding + innerH / 2,
  };
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
  const points = normalizePoints(chart.points);
  const linePath = buildLinePath(points, CHART_WIDTH, CHART_HEIGHT, CHART_PADDING);
  const areaPath = buildAreaPath(points, CHART_WIDTH, CHART_HEIGHT, CHART_PADDING);
  const noDataLabel = labels.noData ?? "Données insuffisantes pour tracer le graphique";
  const singlePoint =
    points.length === 1
      ? singlePointCoords(points[0], CHART_WIDTH, CHART_HEIGHT, CHART_PADDING)
      : null;

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
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        role="img"
        aria-label={`${chart.ticker} price chart`}
      >
        {points.length >= 2 && areaPath ? (
          <path d={areaPath} className="price-chart-area" />
        ) : null}
        {points.length >= 2 && linePath ? (
          <path
            d={linePath}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ) : null}
        {singlePoint ? (
          <circle cx={singlePoint.x} cy={singlePoint.y} r={4} fill="var(--accent)" />
        ) : null}
        {points.length === 0 ? (
          <text
            x={CHART_WIDTH / 2}
            y={CHART_HEIGHT / 2}
            textAnchor="middle"
            className="price-chart-empty"
          >
            {noDataLabel}
          </text>
        ) : null}
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

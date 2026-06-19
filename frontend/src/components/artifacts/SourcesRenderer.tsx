"use client";

import { useMemo, useState } from "react";
import type { SourceItem } from "@/lib/types/chat";
import { Accordion } from "@/components/ui/Accordion";

interface SourcesRendererProps {
  title: string;
  sources: SourceItem[];
  allTickersLabel: string;
  allSectionsLabel: string;
  defaultOpen?: boolean;
}

export function SourcesRenderer({
  title,
  sources,
  allTickersLabel,
  allSectionsLabel,
  defaultOpen = false,
}: SourcesRendererProps) {
  const [ticker, setTicker] = useState(allTickersLabel);
  const [section, setSection] = useState(allSectionsLabel);

  const tickers = useMemo(
    () => [allTickersLabel, ...Array.from(new Set(sources.map((s) => s.ticker))).sort()],
    [allTickersLabel, sources],
  );

  const sections = useMemo(
    () => [allSectionsLabel, ...Array.from(new Set(sources.map((s) => s.section))).sort()],
    [allSectionsLabel, sources],
  );

  const filtered = sources.filter((source) => {
    const tickerMatch = ticker === allTickersLabel || source.ticker === ticker;
    const sectionMatch = section === allSectionsLabel || source.section === section;
    return tickerMatch && sectionMatch;
  });

  return (
    <Accordion title={title} defaultOpen={defaultOpen}>
      <div className="source-filters">
        <select
          className="source-select"
          value={ticker}
          onChange={(event) => setTicker(event.target.value)}
        >
          {tickers.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select
          className="source-select"
          value={section}
          onChange={(event) => setSection(event.target.value)}
        >
          {sections.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </div>
      {filtered.map((source) => (
        <div key={source.id} className="source-card">
          <div className="source-card-title">{source.title}</div>
          <div className="source-card-excerpt">{source.excerpt}</div>
          <div className="source-card-meta">{source.meta}</div>
        </div>
      ))}
    </Accordion>
  );
}

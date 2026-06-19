"use client";

import type { StatItem } from "@/lib/types/chat";
import { Accordion } from "@/components/ui/Accordion";

interface StatsRendererProps {
  title: string;
  stats: StatItem[];
  defaultOpen?: boolean;
}

export function StatsRenderer({
  title,
  stats,
  defaultOpen = false,
}: StatsRendererProps) {
  return (
    <Accordion title={title} defaultOpen={defaultOpen}>
      <div className="stats-grid">
        {stats.map((stat) => (
          <div key={stat.id} className="stat-item">
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label-text">{stat.label}</div>
          </div>
        ))}
      </div>
    </Accordion>
  );
}

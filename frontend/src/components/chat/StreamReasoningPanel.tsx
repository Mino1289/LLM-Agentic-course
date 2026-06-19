"use client";

import type { ReasoningEntry } from "@/lib/agentPipeline";
import { PIPELINE_LABELS } from "@/lib/agentPipeline";
import type { AppLocale } from "@/i18n/routing";

interface StreamReasoningPanelProps {
  title: string;
  entries: ReasoningEntry[];
  locale: AppLocale;
}

function agentLabel(agent: string, locale: AppLocale): string {
  if (agent === "Outils") {
    return locale === "fr" ? "Outils" : "Tools";
  }
  if (agent === "Orchestration") {
    return locale === "fr" ? "Orchestration" : "Orchestration";
  }

  const map = PIPELINE_LABELS[locale];
  const byKey = Object.entries(map).find(([, label]) => label === agent);
  if (byKey) return byKey[1];

  const aliases: Record<string, keyof typeof map> = {
    "Intent Router": "intent_router",
    "Portfolio Manager": "pm_plan",
    "Fundamental Analyst": "fundamental",
    "Quantitative Analyst": "quantitative",
    "Compliance Validator": "compliance",
    "Human Review": "human_review",
    "Simple Agent": "simple_agent",
    Outils: "fundamental",
  };
  const id = aliases[agent];
  if (id) return map[id];
  return agent;
}

export function StreamReasoningPanel({
  title,
  entries,
  locale,
}: StreamReasoningPanelProps) {
  if (entries.length === 0) return null;

  return (
    <details className="stream-reasoning">
      <summary className="stream-reasoning-summary">{title}</summary>
      <ul className="stream-reasoning-list">
        {entries.map((entry) => (
          <li key={entry.id}>
            <em>
              {agentLabel(entry.agent, locale)} — {entry.text}
            </em>
          </li>
        ))}
      </ul>
    </details>
  );
}

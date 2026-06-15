import type { StreamPipelineState } from "@/lib/agentPipeline";
import {
  getVisiblePipelineAgents,
  PIPELINE_LABELS,
} from "@/lib/agentPipeline";
import type { AppLocale } from "@/i18n/routing";

interface AgentPipelinePanelProps {
  pipeline: StreamPipelineState;
  locale: AppLocale;
  modeLabel?: string;
}

export function AgentPipelinePanel({
  pipeline,
  locale,
  modeLabel,
}: AgentPipelinePanelProps) {
  const agents = getVisiblePipelineAgents(pipeline);
  const labels = PIPELINE_LABELS[locale];

  return (
    <div className="agent-pipeline">
      {modeLabel ? <div className="agent-pipeline-mode">{modeLabel}</div> : null}
      <ul className="agent-pipeline-list">
        {agents.map((id) => {
          const status = pipeline.agentStates[id];
          const isRunning = status === "running";
          return (
            <li key={id} className="agent-pipeline-row">
              <span
                className={`agent-led ${isRunning ? "agent-led--active" : "agent-led--idle"}`}
                aria-label={isRunning ? "En cours" : "Inactif"}
              />
              <span
                className={`agent-pipeline-name ${isRunning ? "agent-pipeline-name--active" : ""}`}
              >
                {labels[id]}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

import type { AppLocale } from "@/i18n/routing";

export type OrchestrationMode = "unknown" | "simple" | "multi";
export type AgentLedStatus = "pending" | "running" | "completed" | "failed";

export type PipelineAgentId =
  | "intent_router"
  | "pm_plan"
  | "pm_synthesis"
  | "fundamental"
  | "quantitative"
  | "compliance"
  | "human_review"
  | "simple_agent";

export interface ReasoningEntry {
  id: string;
  agent: string;
  text: string;
}

export interface StreamPipelineState {
  mode: OrchestrationMode;
  agentStates: Record<PipelineAgentId, AgentLedStatus>;
  reasoning: ReasoningEntry[];
  pmRunCount: number;
}

export const MULTI_PIPELINE_ORDER: PipelineAgentId[] = [
  "intent_router",
  "pm_plan",
  "pm_synthesis",
  "fundamental",
  "quantitative",
  "compliance",
  "human_review",
];

const MULTI_INITIAL: Record<PipelineAgentId, AgentLedStatus> = {
  intent_router: "pending",
  pm_plan: "pending",
  pm_synthesis: "pending",
  fundamental: "pending",
  quantitative: "pending",
  compliance: "pending",
  human_review: "pending",
  simple_agent: "pending",
};

export const PIPELINE_LABELS: Record<
  AppLocale,
  Record<PipelineAgentId, string>
> = {
  fr: {
    intent_router: "Routeur d'intention",
    pm_plan: "Portfolio Manager — Plan",
    pm_synthesis: "Portfolio Manager — Synthèse",
    fundamental: "Analyste Fundamental",
    quantitative: "Analyste Quantitatif",
    compliance: "Validateur Compliance",
    human_review: "Revue humaine",
    simple_agent: "Agent simple",
  },
  en: {
    intent_router: "Intent Router",
    pm_plan: "Portfolio Manager — Plan",
    pm_synthesis: "Portfolio Manager — Synthesis",
    fundamental: "Fundamental Analyst",
    quantitative: "Quantitative Analyst",
    compliance: "Compliance Validator",
    human_review: "Human Review",
    simple_agent: "Simple Agent",
  },
};

let reasoningCounter = 0;

function appendReasoning(
  state: StreamPipelineState,
  agent: string,
  text: string,
): ReasoningEntry[] {
  const trimmed = text.trim();
  if (!trimmed) return state.reasoning;
  const last = state.reasoning[state.reasoning.length - 1];
  if (last && last.agent === agent && last.text === trimmed) {
    return state.reasoning;
  }
  reasoningCounter += 1;
  return [
    ...state.reasoning,
    { id: `r-${reasoningCounter}`, agent, text: trimmed },
  ];
}

function resolvePipelineIds(
  agent: string,
  status: string,
  pmRunCount: number,
): { ids: PipelineAgentId[]; nextPmRunCount: number } {
  if (agent === "Simple Agent") {
    return { ids: ["simple_agent"], nextPmRunCount: pmRunCount };
  }

  if (agent === "Intent Router") {
    return { ids: ["intent_router"], nextPmRunCount: pmRunCount };
  }

  if (agent === "Portfolio Manager") {
    if (status === "running") {
      const next = pmRunCount + 1;
      return {
        ids: [next === 1 ? "pm_plan" : "pm_synthesis"],
        nextPmRunCount: next,
      };
    }
    const id: PipelineAgentId = pmRunCount <= 1 ? "pm_plan" : "pm_synthesis";
    return { ids: [id], nextPmRunCount: pmRunCount };
  }

  if (agent === "Analystes (Fundamental + Quant)") {
    return {
      ids: ["fundamental", "quantitative"],
      nextPmRunCount: pmRunCount,
    };
  }

  if (agent === "Fundamental Analyst") {
    return { ids: ["fundamental"], nextPmRunCount: pmRunCount };
  }

  if (agent === "Quantitative Analyst") {
    return { ids: ["quantitative"], nextPmRunCount: pmRunCount };
  }

  if (agent === "Compliance Validator") {
    return { ids: ["compliance"], nextPmRunCount: pmRunCount };
  }

  if (agent === "Human Review") {
    return { ids: ["human_review"], nextPmRunCount: pmRunCount };
  }

  return { ids: [], nextPmRunCount: pmRunCount };
}

function toLedStatus(eventStatus: string): AgentLedStatus {
  if (eventStatus === "running") return "running";
  if (eventStatus === "failed") return "failed";
  return "completed";
}

export function createInitialPipeline(): StreamPipelineState {
  return {
    mode: "unknown",
    agentStates: { ...MULTI_INITIAL },
    reasoning: [],
    pmRunCount: 0,
  };
}

export function applySpokeToPipeline(
  state: StreamPipelineState,
  agent: string,
  status: string,
  message: string,
): StreamPipelineState {
  let mode = state.mode;
  if (message.includes("Hub-and-Spoke") || message.includes("multi-agents")) {
    mode = "multi";
  } else if (message.includes("Agent simple") || message.includes("Simple agent")) {
    mode = "simple";
  }

  const ledStatus = toLedStatus(status);
  const { ids, nextPmRunCount } = resolvePipelineIds(agent, status, state.pmRunCount);
  const agentStates = { ...state.agentStates };

  for (const id of ids) {
    agentStates[id] = ledStatus;
  }

  return {
    mode,
    agentStates,
    reasoning: appendReasoning(state, agent, message),
    pmRunCount: nextPmRunCount,
  };
}

export function applyToolToPipeline(
  state: StreamPipelineState,
  toolName: string,
): StreamPipelineState {
  return {
    ...state,
    reasoning: appendReasoning(
      state,
      "Outils",
      `Exécution de ${toolName}...`,
    ),
  };
}

export function getVisiblePipelineAgents(
  state: StreamPipelineState,
): PipelineAgentId[] {
  if (state.mode === "simple") {
    return ["simple_agent"];
  }
  return MULTI_PIPELINE_ORDER;
}

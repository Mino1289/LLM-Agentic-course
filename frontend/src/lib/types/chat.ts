export type Locale = "fr" | "en";

export type RiskLevel = "low" | "medium" | "high";

export interface AgentStep {
  id: string;
  text: string;
}

export interface SourceItem {
  id: string;
  title: string;
  excerpt: string;
  meta: string;
  ticker: string;
  section: string;
}

export interface ReportArtifact {
  id: string;
  name: string;
  size: string;
  type: "pdf" | "md";
  downloadUrl?: string;
}

export interface StatItem {
  id: string;
  label: string;
  value: string;
}

export interface TradeProposal {
  ticker: string;
  side: string;
  quantity: number | string;
  orderType: string;
  limitPrice?: string;
  riskLevel: RiskLevel;
  justification: string;
  complianceVerdict?: string;
  complianceDetail?: string;
}

export interface MessageArtifacts {
  steps?: AgentStep[];
  sources?: SourceItem[];
  reports?: ReportArtifact[];
  stats?: StatItem[];
  trade?: TradeProposal;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  artifacts?: MessageArtifacts;
}

export interface HistoryItem {
  id: string;
  title: string;
  active?: boolean;
}

export interface HistoryGroup {
  label: string;
  items: HistoryItem[];
}

export interface ToolDefinition {
  name: string;
  description: string;
}

export interface ModelConfig {
  chatProvider: string;
  chatModel: string;
  embeddingProvider: string;
  embeddingModel: string;
}

export interface AgentSettings {
  maxChunks: number;
  subQueries: number;
  priceMaxDays: number;
  priceMaxPoints: number;
  priceMaxTickers: number;
  priceDefaultWindow: number;
  maxIterations: number;
}

export interface SliderFieldConfig {
  key: keyof AgentSettings;
  min: number;
  max: number;
  defaultValue: number;
}

export type TradeDecision = "pending" | "approved" | "cancelled";

import type { SliderFieldConfig } from "@/lib/types/chat";

export const DEFAULT_AGENT_SETTINGS = {
  maxChunks: 8,
  subQueries: 2,
  priceMaxDays: 180,
  priceMaxPoints: 40,
  priceMaxTickers: 3,
  priceDefaultWindow: 90,
  maxIterations: 6,
} as const;

export const SLIDER_FIELDS: SliderFieldConfig[] = [
  { key: "maxChunks", min: 4, max: 12, defaultValue: 8 },
  { key: "subQueries", min: 1, max: 8, defaultValue: 2 },
  { key: "priceMaxDays", min: 30, max: 365, defaultValue: 180 },
  { key: "priceMaxPoints", min: 10, max: 120, defaultValue: 40 },
  { key: "priceMaxTickers", min: 1, max: 5, defaultValue: 3 },
  { key: "priceDefaultWindow", min: 15, max: 180, defaultValue: 90 },
  { key: "maxIterations", min: 2, max: 10, defaultValue: 6 },
];


export const MODEL_CONFIG = {
  chatProvider: "OpenAI",
  chatModel: "GPT-4",
  embeddingProvider: "Cohere",
  embeddingModel: "embed-multilingual-v3.0",
} as const;

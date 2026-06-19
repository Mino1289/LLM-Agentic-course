import { apiFetch } from "@/lib/api/client";
import type { AgentSettings, ModelConfig, ToolDefinition } from "@/lib/types/chat";

export interface ConfigResponse {
  chat_provider: string;
  chat_model: string;
  embedding_provider: string;
  embedding_model: string;
  defaults: AgentSettings;
}

export interface ToolsResponse {
  tools: Array<{
    name: string;
    description: string;
    parameters?: Record<string, unknown>;
  }>;
}

export async function getConfig(): Promise<ConfigResponse> {
  return apiFetch<ConfigResponse>("/api/config");
}

export async function getTools(): Promise<ToolDefinition[]> {
  const data = await apiFetch<ToolsResponse>("/api/tools");
  return data.tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
  }));
}

export function configToModelConfig(config: ConfigResponse): ModelConfig {
  return {
    chatProvider: config.chat_provider,
    chatModel: config.chat_model,
    embeddingProvider: config.embedding_provider,
    embeddingModel: config.embedding_model,
  };
}

export async function getHealth(): Promise<{ status: string; rag_indexed: boolean }> {
  return apiFetch("/api/health");
}

import { apiFetch, apiUrl } from "@/lib/api/client";
import { isAbortError } from "@/lib/constants";
import type {
  AgentSettings,
  ChatMessage,
  MessageArtifacts,
  TradeProposal,
} from "@/lib/types/chat";

export interface ChatStreamRequest {
  conversation_id?: string;
  message: string;
  locale: string;
  settings: AgentSettings;
}

export interface ChatResponsePayload {
  conversation_id: string;
  run_id?: string;
  answer: string;
  human_review_pending: boolean;
  artifacts: MessageArtifacts;
  message?: ChatMessage;
}

export interface HumanReviewPayload {
  run_id: string;
  conversation_id: string;
  answer: string;
  trade: TradeProposal & {
    compliance_verdict?: string;
    compliance_detail?: string;
  };
  artifacts: MessageArtifacts;
}

export type StreamEvent =
  | { type: "token"; token: string }
  | { type: "spoke"; agent: string; status: string; message: string }
  | { type: "tool_start"; name: string }
  | { type: "tool_end"; name: string }
  | { type: "done"; payload: ChatResponsePayload }
  | { type: "human_review_required"; payload: HumanReviewPayload }
  | { type: "error"; message: string };

export interface StreamHandlers {
  onToken: (token: string) => void;
  onSpoke?: (event: Extract<StreamEvent, { type: "spoke" }>) => void;
  onToolStart?: (name: string) => void;
  onDone: (payload: ChatResponsePayload) => void;
  onHumanReview: (payload: HumanReviewPayload) => void;
  onError: (message: string) => void;
}

function parseSseBlock(block: string): StreamEvent | null {
  const dataLine = block
    .split("\n")
    .find((line) => line.startsWith("data:"));
  if (!dataLine) return null;
  const json = dataLine.slice(5).trim();
  if (!json) return null;
  return JSON.parse(json) as StreamEvent;
}

export async function streamChat(
  request: ChatStreamRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<string | undefined> {
  if (signal?.aborted) {
    return undefined;
  }

  try {
    const response = await fetch(apiUrl("/api/chat/stream"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      const text = await response.text();
      handlers.onError(text || `Stream failed: ${response.status}`);
      return undefined;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      handlers.onError("No response body");
      return undefined;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let conversationId: string | undefined;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() ?? "";

      for (const block of blocks) {
        const event = parseSseBlock(block);
        if (!event) continue;

        switch (event.type) {
          case "token":
            handlers.onToken(event.token);
            break;
          case "spoke":
            handlers.onSpoke?.(event);
            break;
          case "tool_start":
            handlers.onToolStart?.(event.name);
            break;
          case "done":
            conversationId = event.payload.conversation_id;
            handlers.onDone(event.payload);
            break;
          case "human_review_required":
            conversationId = event.payload.conversation_id;
            handlers.onHumanReview(event.payload);
            break;
          case "error":
            handlers.onError(event.message);
            break;
          default:
            break;
        }
      }
    }

    return conversationId;
  } catch (error) {
    if (isAbortError(error) || signal?.aborted) {
      return undefined;
    }
    handlers.onError(error instanceof Error ? error.message : "Stream failed");
    return undefined;
  }
}

export async function resumeTrade(
  runId: string,
  approved: boolean,
  locale: string,
): Promise<ChatResponsePayload> {
  return apiFetch<ChatResponsePayload>(
    `/api/chat/resume?locale=${encodeURIComponent(locale)}`,
    {
      method: "POST",
      body: JSON.stringify({ run_id: runId, approved }),
    },
  );
}

export function reportDownloadUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return apiUrl(path);
}

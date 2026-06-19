import { apiFetch } from "@/lib/api/client";
import type { AgentSettings, ChatMessage, HistoryGroup } from "@/lib/types/chat";

export interface ConversationDetail {
  id: string;
  title: string;
  messages: ChatMessage[];
  settings: AgentSettings;
}

export interface ConversationListResponse {
  groups: HistoryGroup[];
}

export async function createConversation(
  locale: string,
  settings?: AgentSettings,
): Promise<ConversationDetail> {
  return apiFetch<ConversationDetail>("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ locale, settings }),
  });
}

export async function listConversations(
  locale: string,
): Promise<ConversationListResponse> {
  return apiFetch<ConversationListResponse>(
    `/api/conversations?locale=${encodeURIComponent(locale)}`,
  );
}

export async function getConversation(
  conversationId: string,
): Promise<ConversationDetail> {
  return apiFetch<ConversationDetail>(`/api/conversations/${conversationId}`);
}

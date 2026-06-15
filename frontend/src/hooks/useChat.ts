"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AgentSettings,
  ChatMessage,
  HistoryGroup,
  MessageArtifacts,
} from "@/lib/types/chat";
import {
  applySpokeToPipeline,
  applyToolToPipeline,
  createInitialPipeline,
  type StreamPipelineState,
} from "@/lib/agentPipeline";
import {
  streamChat,
  type ChatResponsePayload,
  type HumanReviewPayload,
} from "@/lib/api/chat";
import {
  createConversation,
  listConversations,
  getConversation,
} from "@/lib/api/conversations";
import { formatTime } from "@/lib/utils";
import { isAbortError } from "@/lib/constants";
import type { AppLocale } from "@/i18n/routing";

function mapArtifacts(raw: MessageArtifacts | undefined): MessageArtifacts | undefined {
  if (!raw) return undefined;
  return raw;
}

function payloadToMessage(
  payload: ChatResponsePayload | HumanReviewPayload,
  fallbackContent = "",
  messageId?: string,
): ChatMessage {
  const msg = "message" in payload ? payload.message : undefined;
  if (msg) {
    const artifacts = mapArtifacts(msg.artifacts) ?? {};
    if ("trade" in payload && payload.trade) {
      artifacts.trade = payload.trade;
    }
    return {
      ...msg,
      artifacts,
    };
  }

  const artifacts = mapArtifacts(payload.artifacts) ?? {};
  if ("trade" in payload && payload.trade) {
    artifacts.trade = payload.trade;
  }

  return {
    id: messageId ?? `assistant-${Date.now()}`,
    role: "assistant",
    content: payload.answer || fallbackContent,
    timestamp: formatTime(),
    artifacts,
  };
}

export function useChat(
  locale: AppLocale,
  settings: AgentSettings,
  onHumanReview?: (runId: string, messageId: string) => void,
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationTitle, setConversationTitle] = useState("");
  const [historyGroups, setHistoryGroups] = useState<HistoryGroup[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [streamPipeline, setStreamPipeline] = useState<StreamPipelineState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);

  const clearStreamingState = useCallback(() => {
    setIsTyping(false);
    setStreamingMessageId(null);
    setStreamPipeline(null);
    streamingIdRef.current = null;
  }, []);

  const abortActiveStream = useCallback(() => {
    const controller = streamRef.current;
    if (!controller) return;
    streamRef.current = null;
    controller.abort();
    clearStreamingState();
  }, [clearStreamingState]);

  const refreshHistory = useCallback(async () => {
    try {
      const data = await listConversations(locale);
      setHistoryGroups(data.groups);
    } catch {
      setHistoryGroups([]);
    }
  }, [locale]);

  const initConversation = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const conv = await createConversation(locale, settings);
      setConversationId(conv.id);
      setConversationTitle(conv.title);
      setMessages(conv.messages);
      await refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversation");
    } finally {
      setIsLoading(false);
    }
  }, [locale, settings, refreshHistory]);

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (active) void initConversation();
    });
    return () => {
      active = false;
      abortActiveStream();
    };
  }, [locale, abortActiveStream]); // eslint-disable-line react-hooks/exhaustive-deps

  const resetConversation = useCallback(async () => {
    abortActiveStream();
    await initConversation();
  }, [abortActiveStream, initConversation]);

  const loadConversation = useCallback(
    async (id: string) => {
      abortActiveStream();
      try {
        const conv = await getConversation(id);
        setConversationId(conv.id);
        setConversationTitle(conv.title);
        setMessages(conv.messages);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load conversation");
      }
    },
    [abortActiveStream],
  );

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !conversationId || isTyping) return;

      setError(null);
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: trimmed,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsTyping(true);

      const assistantId = `assistant-${Date.now()}`;
      streamingIdRef.current = assistantId;
      setStreamingMessageId(assistantId);
      setStreamPipeline(createInitialPipeline());
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: formatTime(),
        },
      ]);

      abortActiveStream();
      const controller = new AbortController();
      streamRef.current = controller;

      try {
        await streamChat(
          {
            conversation_id: conversationId,
            message: trimmed,
            locale,
            settings,
          },
          {
            onToken: (token) => {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + token }
                    : msg,
                ),
              );
            },
            onSpoke: (event) => {
              setStreamPipeline((prev) =>
                prev
                  ? applySpokeToPipeline(
                      prev,
                      event.agent,
                      event.status,
                      event.message,
                    )
                  : prev,
              );
            },
            onToolStart: (name) => {
              setStreamPipeline((prev) =>
                prev ? applyToolToPipeline(prev, name) : prev,
              );
            },
            onDone: (payload) => {
              clearStreamingState();
              const finalMessage = payloadToMessage(payload, "", assistantId);
              setMessages((prev) =>
                prev.map((msg) => (msg.id === assistantId ? finalMessage : msg)),
              );
              setConversationId(payload.conversation_id);
              void refreshHistory();
            },
            onHumanReview: (payload) => {
              clearStreamingState();
              const finalMessage = payloadToMessage(payload, "", assistantId);
              setMessages((prev) =>
                prev.map((msg) => (msg.id === assistantId ? finalMessage : msg)),
              );
              onHumanReview?.(payload.run_id, assistantId);
              void refreshHistory();
            },
            onError: (message) => {
              clearStreamingState();
              setError(message);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? {
                        ...msg,
                        content: `Erreur: ${message}`,
                      }
                    : msg,
                ),
              );
            },
          },
          controller.signal,
        );
      } catch (error) {
        if (isAbortError(error)) {
          clearStreamingState();
          return;
        }
        throw error;
      } finally {
        if (streamRef.current === controller) {
          streamRef.current = null;
        }
      }
    },
    [
      conversationId,
      isTyping,
      locale,
      settings,
      refreshHistory,
      onHumanReview,
      abortActiveStream,
      clearStreamingState,
    ],
  );

  const appendAssistantMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateMessage = useCallback((messageId: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === messageId ? { ...msg, ...patch } : msg)),
    );
  }, []);

  return {
    messages,
    conversationId,
    conversationTitle,
    historyGroups,
    isTyping,
    streamingMessageId,
    streamPipeline,
    isLoading,
    error,
    resetConversation,
    sendMessage,
    loadConversation,
    appendAssistantMessage,
    updateMessage,
    refreshHistory,
  };
}

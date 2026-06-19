"use client";

import { useCallback, useState } from "react";
import type { MessageArtifacts, TradeDecision } from "@/lib/types/chat";
import { resumeTrade } from "@/lib/api/chat";
import { formatTime } from "@/lib/utils";
import type { AppLocale } from "@/i18n/routing";

function isPendingRunNotFound(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("Pending run not found");
}

export function useTradeApproval(locale: AppLocale) {
  const [runId, setRunId] = useState<string | null>(null);
  const [messageId, setMessageId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<Record<string, TradeDecision>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const registerReview = useCallback(
    (nextRunId: string, nextMessageId: string, nextConversationId: string) => {
      setRunId(nextRunId);
      setMessageId(nextMessageId);
      setConversationId(nextConversationId);
      setError(null);
      setDecisions((prev) => ({ ...prev, [nextMessageId]: "pending" }));
    },
    [],
  );

  const reset = useCallback(() => {
    setRunId(null);
    setMessageId(null);
    setConversationId(null);
    setDecisions({});
    setIsSubmitting(false);
    setError(null);
  }, []);

  const getDecision = useCallback(
    (targetMessageId: string): TradeDecision =>
      decisions[targetMessageId] ?? "approved",
    [decisions],
  );

  const submitDecision = useCallback(
    async (approved: boolean) => {
      if (!runId || !messageId || isSubmitting) return null;
      setIsSubmitting(true);
      setError(null);
      try {
        const response = await resumeTrade(
          runId,
          approved,
          locale,
          conversationId ?? undefined,
        );
        setDecisions((prev) => ({
          ...prev,
          [messageId]: approved ? "approved" : "cancelled",
        }));
        setRunId(null);
        setConversationId(null);
        return {
          messageId,
          content: response.answer,
          artifacts: response.artifacts as MessageArtifacts | undefined,
          timestamp: formatTime(),
        };
      } catch (err) {
        if (!approved && isPendingRunNotFound(err)) {
          const fallback = "Ordre annulé par l'utilisateur.";
          setDecisions((prev) => ({ ...prev, [messageId]: "cancelled" }));
          setRunId(null);
          setConversationId(null);
          return {
            messageId,
            content: fallback,
            timestamp: formatTime(),
          };
        }
        const message =
          err instanceof Error ? err.message : "Échec de la validation du trade.";
        setError(message);
        return null;
      } finally {
        setIsSubmitting(false);
      }
    },
    [runId, messageId, conversationId, isSubmitting, locale],
  );

  const approve = useCallback(() => submitDecision(true), [submitDecision]);
  const cancel = useCallback(() => submitDecision(false), [submitDecision]);

  return {
    registerReview,
    getDecision,
    approve,
    cancel,
    reset,
    isSubmitting,
    error,
    pendingMessageId: messageId,
  };
}

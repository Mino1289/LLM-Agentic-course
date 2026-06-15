"use client";

import { useCallback, useState } from "react";
import type { TradeDecision } from "@/lib/types/chat";
import { resumeTrade } from "@/lib/api/chat";
import { formatTime } from "@/lib/utils";
import type { AppLocale } from "@/i18n/routing";

export function useTradeApproval(locale: AppLocale) {
  const [runId, setRunId] = useState<string | null>(null);
  const [messageId, setMessageId] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<Record<string, TradeDecision>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const registerReview = useCallback((nextRunId: string, nextMessageId: string) => {
    setRunId(nextRunId);
    setMessageId(nextMessageId);
    setDecisions((prev) => ({ ...prev, [nextMessageId]: "pending" }));
  }, []);

  const reset = useCallback(() => {
    setRunId(null);
    setMessageId(null);
    setDecisions({});
    setIsSubmitting(false);
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
      setDecisions((prev) => ({
        ...prev,
        [messageId]: approved ? "approved" : "cancelled",
      }));
      try {
        const response = await resumeTrade(runId, approved, locale);
        setRunId(null);
        return {
          messageId,
          content: response.answer,
          artifacts: response.artifacts,
          timestamp: formatTime(),
        };
      } finally {
        setIsSubmitting(false);
      }
    },
    [runId, messageId, isSubmitting, locale],
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
    pendingMessageId: messageId,
  };
}

"use client";

import { useEffect, useRef } from "react";
import type { ChatMessage, TradeDecision } from "@/lib/types/chat";
import type { StreamPipelineState } from "@/lib/agentPipeline";
import { MessageAssistant, MessageUser } from "@/components/chat/MessageBubble";
import { ArtifactStack } from "@/components/artifacts/ArtifactStack";
import { TradeApprovalCard } from "@/components/artifacts/TradeApprovalCard";
import type { AppLocale } from "@/i18n/routing";

interface ChatAreaProps {
  messages: ChatMessage[];
  locale: AppLocale;
  streamingMessageId: string | null;
  streamPipeline: StreamPipelineState | null;
  streamReasoningTitle: string;
  pipelineModeMultiLabel: string;
  pipelineModeSimpleLabel: string;
  emptyTitle: string;
  avatarLabels: {
    user: string;
    assistant: string;
  };
  artifactLabels: {
    thoughts: string;
    sources: string;
    reports: string;
    stats: string;
    allTickers: string;
    allSections: string;
    download: string;
    trade: {
      title: string;
      ticker: string;
      side: string;
      quantity: string;
      orderType: string;
      limitPrice: string;
      risk: string;
      justification: string;
      compliance: string;
      approve: string;
      cancel: string;
      approved: string;
      cancelled: string;
      riskLow: string;
      riskMedium: string;
      riskHigh: string;
    };
  };
  getTradeDecision: (messageId: string) => TradeDecision;
  onApprove: () => void;
  onCancel: () => void;
  isTradeSubmitting?: boolean;
  pendingTradeMessageId?: string | null;
}

export function ChatArea({
  messages,
  locale,
  streamingMessageId,
  streamPipeline,
  streamReasoningTitle,
  pipelineModeMultiLabel,
  pipelineModeSimpleLabel,
  emptyTitle,
  avatarLabels,
  artifactLabels,
  getTradeDecision,
  onApprove,
  onCancel,
  isTradeSubmitting = false,
  pendingTradeMessageId = null,
}: ChatAreaProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingMessageId, streamPipeline]);

  const showEmpty = messages.length === 0;

  return (
    <div className="chat-area" id="chatArea">
      {showEmpty ? (
        <div className="chat-empty">
          <h2>{emptyTitle}</h2>
        </div>
      ) : (
        messages.map((message) => {
          if (message.role === "user") {
            return (
              <MessageUser
                key={message.id}
                message={message}
                avatarLabel={avatarLabels.user}
              />
            );
          }

          const stackLabels = {
            thoughts: artifactLabels.thoughts,
            sources: artifactLabels.sources,
            reports: artifactLabels.reports,
            stats: artifactLabels.stats,
            allTickers: artifactLabels.allTickers,
            allSections: artifactLabels.allSections,
            download: artifactLabels.download,
          };

          const trade = message.artifacts?.trade;
          const decision = getTradeDecision(message.id);
          const isPendingTrade =
            trade &&
            decision === "pending" &&
            message.id === pendingTradeMessageId &&
            !isTradeSubmitting;

          const isStreamingMessage = message.id === streamingMessageId;

          return (
            <MessageAssistant
              key={message.id}
              message={message}
              avatarLabel={avatarLabels.assistant}
              locale={locale}
              streamPipeline={isStreamingMessage ? streamPipeline : null}
              streamReasoningTitle={streamReasoningTitle}
              pipelineModeMultiLabel={pipelineModeMultiLabel}
              pipelineModeSimpleLabel={pipelineModeSimpleLabel}
              bubbleChildren={
                message.artifacts ? (
                  <ArtifactStack artifacts={message.artifacts} labels={stackLabels} />
                ) : null
              }
              bodyChildren={
                trade ? (
                  <TradeApprovalCard
                    trade={trade}
                    labels={artifactLabels.trade}
                    decision={decision}
                    onApprove={isPendingTrade ? onApprove : () => undefined}
                    onCancel={isPendingTrade ? onCancel : () => undefined}
                  />
                ) : null
              }
            />
          );
        })
      )}

      <div ref={endRef} />
    </div>
  );
}

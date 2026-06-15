import type { ChatMessage } from "@/lib/types/chat";
import type { StreamPipelineState } from "@/lib/agentPipeline";
import { MarkdownContent } from "@/components/chat/MarkdownContent";
import { AgentPipelinePanel } from "@/components/chat/AgentPipelinePanel";
import { StreamReasoningPanel } from "@/components/chat/StreamReasoningPanel";
import type { AppLocale } from "@/i18n/routing";

interface MessageUserProps {
  message: ChatMessage;
  avatarLabel: string;
}

export function MessageUser({ message, avatarLabel }: MessageUserProps) {
  return (
    <div className="message message-user">
      <div className="message-avatar">{avatarLabel}</div>
      <div className="message-body">
        <div className="message-bubble">{message.content}</div>
        <span className="message-time">{message.timestamp}</span>
      </div>
    </div>
  );
}

interface MessageAssistantProps {
  message: ChatMessage;
  avatarLabel: string;
  locale: AppLocale;
  streamPipeline?: StreamPipelineState | null;
  streamReasoningTitle?: string;
  pipelineModeMultiLabel?: string;
  pipelineModeSimpleLabel?: string;
  bubbleChildren?: React.ReactNode;
  bodyChildren?: React.ReactNode;
}

export function MessageAssistant({
  message,
  avatarLabel,
  locale,
  streamPipeline,
  streamReasoningTitle = "",
  pipelineModeMultiLabel = "",
  pipelineModeSimpleLabel = "",
  bubbleChildren,
  bodyChildren,
}: MessageAssistantProps) {
  const hasContent = message.content.trim().length > 0;
  const showPipeline = !hasContent && streamPipeline;

  const modeLabel =
    streamPipeline?.mode === "multi"
      ? pipelineModeMultiLabel
      : streamPipeline?.mode === "simple"
        ? pipelineModeSimpleLabel
        : undefined;

  return (
    <div className="message message-assistant">
      <div className="message-avatar">{avatarLabel}</div>
      <div className="message-body">
        <div className="message-bubble">
          {showPipeline ? (
            <AgentPipelinePanel
              pipeline={streamPipeline}
              locale={locale}
              modeLabel={modeLabel}
            />
          ) : null}
          {hasContent ? <MarkdownContent content={message.content} /> : null}
          {bubbleChildren}
        </div>
        {showPipeline && streamPipeline.reasoning.length > 0 ? (
          <StreamReasoningPanel
            title={streamReasoningTitle}
            entries={streamPipeline.reasoning}
            locale={locale}
          />
        ) : null}
        {bodyChildren}
        <span className="message-time">{message.timestamp}</span>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { configToModelConfig, getConfig } from "@/lib/api/config";
import { useChat } from "@/hooks/useChat";
import { useSettings } from "@/hooks/useSettings";
import { useDisclosure, useSidebarState } from "@/hooks/useLayoutState";
import { useTradeApproval } from "@/hooks/useTradeApproval";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { ConfigPanel } from "@/components/layout/ConfigPanel";
import { ChatArea } from "@/components/chat/ChatArea";
import { ChatInput } from "@/components/chat/ChatInput";
import type { AppLocale } from "@/i18n/routing";
import type { ModelConfig } from "@/lib/types/chat";
import { MODEL_CONFIG } from "@/lib/mock/settings";

export function AppShell() {
  const locale = useLocale() as AppLocale;
  const tChat = useTranslations("chat");
  const tArtifacts = useTranslations("artifacts");
  const tTrade = useTranslations("trade");

  const { collapsed, mobileOpen, toggleSidebar, closeMobile } = useSidebarState();
  const configPanel = useDisclosure();
  const { settings, updateSetting } = useSettings();
  const tradeApproval = useTradeApproval(locale);

  const {
    messages,
    streamingMessageId,
    streamPipeline,
    isLoading,
    error,
    conversationId,
    conversationTitle,
    historyGroups,
    resetConversation,
    sendMessage,
    loadConversation,
    appendAssistantMessage,
  } = useChat(locale, settings, tradeApproval.registerReview);

  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    chatProvider: MODEL_CONFIG.chatProvider,
    chatModel: MODEL_CONFIG.chatModel,
    embeddingProvider: MODEL_CONFIG.embeddingProvider,
    embeddingModel: MODEL_CONFIG.embeddingModel,
  });

  useEffect(() => {
    void getConfig()
      .then((config) => setModelConfig(configToModelConfig(config)))
      .catch(() => undefined);
  }, []);

  const handleNewChat = () => {
    void resetConversation();
    tradeApproval.reset();
    closeMobile();
  };

  const handleSend = (text: string) => {
    void sendMessage(text);
  };

  const handleApprove = async () => {
    const result = await tradeApproval.approve();
    if (result) {
      appendAssistantMessage({
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.content,
        timestamp: result.timestamp,
        artifacts: result.artifacts,
      });
    }
  };

  const handleCancel = async () => {
    const result = await tradeApproval.cancel();
    if (result) {
      appendAssistantMessage({
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.content,
        timestamp: result.timestamp,
      });
    }
  };

  const artifactLabels = {
    thoughts: tArtifacts("thoughts"),
    sources: tArtifacts("sources"),
    reports: tArtifacts("reports"),
    stats: tArtifacts("stats"),
    allTickers: tArtifacts("allTickers"),
    allSections: tArtifacts("allSections"),
    download: tArtifacts("download"),
    trade: {
      title: tTrade("title"),
      ticker: tTrade("ticker"),
      side: tTrade("side"),
      quantity: tTrade("quantity"),
      orderType: tTrade("orderType"),
      limitPrice: tTrade("limitPrice"),
      risk: tTrade("risk"),
      justification: tTrade("justification"),
      compliance: tTrade("compliance"),
      approve: tTrade("approve"),
      cancel: tTrade("cancel"),
      approved: tTrade("approved"),
      cancelled: tTrade("cancelled"),
      riskLow: tTrade("riskLow"),
      riskMedium: tTrade("riskMedium"),
      riskHigh: tTrade("riskHigh"),
    },
  };

  return (
    <div className="app-shell" id="app">
      <Sidebar
        historyGroups={historyGroups}
        collapsed={collapsed}
        mobileOpen={mobileOpen}
        modelConfig={modelConfig}
        onNewChat={handleNewChat}
        onOpenSettings={configPanel.open}
        onSelectConversation={(id) => {
          void loadConversation(id);
          tradeApproval.reset();
          closeMobile();
        }}
        activeConversationId={conversationId}
      />

      <div className="main-panel">
        <Topbar title={conversationTitle} onToggleSidebar={toggleSidebar} />
        {error ? <div className="px-6 py-2 text-xs text-[var(--danger)]">{error}</div> : null}
        {isLoading ? (
          <div className="chat-empty">
            <h2>...</h2>
          </div>
        ) : (
          <ChatArea
            messages={messages}
            locale={locale}
            streamingMessageId={streamingMessageId}
            streamPipeline={streamPipeline}
            streamReasoningTitle={tChat("streamReasoning")}
            pipelineModeMultiLabel={tChat("pipelineModeMulti")}
            pipelineModeSimpleLabel={tChat("pipelineModeSimple")}
            emptyTitle={tChat("emptyTitle")}
            avatarLabels={{
              user: tChat("userAvatar"),
              assistant: tChat("assistantAvatar"),
            }}
            artifactLabels={artifactLabels}
            getTradeDecision={tradeApproval.getDecision}
            onApprove={handleApprove}
            onCancel={handleCancel}
            isTradeSubmitting={tradeApproval.isSubmitting}
            pendingTradeMessageId={tradeApproval.pendingMessageId}
          />
        )}
        <ChatInput
          placeholder={tChat("inputPlaceholder")}
          hint={tChat("inputHint")}
          sendLabel={tChat("send")}
          onSend={handleSend}
        />
      </div>

      <ConfigPanel
        open={configPanel.isOpen}
        onClose={configPanel.close}
        locale={locale}
        settings={settings}
        onSettingChange={updateSetting}
      />
    </div>
  );
}

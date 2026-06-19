"use client";

import { MessageSquare, Plus, Settings } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";
import { useRouter, usePathname } from "@/i18n/navigation";
import type { HistoryGroup, ModelConfig } from "@/lib/types/chat";
import { cn } from "@/lib/utils";

interface SidebarProps {
  historyGroups: HistoryGroup[];
  collapsed: boolean;
  mobileOpen: boolean;
  modelConfig: ModelConfig;
  onNewChat: () => void;
  onOpenSettings: () => void;
  onSelectConversation: (id: string) => void;
  activeConversationId?: string | null;
}

export function Sidebar({
  historyGroups,
  collapsed,
  mobileOpen,
  modelConfig,
  onNewChat,
  onOpenSettings,
  onSelectConversation,
  activeConversationId,
}: SidebarProps) {
  const t = useTranslations("sidebar");
  const router = useRouter();
  const pathname = usePathname();
  const locale = useLocale();

  const switchLocale = () => {
    router.replace(pathname, { locale: locale === "fr" ? "en" : "fr" });
  };

  return (
    <aside
      className={cn(
        "sidebar",
        collapsed && "collapsed",
        mobileOpen && "open-mobile",
      )}
      id="sidebar"
    >
      <div className="sidebar-header">
        <button type="button" className="btn-new-chat" onClick={onNewChat}>
          <Plus size={16} />
          {t("newChat")}
        </button>
      </div>

      <div className="sidebar-history">
        {historyGroups.map((group) => (
          <div key={group.label}>
            <div className="history-label">{group.label}</div>
            {group.items.map((item) => (
              <button
                key={item.id}
                type="button"
                className={cn(
                  "history-item",
                  (item.active || item.id === activeConversationId) && "active",
                )}
                onClick={() => onSelectConversation(item.id)}
              >
                <MessageSquare size={14} className="opacity-50" />
                {item.title}
              </button>
            ))}
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="model-line">
          <span className="model-label">{t("chatLabel")}</span>{" "}
          {modelConfig.chatProvider} / {modelConfig.chatModel}
        </div>
        <div className="model-line">
          <span className="model-label">{t("embeddingsLabel")}</span>{" "}
          {modelConfig.embeddingProvider}
        </div>
        <button type="button" className="locale-switch" onClick={switchLocale}>
          {t("localeSwitch")}
        </button>
        <button type="button" className="btn-settings" onClick={onOpenSettings}>
          <Settings size={16} />
          {t("settings")}
        </button>
      </div>
    </aside>
  );
}

"use client";

import { PanelLeft } from "lucide-react";
import { useTranslations } from "next-intl";
import { APP_BADGE } from "@/lib/constants";

interface TopbarProps {
  title: string;
  onToggleSidebar: () => void;
}

export function Topbar({ title, onToggleSidebar }: TopbarProps) {
  const t = useTranslations("topbar");

  return (
    <div className="topbar">
      <button
        type="button"
        className="btn-toggle-sidebar"
        onClick={onToggleSidebar}
        aria-label={t("toggleSidebar")}
      >
        <PanelLeft size={18} />
      </button>
      <span className="topbar-title">{title}</span>
      <span className="topbar-model">{APP_BADGE}</span>
    </div>
  );
}

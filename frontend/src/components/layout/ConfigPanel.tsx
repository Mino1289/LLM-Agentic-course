"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";
import type { AgentSettings, ToolDefinition } from "@/lib/types/chat";
import { SLIDER_FIELDS } from "@/lib/mock/settings";
import { getTools } from "@/lib/api/config";
import { Accordion } from "@/components/ui/Accordion";
import type { AppLocale } from "@/i18n/routing";

interface ConfigPanelProps {
  open: boolean;
  onClose: () => void;
  locale: AppLocale;
  settings: AgentSettings;
  onSettingChange: <K extends keyof AgentSettings>(
    key: K,
    value: AgentSettings[K],
  ) => void;
}

const CONFIG_LABEL_KEYS = {
  maxChunks: "maxChunks",
  subQueries: "subQueries",
  priceMaxDays: "priceMaxDays",
  priceMaxPoints: "priceMaxPoints",
  priceMaxTickers: "priceMaxTickers",
  priceDefaultWindow: "priceDefaultWindow",
  maxIterations: "maxIterations",
} as const;

export function ConfigPanel({
  open,
  onClose,
  locale,
  settings,
  onSettingChange,
}: ConfigPanelProps) {
  const t = useTranslations("config");
  const [tools, setTools] = useState<ToolDefinition[]>([]);

  useEffect(() => {
    if (!open) return;
    void getTools()
      .then(setTools)
      .catch(() => setTools([]));
  }, [open, locale]);

  return (
    <>
      <div
        className={`config-overlay${open ? " open" : ""}`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <div className={`config-panel${open ? " open" : ""}`} id="configPanel">
        <div className="config-header">
          <span className="config-title">{t("title")}</span>
          <button
            type="button"
            className="btn-close-config"
            onClick={onClose}
            aria-label={t("close")}
          >
            <X size={16} />
          </button>
        </div>

        <div className="config-body">
          <div className="config-section">
            <div className="config-section-title">{t("agentSection")}</div>
            {SLIDER_FIELDS.map((field) => (
              <div key={field.key} className="slider-field">
                <div className="slider-label">
                  <span className="slider-label-text">
                    {t(CONFIG_LABEL_KEYS[field.key])}
                  </span>
                  <span className="slider-value">{settings[field.key]}</span>
                </div>
                <input
                  type="range"
                  className="slider-track"
                  min={field.min}
                  max={field.max}
                  value={settings[field.key]}
                  onChange={(event) =>
                    onSettingChange(field.key, Number(event.target.value))
                  }
                />
                <div className="slider-range">
                  <span>{field.min}</span>
                  <span>{field.max}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="config-section">
            <Accordion title={t("toolsSection")} defaultOpen>
              {tools.map((tool) => (
                <div key={tool.name} className="tool-item">
                  <div className="tool-name">{tool.name}</div>
                  <div className="tool-desc">{tool.description}</div>
                </div>
              ))}
            </Accordion>
          </div>
        </div>
      </div>
    </>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import type { AgentSettings } from "@/lib/types/chat";
import { getConfig } from "@/lib/api/config";
import { DEFAULT_AGENT_SETTINGS } from "@/lib/mock/settings";

export function useSettings() {
  const [settings, setSettings] = useState<AgentSettings>(DEFAULT_AGENT_SETTINGS);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    void getConfig()
      .then((config) => {
        if (mounted) {
          setSettings(config.defaults);
        }
      })
      .catch(() => {
        if (mounted) {
          setSettings(DEFAULT_AGENT_SETTINGS);
        }
      })
      .finally(() => {
        if (mounted) setIsLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const updateSetting = useCallback(
    <K extends keyof AgentSettings>(key: K, value: AgentSettings[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  return { settings, updateSetting, isLoading };
}

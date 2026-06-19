"use client";

import type { AgentStep } from "@/lib/types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Accordion } from "@/components/ui/Accordion";

interface ThoughtsRendererProps {
  title: string;
  steps: AgentStep[];
  defaultOpen?: boolean;
}

export function ThoughtsRenderer({
  title,
  steps,
  defaultOpen = false,
}: ThoughtsRendererProps) {
  return (
    <Accordion title={title} defaultOpen={defaultOpen}>
      <ol className="step-list">
        {steps.map((step, index) => (
          <li key={step.id}>
            <span className="step-num">{index + 1}.</span>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{step.text}</ReactMarkdown>
          </li>
        ))}
      </ol>
    </Accordion>
  );
}

"use client";

import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface AccordionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  className?: string;
}

export function Accordion({
  title,
  defaultOpen = false,
  children,
  className,
}: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={cn("accordion", open && "open", className)}>
      <button
        type="button"
        className="accordion-trigger"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <ChevronRight />
        {title}
      </button>
      <div className="accordion-content">{children}</div>
    </div>
  );
}

"use client";

import { Send } from "lucide-react";
import { useRef } from "react";

interface ChatInputProps {
  placeholder: string;
  hint: string;
  sendLabel: string;
  onSend: (text: string) => void;
}

export function ChatInput({ placeholder, hint, sendLabel, onSend }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  };

  const submit = () => {
    const value = textareaRef.current?.value ?? "";
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    if (textareaRef.current) {
      textareaRef.current.value = "";
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="input-area">
      <div className="input-box">
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={placeholder}
          onInput={autoResize}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          className="btn-send"
          aria-label={sendLabel}
          onClick={submit}
        >
          <Send size={16} />
        </button>
      </div>
      <div className="input-hint">{hint}</div>
    </div>
  );
}

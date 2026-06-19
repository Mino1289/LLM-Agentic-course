"use client";

import { Check, Lock, X } from "lucide-react";
import type { TradeDecision, TradeProposal } from "@/lib/types/chat";
import { Accordion } from "@/components/ui/Accordion";
import { cn } from "@/lib/utils";

interface TradeApprovalCardProps {
  trade: TradeProposal;
  labels: {
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
  decision: TradeDecision;
  onApprove: () => void;
  onCancel: () => void;
}

function riskBadgeClass(level: TradeProposal["riskLevel"]) {
  if (level === "low") return "badge-success";
  if (level === "high") return "badge-danger";
  return "badge-warn";
}

function riskLabel(
  level: TradeProposal["riskLevel"],
  labels: TradeApprovalCardProps["labels"],
) {
  if (level === "low") return labels.riskLow;
  if (level === "high") return labels.riskHigh;
  return labels.riskMedium;
}

export function TradeApprovalCard({
  trade,
  labels,
  decision,
  onApprove,
  onCancel,
}: TradeApprovalCardProps) {
  const isPending = decision === "pending";

  return (
    <div
      className={cn(
        "trade-card",
        decision === "approved" && "approved",
        decision === "cancelled" && "cancelled",
      )}
    >
      <div className="trade-header">
        <div className="trade-header-icon">
          <Lock size={18} />
        </div>
        <div className="trade-header-title">{labels.title}</div>
        {trade.complianceVerdict ? (
          <span
            className={cn(
              "badge",
              trade.complianceVerdict === "PASS"
                ? "badge-success"
                : trade.complianceVerdict === "FAIL"
                  ? "badge-danger"
                  : "badge-warn",
            )}
          >
            {trade.complianceVerdict}
          </span>
        ) : null}
      </div>

      <div className="trade-grid">
        <div className="trade-field">
          <div className="trade-field-label">{labels.ticker}</div>
          <div className="trade-field-value">{trade.ticker}</div>
        </div>
        <div className="trade-field">
          <div className="trade-field-label">{labels.side}</div>
          <div className="trade-field-value">{trade.side}</div>
        </div>
        <div className="trade-field">
          <div className="trade-field-label">{labels.quantity}</div>
          <div className="trade-field-value">{trade.quantity}</div>
        </div>
        <div className="trade-field">
          <div className="trade-field-label">{labels.orderType}</div>
          <div className="trade-field-value">{trade.orderType}</div>
        </div>
        <div className="trade-field">
          <div className="trade-field-label">{labels.limitPrice}</div>
          <div className="trade-field-value">{trade.limitPrice || "—"}</div>
        </div>
        <div className="trade-field">
          <div className="trade-field-label">{labels.risk}</div>
          <div className="trade-field-value">
            <span className={cn("badge", riskBadgeClass(trade.riskLevel))}>
              {riskLabel(trade.riskLevel, labels)}
            </span>
          </div>
        </div>
      </div>

      <Accordion
        title={labels.justification}
        defaultOpen
        className="!mx-5 !mb-0 !rounded-none !border-x-0 !border-b-0"
      >
        <p className="whitespace-pre-wrap break-words">{trade.justification}</p>
      </Accordion>

      {trade.complianceDetail ? (
        <Accordion
          title={labels.compliance}
          defaultOpen={trade.complianceVerdict === "FAIL"}
          className="!mx-5 !mb-0 !rounded-none !border-x-0 !border-b-0"
        >
          <p className="whitespace-pre-wrap break-words">{trade.complianceDetail}</p>
        </Accordion>
      ) : null}

      <div className="trade-actions">
        <button
          type="button"
          className={cn("btn-approve", decision === "cancelled" && "btn-approve-muted")}
          onClick={onApprove}
          disabled={!isPending}
        >
          <Check size={16} />
          {decision === "approved" ? labels.approved : labels.approve}
        </button>
        <button
          type="button"
          className="btn-cancel-trade"
          onClick={onCancel}
          disabled={!isPending}
        >
          <X size={16} />
          {decision === "cancelled" ? labels.cancelled : labels.cancel}
        </button>
      </div>
    </div>
  );
}

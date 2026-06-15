"use client";

import type { ReportArtifact } from "@/lib/types/chat";
import { reportDownloadUrl } from "@/lib/api/chat";
import { Accordion } from "@/components/ui/Accordion";

interface ReportsRendererProps {
  title: string;
  reports: ReportArtifact[];
  downloadLabel: string;
  defaultOpen?: boolean;
}

export function ReportsRenderer({
  title,
  reports,
  downloadLabel,
  defaultOpen = false,
}: ReportsRendererProps) {
  const handleDownload = (report: ReportArtifact) => {
    const url = report.downloadUrl
      ? reportDownloadUrl(report.downloadUrl)
      : undefined;
    if (!url) return;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <Accordion title={title} defaultOpen={defaultOpen}>
      <div className="report-grid">
        {reports.map((report) => (
          <div key={report.id} className="report-tile">
            <div className={`report-icon ${report.type}`}>
              {report.type.toUpperCase()}
            </div>
            <div className="report-info">
              <div className="report-name">{report.name}</div>
              <div className="report-size">{report.size}</div>
            </div>
            <button
              type="button"
              className="btn-download"
              onClick={() => handleDownload(report)}
              disabled={!report.downloadUrl}
            >
              {downloadLabel}
            </button>
          </div>
        ))}
      </div>
    </Accordion>
  );
}

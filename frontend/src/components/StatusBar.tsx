import type { AnalysisDTO, EvidenceStateDTO } from "../types";

interface StatusBarProps {
  analysis?: AnalysisDTO;
  evidence?: EvidenceStateDTO;
  elapsedSeconds?: number;
  routeFallback?: string;
  routingExplanation?: string;
}

function modeLabel(mode?: string) {
  if (mode === "strong") return "高置信";
  if (mode === "weak") return "低置信";
  if (mode === "insufficient") return "证据不足";
  return "未知";
}

export function StatusBar({
  analysis,
  evidence,
  elapsedSeconds,
  routeFallback,
  routingExplanation,
}: StatusBarProps) {
  return (
    <section className="panel panel-status">
      <div className="panel-header">
        <h2>系统状态</h2>
      </div>
      <div className="status-grid">
        <div className="status-item">
          <span className="status-label">路由策略</span>
          <strong>{analysis?.strategy || "-"}</strong>
        </div>
        <div className="status-item">
          <span className="status-label">证据模式</span>
          <strong>{modeLabel(evidence?.mode)}</strong>
        </div>
        <div className="status-item">
          <span className="status-label">响应耗时</span>
          <strong>{elapsedSeconds ? `${elapsedSeconds.toFixed(2)}s` : "-"}</strong>
        </div>
        <div className="status-item">
          <span className="status-label">回退状态</span>
          <strong>{routeFallback || "无"}</strong>
        </div>
      </div>
      {routingExplanation ? <div className="status-note">{routingExplanation}</div> : null}
      {evidence ? (
        <div className="status-note">
          {`reason=${evidence.reason}, rerank=${evidence.top_rerank_score.toFixed(3)}, must_hit=${evidence.top_must_hit_count}`}
        </div>
      ) : null}
    </section>
  );
}

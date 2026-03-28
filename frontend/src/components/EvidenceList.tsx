import type { DocumentDTO } from "../types";

interface EvidenceListProps {
  documents: DocumentDTO[];
}

export function EvidenceList({ documents }: EvidenceListProps) {
  return (
    <section className="panel evidence-panel">
      <div className="panel-header">
        <h2>证据卡片</h2>
        <p>涉及的法规、条号与证据摘要</p>
      </div>
      {!documents.length ? <div className="placeholder">暂无证据卡片。</div> : null}
      <div className="evidence-grid">
        {documents.map((doc, index) => (
          <article key={`${doc.display_title}-${doc.article_id}-${index}`} className="evidence-card">
            <div className="evidence-head">
              <span className="evidence-title">
                {doc.display_title && !doc.display_title.includes("未知")
                  ? doc.display_title
                  : doc.article_title || doc.law_name || "相关证据"}
              </span>
            </div>
            {(doc.law_name || doc.article_id) && (
              <div className="evidence-meta">
                {doc.law_name ? <span>{doc.law_name}</span> : null}
                {doc.article_id ? <span>{doc.article_id}</span> : null}
              </div>
            )}
            {doc.snippet ? <p className="evidence-snippet">{doc.snippet}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

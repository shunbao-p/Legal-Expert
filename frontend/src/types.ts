export type EvidenceMode = "strong" | "weak" | "insufficient";
export type HealthStatus = "ready" | "starting" | "failed";

export interface AnalysisDTO {
  strategy: string;
  query_complexity: number;
  relationship_intensity: number;
  confidence: number;
  reasoning_required: boolean;
  reasoning: string;
}

export interface EvidenceStateDTO {
  mode: EvidenceMode;
  reason: string;
  top_rerank_score: number;
  top_must_hit_count: number;
}

export interface DocumentDTO {
  display_title: string;
  law_name: string;
  article_id: string;
  article_title: string;
  snippet: string;
  score: number;
  search_type: string;
  route_strategy: string;
  search_source: string;
  route_fallback: string;
}

export interface ChatResponse {
  answer: string;
  analysis: AnalysisDTO;
  evidence: EvidenceStateDTO;
  documents: DocumentDTO[];
  elapsed_seconds: number;
  route_fallback: string;
  routing_explanation: string;
}

export interface HealthResponse {
  status: HealthStatus;
  initialized: boolean;
  system_ready: boolean;
  startup_error: string;
}

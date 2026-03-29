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

export interface ChatSessionResponse {
  chat_id: string;
}

export interface SessionFileDTO {
  file_id: string;
  file_name: string;
  modality: string;
  status: string;
  size_bytes: number;
  uploaded_at: string;
  active: boolean;
  parsed_chunks: number;
  error: string;
}

export interface UploadFileResponse {
  file: SessionFileDTO;
}

export interface SessionFilesResponse {
  chat_id: string;
  files: SessionFileDTO[];
}

export interface DeleteFileResponse {
  chat_id: string;
  file_id: string;
  deleted: boolean;
}

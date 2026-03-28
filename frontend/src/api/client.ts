import type { ChatResponse, HealthResponse } from "../types";

export interface ChatRequest {
  question: string;
  explain_routing?: boolean;
}

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || "问答请求失败");
  }

  return (await response.json()) as ChatResponse;
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch("/health");
  if (!response.ok) {
    throw new Error("健康检查失败");
  }
  return (await response.json()) as HealthResponse;
}

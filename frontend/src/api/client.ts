import type {
  ChatResponse,
  ChatSessionResponse,
  DeleteFileResponse,
  HealthResponse,
  SessionFilesResponse,
  UploadFileResponse,
} from "../types";

export interface ChatRequest {
  chat_id: string;
  question: string;
  explain_routing?: boolean;
  active_file_ids?: string[];
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

export async function createChat(): Promise<ChatSessionResponse> {
  const response = await fetch("/chats", { method: "POST" });
  if (!response.ok) {
    throw new Error("会话创建失败");
  }
  return (await response.json()) as ChatSessionResponse;
}

export async function uploadChatFile(chatId: string, file: File): Promise<UploadFileResponse> {
  const formData = new FormData();
  formData.append("chat_id", chatId);
  formData.append("file", file);
  const response = await fetch("/files/upload", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || "文件上传失败");
  }
  return (await response.json()) as UploadFileResponse;
}

export async function listChatFiles(chatId: string): Promise<SessionFilesResponse> {
  const response = await fetch(`/chats/${encodeURIComponent(chatId)}/files`);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || "获取会话文件失败");
  }
  return (await response.json()) as SessionFilesResponse;
}

export async function deleteChatFile(chatId: string, fileId: string): Promise<DeleteFileResponse> {
  const response = await fetch(`/chats/${encodeURIComponent(chatId)}/files/${encodeURIComponent(fileId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail || "删除文件失败");
  }
  return (await response.json()) as DeleteFileResponse;
}

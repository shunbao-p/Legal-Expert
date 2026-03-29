import { useEffect, useRef, useState } from "react";

import { chat, createChat, deleteChatFile, health, listChatFiles, uploadChatFile } from "./api/client";
import { AnswerPanel } from "./components/AnswerPanel";
import { EvidenceList } from "./components/EvidenceList";
import { QuestionInput } from "./components/QuestionInput";
import type { DocumentDTO, SessionFileDTO } from "./types";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
}

export default function App() {
  const streamTimerRef = useRef<number | null>(null);
  const nextMessageIdRef = useRef(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [statusText, setStatusText] = useState("服务检查中...");
  const [chatId, setChatId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentDTO[]>([]);
  const [sessionFiles, setSessionFiles] = useState<SessionFileDTO[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function initialize() {
      try {
        const [healthResult, chatResult] = await Promise.all([health(), createChat()]);
        if (cancelled) {
          return;
        }
        setChatId(chatResult.chat_id);
        try {
          const filesResult = await listChatFiles(chatResult.chat_id);
          if (!cancelled) {
            setSessionFiles(filesResult.files || []);
          }
        } catch {
          if (!cancelled) {
            setSessionFiles([]);
          }
        }
        if (healthResult.startup_error) {
          setStatusText("服务暂不可用");
        } else if (healthResult.system_ready) {
          setStatusText("待命中");
        } else {
          setStatusText("服务准备中");
        }
      } catch {
        if (cancelled) {
          return;
        }
        setStatusText("连接异常");
        setError("初始化失败，请刷新页面重试。");
      }
    }
    void initialize();
    return () => {
      cancelled = true;
      if (streamTimerRef.current !== null) {
        window.clearInterval(streamTimerRef.current);
      }
    };
  }, []);

  function appendMessage(role: ChatMessage["role"], content: string) {
    const id = nextMessageIdRef.current;
    nextMessageIdRef.current += 1;
    setMessages((prev) => [...prev, { id, role, content }]);
    return id;
  }

  function streamAssistantAnswer(messageId: number, fullText: string) {
    return new Promise<void>((resolve) => {
      let index = 0;
      if (!fullText) {
        resolve();
        return;
      }
      if (streamTimerRef.current !== null) {
        window.clearInterval(streamTimerRef.current);
      }
      streamTimerRef.current = window.setInterval(() => {
        index = Math.min(fullText.length, index + 2);
        const nextText = fullText.slice(0, index);
        setMessages((prev) => prev.map((item) => (item.id === messageId ? { ...item, content: nextText } : item)));
        if (index >= fullText.length) {
          if (streamTimerRef.current !== null) {
            window.clearInterval(streamTimerRef.current);
            streamTimerRef.current = null;
          }
          resolve();
        }
      }, 16);
    });
  }

  async function refreshFiles(currentChatId: string) {
    const filesResult = await listChatFiles(currentChatId);
    setSessionFiles(filesResult.files || []);
  }

  async function uploadFiles(fileList: FileList | null) {
    if (!chatId || !fileList || fileList.length === 0) {
      return;
    }
    setError("");
    setStatusText("文件解析中...");
    try {
      const files = Array.from(fileList);
      const uploaded: SessionFileDTO[] = [];
      for (const file of files) {
        const result = await uploadChatFile(chatId, file);
        uploaded.push(result.file);
      }
      await refreshFiles(chatId);
      const failed = uploaded.filter((item) => item.status !== "ready");
      if (failed.length > 0) {
        setStatusText("部分文件解析失败");
        setError(
          failed
            .map((item) => `${item.file_name}: ${item.error || "解析失败"}`)
            .join("；"),
        );
      } else {
        setStatusText("文件已就绪");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "文件上传失败");
      setStatusText("上传失败");
    }
  }

  async function removeFile(fileId: string) {
    if (!chatId) {
      return;
    }
    setError("");
    try {
      await deleteChatFile(chatId, fileId);
      await refreshFiles(chatId);
      setStatusText("文件已移除");
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除文件失败");
      setStatusText("删除失败");
    }
  }

  async function submitQuestion(question: string) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || loading) {
      return;
    }
    if (!chatId) {
      setError("会话未初始化完成，请稍后重试。");
      return;
    }

    appendMessage("user", cleanQuestion);
    setLoading(true);
    setError("");
    setStatusText("查询中...");

    try {
      const readyFileIds = sessionFiles
        .filter((item) => item.active && item.status === "ready")
        .map((item) => item.file_id);
      const result = await chat({
        chat_id: chatId,
        question: cleanQuestion,
        explain_routing: false,
        active_file_ids: readyFileIds.length > 0 ? readyFileIds : undefined,
      });
      setDocuments(result.documents);

      setStatusText("回答生成中...");
      const assistantId = appendMessage("assistant", "");
      await streamAssistantAnswer(assistantId, result.answer || "暂无可展示回答。");
      setStatusText("已完成");
    } catch (err) {
      setError(err instanceof Error ? err.message : "问答请求失败");
      appendMessage("assistant", "抱歉，当前请求失败，请稍后重试。");
      setStatusText("请求失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Legal Consultation</p>
          <h1>法律咨询专家</h1>
        </div>
        <div className="service-pill">{statusText}</div>
      </header>

      <div className="main-layout">
        <aside className="evidence-side">
          <EvidenceList documents={documents} />
        </aside>
        <section className="chat-side">
          <AnswerPanel messages={messages} loading={loading} error={error} />
          <QuestionInput
            loading={loading}
            chatReady={Boolean(chatId)}
            files={sessionFiles}
            onSubmit={submitQuestion}
            onUploadFiles={uploadFiles}
            onRemoveFile={removeFile}
          />
        </section>
      </div>
    </main>
  );
}

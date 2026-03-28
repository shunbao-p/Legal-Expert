import { useEffect, useRef, useState } from "react";

import { chat, health } from "./api/client";
import { AnswerPanel } from "./components/AnswerPanel";
import { EvidenceList } from "./components/EvidenceList";
import { QuestionInput } from "./components/QuestionInput";
import type { DocumentDTO } from "./types";

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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentDTO[]>([]);

  useEffect(() => {
    let cancelled = false;
    health()
      .then((result) => {
        if (!cancelled) {
          if (result.startup_error) {
            setStatusText("服务暂不可用");
          } else if (result.system_ready) {
            setStatusText("待命中");
          } else {
            setStatusText("服务准备中");
          }
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStatusText("连接异常");
        }
      });
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

  async function submitQuestion(question: string) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || loading) {
      return;
    }

    appendMessage("user", cleanQuestion);
    setLoading(true);
    setError("");
    setStatusText("查询中...");

    try {
      const result = await chat({ question: cleanQuestion, explain_routing: false });
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
          <QuestionInput loading={loading} onSubmit={submitQuestion} />
        </section>
      </div>
    </main>
  );
}

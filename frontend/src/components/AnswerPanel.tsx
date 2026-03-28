import { useEffect, useRef } from "react";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
}

interface AnswerPanelProps {
  messages: ChatMessage[];
  loading: boolean;
  error: string;
}

export function AnswerPanel({ messages, loading, error }: AnswerPanelProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  return (
    <section className="chat-panel">
      {!messages.length ? <div className="placeholder">请在下方输入法律问题开始咨询。</div> : null}
      {error ? <div className="error-box">{error}</div> : null}
      <div className="message-list">
        {messages.map((message) => (
          <article
            key={message.id}
            className={message.role === "user" ? "message-item message-user" : "message-item message-assistant"}
          >
            <div className="message-role">{message.role === "user" ? "你" : "法律咨询专家"}</div>
            <div className="message-content">{message.content || (loading ? "..." : "")}</div>
          </article>
        ))}
        {loading ? <div className="typing-hint">正在整理回答...</div> : null}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}

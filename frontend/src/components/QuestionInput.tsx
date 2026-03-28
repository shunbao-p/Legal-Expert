import { useState } from "react";

interface QuestionInputProps {
  loading: boolean;
  onSubmit: (question: string) => Promise<void>;
}

const EXAMPLE_QUESTIONS = [
  "什么情况下离婚需要受处罚？",
  "言语中伤他人算违法吗？",
  "恶意伤人的处罚是什么？",
  "未成年人犯罪的处理办法是什么？",
];

export function QuestionInput({ loading, onSubmit }: QuestionInputProps) {
  const [question, setQuestion] = useState("");
  const [isComposing, setIsComposing] = useState(false);

  async function submitCurrentQuestion() {
    const trimmed = question.trim();
    if (!trimmed || loading) {
      return;
    }
    await onSubmit(trimmed);
    setQuestion("");
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitCurrentQuestion();
  }

  return (
    <section className="composer">
      <div className="example-list">
        {EXAMPLE_QUESTIONS.map((item) => (
          <button key={item} type="button" className="example-chip" onClick={() => setQuestion(item)} disabled={loading}>
            {item}
          </button>
        ))}
      </div>
      <form className="composer-form" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !isComposing) {
              event.preventDefault();
              void submitCurrentQuestion();
            }
          }}
          placeholder="请输入你的法律问题，例如：拖欠工资怎么处罚？"
          rows={2}
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? "生成中..." : "发送"}
        </button>
      </form>
    </section>
  );
}

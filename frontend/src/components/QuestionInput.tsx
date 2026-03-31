import { useRef, useState } from "react";

import type { SessionFileDTO } from "../types";

interface QuestionInputProps {
  loading: boolean;
  chatReady: boolean;
  files: SessionFileDTO[];
  onSubmit: (question: string) => Promise<void>;
  onUploadFiles: (files: FileList | null) => Promise<void>;
  onRemoveFile: (fileId: string) => Promise<void>;
}

const EXAMPLE_QUESTIONS = [
  "什么情况下离婚需要受处罚？",
  "言语中伤他人算违法吗？",
  "未成年人在游戏平台大额充值后，家长通常如何申请退款？",
  "恶意伤人的处罚是什么？",
  "未成年人犯罪的处理办法是什么？",
  "网购商品存在质量问题但商家拒绝退货，下一步怎么处理？",
  "公司入职后一直不签书面劳动合同，员工可以要求什么赔偿？",
];

export function QuestionInput({
  loading,
  chatReady,
  files,
  onSubmit,
  onUploadFiles,
  onRemoveFile,
}: QuestionInputProps) {
  const [question, setQuestion] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  async function handlePickFiles(event: React.ChangeEvent<HTMLInputElement>) {
    setUploading(true);
    try {
      await onUploadFiles(event.target.files);
    } finally {
      setUploading(false);
      if (event.target) {
        event.target.value = "";
      }
    }
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
      <div className="file-toolbar">
        <input
          ref={fileInputRef}
          type="file"
          className="file-input-hidden"
          multiple
          accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.png,.jpg,.jpeg,.bmp,.webp,.mp3,.wav,.m4a,.flac,.txt,.md"
          onChange={handlePickFiles}
          disabled={loading || uploading || !chatReady}
        />
        <button
          type="button"
          className="file-upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={loading || uploading || !chatReady}
        >
          {uploading ? "上传中..." : "上传文件"}
        </button>
        <span className="file-hint">支持 PDF/Word/Excel/图片/音频</span>
      </div>
      {files.length > 0 ? (
        <div className="file-chip-list">
          {files.map((file) => (
            <div key={file.file_id} className="file-chip">
              <span className="file-chip-name">{file.file_name}</span>
              <span className="file-chip-status">{file.status}</span>
              <button
                type="button"
                className="file-chip-remove"
                onClick={() => void onRemoveFile(file.file_id)}
                disabled={loading || uploading}
                aria-label={`移除 ${file.file_name}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}
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

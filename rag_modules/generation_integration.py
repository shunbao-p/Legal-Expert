"""
生成集成模块（法律场景）
"""

import logging
import os
from typing import List, Optional

from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责法律场景答案生成"""

    DISCLAIMER_TEXT = "免责声明：本回答仅供参考，不构成法律意见，需由专业法律人士复核。"

    def __init__(
        self,
        model_name: str = "kimi-k2-0711-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        enable_legal_disclaimer: bool = True,
        risk_notice_level: str = "light",
        llm_dispatcher: Optional[object] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_legal_disclaimer = enable_legal_disclaimer
        self.risk_notice_level = risk_notice_level
        self.llm_dispatcher = llm_dispatcher

        # 向后兼容：保留直接Kimi客户端能力，供未接入调度层的调用路径使用。
        self.client = None
        api_key = os.getenv("MOONSHOT_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

        if self.llm_dispatcher is None and self.client is None:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量，或接入 llm_dispatcher")

        logger.info("生成模块初始化完成，模型: %s, dispatcher=%s", model_name, bool(self.llm_dispatcher))

    def _create_generation_completion(self, messages, stream: bool = False, timeout: Optional[int] = None):
        if self.llm_dispatcher is not None:
            response, provider, model = self.llm_dispatcher.create_chat_completion(
                role="generation",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
                timeout=timeout,
            )
            logger.info("生成调用通道: provider=%s model=%s stream=%s", provider, model, stream)
            return response
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            timeout=timeout,
        )

    def _build_context(self, documents: List[Document]) -> str:
        context_parts: List[str] = []
        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue
            level = doc.metadata.get("retrieval_level", "")
            law_name = doc.metadata.get("law_name", "")
            article_id = doc.metadata.get("article_id", "")
            prefix = []
            if level:
                prefix.append(level.upper())
            if law_name:
                prefix.append(law_name)
            if article_id:
                prefix.append(str(article_id))
            header = f"[{'|'.join(prefix)}] " if prefix else ""
            context_parts.append(f"{header}{content}")
        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, documents: List[Document], answer_mode: str = "strong") -> str:
        context = self._build_context(documents)
        disclaimer_clause = (
            "最后必须保留免责声明：本回答仅供参考，不构成法律意见，需由专业法律人士复核。"
            if self.enable_legal_disclaimer
            else "免责声明可省略。"
        )

        risk_instruction = (
            "风险提示保持轻量，聚焦可能后果与应对方向。"
            if self.risk_notice_level == "light"
            else "风险提示可适当展开，但避免绝对化判断。"
        )
        mode = (answer_mode or "strong").lower()
        if mode == "weak":
            mode_instruction = (
                "当前为低置信度回答模式：结论必须写明“当前判断置信度较低，需要人工复核”，"
                "并在适用边界中明确不确定点。"
            )
        elif mode == "insufficient":
            mode_instruction = (
                "当前为证据不足模式：结论必须写“当前检索证据不足以作出确定判断”，"
                "并给出补充信息建议，不得输出确定性结论。"
            )
        else:
            mode_instruction = "当前为正常回答模式：结论需谨慎但可给出结构化判断。"

        return f"""
你是法律法规咨询助手（非律师执业意见提供者）。
请严格基于检索信息回答，不得编造法条。

检索到的相关信息：
{context}

用户问题：{question}

输出格式必须包含以下小节（按顺序）：
1. 结论
2. 依据条款
3. 关联说明
4. 适用边界
5. 风险提示
6. 免责声明

要求：
- 若证据不足，必须在“结论”中明确写“当前检索证据不足以作出确定判断”。
- “依据条款”尽量列出法规名与条文编号；无法确定时明确说明“待核对原文”。
- {risk_instruction}
- {mode_instruction}
- 不得给出“必然胜诉/绝对合法/绝对违法”等确定性执业结论。
- {disclaimer_clause}

请输出结构化中文回答：
"""

    def _has_disclaimer(self, text: str) -> bool:
        compact = text.replace(" ", "")
        return (
            "仅供参考" in compact
            and "不构成法律意见" in compact
            and ("复核" in compact or "专业法律人士" in compact)
        )

    def _ensure_disclaimer(self, text: str) -> str:
        if not self.enable_legal_disclaimer:
            return text
        if self._has_disclaimer(text):
            return text
        logger.warning("模型输出未命中免责声明，已自动追加 disclaimer_appended=true")
        return f"{text.rstrip()}\n\n{self.DISCLAIMER_TEXT}"

    def _build_evidence_summary(self, documents: List[Document], limit: int = 3) -> str:
        if not documents:
            return "- 无可用检索证据。"
        lines: List[str] = []
        for doc in documents[:limit]:
            law_name = doc.metadata.get("law_name") or "未标注法规"
            article_id = doc.metadata.get("article_id") or "条文号未知"
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = f"{snippet[:120]}..."
            lines.append(f"- {law_name} / {article_id}: {snippet or '无正文片段'}")
        return "\n".join(lines)

    def _build_degraded_answer(self, documents: List[Document], error: Exception) -> str:
        answer = (
            "结论\n"
            "当前生成通道暂时不可用，已返回检索证据摘要供人工复核。\n\n"
            "已检索证据摘要\n"
            f"{self._build_evidence_summary(documents)}\n\n"
            "建议\n"
            "请稍后重试，或由专业法律人士结合原文进一步确认。"
        )
        logger.error("触发生成降级输出: %s", error)
        return self._ensure_disclaimer(answer)

    def _build_insufficient_answer(self, question: str, documents: List[Document]) -> str:
        answer = (
            "结论\n"
            "当前检索证据不足以作出确定判断。\n\n"
            "依据条款\n"
            "已检索到部分相关信息，但证据命中不足，需补充更具体的法规名、条文号或事实细节。\n\n"
            "关联说明\n"
            f"问题：{question}\n"
            f"证据摘要：\n{self._build_evidence_summary(documents)}\n\n"
            "适用边界\n"
            "当前回答仅用于检索辅助，不构成法律定性意见。\n\n"
            "风险提示\n"
            "在证据不充分时直接采纳结论可能导致判断偏差，建议由专业法律人士复核。\n\n"
            "免责声明\n"
            f"{self.DISCLAIMER_TEXT}"
        )
        return self._ensure_disclaimer(answer)

    def generate_adaptive_answer(
        self,
        question: str,
        documents: List[Document],
        answer_mode: str = "strong",
    ) -> str:
        if (answer_mode or "").lower() == "insufficient":
            return self._build_insufficient_answer(question, documents)

        prompt = self._build_prompt(question, documents, answer_mode=answer_mode)
        try:
            response = self._create_generation_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=60,
            )
            raw_answer = response.choices[0].message.content.strip()
            return self._ensure_disclaimer(raw_answer)
        except Exception as e:
            return self._build_degraded_answer(documents, e)

    def generate_adaptive_answer_stream(
        self,
        question: str,
        documents: List[Document],
        max_retries: int = 1,
        answer_mode: str = "strong",
    ):
        _ = max_retries  # 保留参数以兼容历史调用签名。
        if (answer_mode or "").lower() == "insufficient":
            yield self._build_insufficient_answer(question, documents)
            return

        prompt = self._build_prompt(question, documents, answer_mode=answer_mode)
        try:
            response = self._create_generation_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=60,
            )
            print("开始流式生成回答...\n")

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield delta

            # 流式收敛后做免责声明后校验，若缺失自动补全
            if self.enable_legal_disclaimer and not self._has_disclaimer(full_response):
                logger.warning("流式输出未命中免责声明，已自动追加 disclaimer_appended=true")
                yield f"\n\n{self.DISCLAIMER_TEXT}"
            return
        except Exception as e:
            yield self._build_degraded_answer(documents, e)
            return

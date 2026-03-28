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
                "当前为低置信度模式：请在结尾明确写出“内容仅供参考，如需更权威回复建议您前往专业机构咨询”，"
                "并点明关键不确定点。"
            )
        elif mode == "insufficient":
            mode_instruction = (
                "当前为证据不足模式：结论必须写“当前检索证据不足以作出确定判断，请更加详细描述您的问题”，"
                "并给出补充信息建议，不得输出确定性结论。"
            )
        else:
            mode_instruction = "当前为正常回答模式：结论需谨慎、可执行，语气自然专业。"

        return f"""
你是面向普通用户的法律咨询专家。
请严格基于检索信息回答，不得编造法条，不要输出开发者口吻。

检索到的相关信息：
{context}

用户问题：{question}

输出要求（用户导向）：
- 先用 1 段自然语言直接回答用户问题。
- 全文尽量使用 2-4 段连续表达，必要时可加少量小标题，但不要强制编号，不要机械分点。
- 依据说明要自然融入正文，优先引用“法规名+条号”；条号不确定时不进行引用“法规名+条号”。
- 避免“命中率、召回、rerank、候选集”等技术术语，改为用户能理解的表达。
- 若证据不足，必须明确写“当前检索证据不足以作出确定判断”，并说明还需要补充什么信息。
- {risk_instruction}
- {mode_instruction}
- 不得给出“必然胜诉/绝对合法/绝对违法”等确定性执业结论。
- {disclaimer_clause}

请输出自然、专业、面向用户的中文回答：
"""

    def _has_disclaimer(self, text: str) -> bool:
        compact = text.replace(" ", "")
        return (
             "本回答不构成法律意见" in compact
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
            "当前回答生成通道暂时不可用，我先把已检索到的关键信息给你，供你快速核对。\n\n"
            f"已检索证据摘要：\n{self._build_evidence_summary(documents)}\n\n"
            "建议你稍后重试一次；如果问题紧急，优先让专业法律人士结合法规原文做最终判断。"
        )
        logger.error("触发生成降级输出: %s", error)
        return self._ensure_disclaimer(answer)

    def _build_insufficient_answer(self, question: str, documents: List[Document]) -> str:
        answer = (
            f"就你问的“{question}”，当前检索证据不足以作出确定判断。\n\n"
            f"目前可供参考的检索信息如下：\n{self._build_evidence_summary(documents)}\n\n"
            "要把结论做实，建议补充更具体的事实要点（时间、行为、主体身份、是否有书面材料等），"
            "或直接给出你关心的法规名/条号，我可以据此继续精确检索并给出更稳妥的分析。\n\n"
            "在证据不充分时直接依据当前结果决策，可能导致判断偏差，建议让专业法律人士复核。"
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

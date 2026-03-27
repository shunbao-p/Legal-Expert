"""
生成集成模块（法律场景）
"""

import logging
import os
import time
from typing import List

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
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_legal_disclaimer = enable_legal_disclaimer
        self.risk_notice_level = risk_notice_level

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
        logger.info(f"生成模块初始化完成，模型: {model_name}")

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

    def _build_prompt(self, question: str, documents: List[Document]) -> str:
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

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        prompt = self._build_prompt(question, documents)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_answer = response.choices[0].message.content.strip()
            return self._ensure_disclaimer(raw_answer)
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self, question: str, documents: List[Document], max_retries: int = 3
    ):
        prompt = self._build_prompt(question, documents)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60,
                )

                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

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
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                logger.error("流式生成完全失败，尝试非流式后备方案")
                print("⚠️ 流式生成失败，切换到标准模式...")
                try:
                    yield self.generate_adaptive_answer(question, documents)
                    return
                except Exception as fallback_error:
                    logger.error(f"后备生成也失败: {fallback_error}")
                    yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                    return

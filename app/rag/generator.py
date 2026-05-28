from typing import List, Optional

from google import genai
from google.genai import types

from app.core.config import Settings
from app.models.schemas import EvidenceItem
from app.rag.model_providers import (
    GEMINI_PROVIDER,
    active_llm_model,
    normalize_provider,
    openai_compatible_client,
)
from app.services.safety import safety_notice


SYSTEM_PROMPT = """You are a research assistant for a multimodal RAG project about ultrasound and thermal imaging.
Use only the supplied retrieved evidence and visual summary.
Do not provide definitive diagnosis or treatment instructions.
If evidence is weak or missing, say so directly.
Always include short source references like [1], [2] that map to the evidence list."""


class AnswerGenerator:
    def generate(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str] = None,
        use_llm: bool = True,
    ) -> tuple[str, bool, Optional[str], str]:
        raise NotImplementedError


class RagAnswerGenerator(AnswerGenerator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = normalize_provider(settings.llm_provider)
        self.gemini_client: Optional[genai.Client] = None
        self.openai_client = openai_compatible_client(settings, self.provider)
        if self.provider == GEMINI_PROVIDER and settings.gemini_api_key:
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key)

    def generate(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str] = None,
        use_llm: bool = True,
    ) -> tuple[str, bool, Optional[str], str]:
        if self.gemini_client and self.settings.enable_gemini_generation and use_llm:
            answer = self._generate_with_gemini(question, evidence, visual_summary)
            return answer, True, active_llm_model(self.settings), self.provider
        if self.openai_client and self.settings.enable_llm_generation and use_llm:
            answer = self._generate_with_openai_compatible(question, evidence, visual_summary)
            return answer, True, active_llm_model(self.settings), self.provider
        return self._generate_fallback(question, evidence, visual_summary), False, None, "local"

    def _generate_with_gemini(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str],
    ) -> str:
        context = _format_context(evidence, self.settings.max_context_chars)
        visual = visual_summary or "No image was provided for this question."
        user_prompt = f"""Question:
{question}

Visual summary:
{visual}

Retrieved evidence:
{context}

Answer in concise Chinese by default unless the user asks for another language. Structure the answer with:
1. direct answer
2. evidence
3. uncertainty / next data needed
"""
        response = self.gemini_client.models.generate_content(
            model=active_llm_model(self.settings),
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.settings.gemini_thinking_level
                ),
            ),
        )
        return (response.text or "").strip()

    def _generate_with_openai_compatible(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str],
    ) -> str:
        context = _format_context(evidence, self.settings.max_context_chars)
        visual = visual_summary or "No image was provided for this question."
        user_prompt = f"""Question:
{question}

Visual summary:
{visual}

Retrieved evidence:
{context}

Answer in concise Chinese by default unless the user asks for another language. Structure the answer with:
1. direct answer
2. evidence
3. uncertainty / next data needed
"""
        response = self.openai_client.chat.completions.create(
            model=active_llm_model(self.settings),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _generate_fallback(
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str],
    ) -> str:
        if not evidence:
            return (
                "当前知识库没有检索到可用证据，因此不能给出可靠回答。"
                f"\n\n安全提示：{safety_notice()}"
            )

        lines = [
            "本地 fallback 模式已根据检索证据生成一个保守回答；配置所选 LLM_PROVIDER 的 API key 后会切换到模型生成。",
            "",
            f"问题：{question}",
        ]
        if visual_summary:
            lines.extend(["", f"图像摘要：{visual_summary}"])
        lines.append("")
        lines.append("可用证据：")
        for idx, item in enumerate(evidence, start=1):
            excerpt = item.content.replace("\n", " ")[:420]
            lines.append(f"[{idx}] {item.title}: {excerpt}")
        lines.extend(
            [
                "",
                "结论：以上证据只能支持研究/学习层面的归纳，不能作为诊断或治疗建议。",
            ]
        )
        return "\n".join(lines)


def _format_context(evidence: List[EvidenceItem], max_chars: int) -> str:
    blocks = []
    total = 0
    for idx, item in enumerate(evidence, start=1):
        block = (
            f"[{idx}] title={item.title}; modality={item.modality}; "
            f"score={item.score}; source={item.source_path or 'unknown'}\n{item.content}"
        )
        remaining = max_chars - total
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total += len(block)
    return "\n\n".join(blocks)

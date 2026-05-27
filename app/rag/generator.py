from typing import List, Optional

from openai import OpenAI

from app.core.config import Settings
from app.models.schemas import EvidenceItem
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
        use_openai: bool = True,
    ) -> tuple[str, bool, Optional[str]]:
        raise NotImplementedError


class RagAnswerGenerator(AnswerGenerator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Optional[OpenAI] = None
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str] = None,
        use_openai: bool = True,
    ) -> tuple[str, bool, Optional[str]]:
        if self.client and self.settings.enable_openai_generation and use_openai:
            answer = self._generate_with_openai(question, evidence, visual_summary)
            return answer, True, self.settings.openai_chat_model
        return self._generate_fallback(question, evidence, visual_summary), False, None

    def _generate_with_openai(
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
        response = self.client.responses.create(
            model=self.settings.openai_chat_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()

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
            "本地 fallback 模式已根据检索证据生成一个保守回答；配置 OPENAI_API_KEY 后会切换到 LLM 生成。",
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

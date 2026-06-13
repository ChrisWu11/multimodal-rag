from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.models.schemas import EvidenceItem
from app.rag.langchain_providers import build_chat_model, normalize_provider
from app.services.safety import safety_notice


SYSTEM_PROMPT = """You are a research assistant for a multimodal RAG project about ultrasound and thermal imaging.
Use only the supplied retrieved evidence and visual summary.
Do not provide definitive diagnosis or treatment instructions.
If evidence is weak or missing, say so directly.
Every factual claim must include short source references like [1], [2] that map to the evidence list.
Finish with a short "Sources used" list containing title, year, DOI, page(s), and chunk id when available."""


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
        self.chat_model = build_chat_model(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    """Question:
{question}

Visual summary:
{visual_summary}

Retrieved evidence:
{context}

Answer in English by default. Only answer in Chinese if the user explicitly asks for Chinese.
Use a concise academic tone and separate mature methods from experimental or proof-of-concept evidence.
Structure the answer with:
1. Direct answer
2. Evidence
3. Limitations / uncertainty
4. Sources used
""",
                ),
            ]
        )
        self.chain = self.prompt | self.chat_model | StrOutputParser() if self.chat_model else None

    def generate(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str] = None,
        use_llm: bool = True,
    ) -> tuple[str, bool, Optional[str], str]:
        if self.chain and use_llm:
            model = _model_name(self.settings, self.provider)
            try:
                answer = self._generate_with_langchain(question, evidence, visual_summary)
                return answer, True, model, self.provider
            except Exception as exc:  # noqa: BLE001
                answer = self._generate_fallback(question, evidence, visual_summary)
                answer += (
                    f"\n\nLLM generation failed, so the system fell back to a local evidence summary. "
                    f"Provider={self.provider}; model={model}; error={_safe_error(exc)}"
                )
                return answer, False, model, self.provider
        return self._generate_fallback(question, evidence, visual_summary), False, None, "local"

    def _generate_with_langchain(
        self,
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str],
    ) -> str:
        context = _format_context(evidence, self.settings.max_context_chars)
        visual = visual_summary or "No image was provided for this question."
        return self.chain.invoke(
            {
                "question": question,
                "visual_summary": visual,
                "context": context,
            }
        )

    @staticmethod
    def _generate_fallback(
        question: str,
        evidence: List[EvidenceItem],
        visual_summary: Optional[str],
    ) -> str:
        if not evidence:
            return (
                "No relevant evidence was retrieved from the current knowledge base, so a reliable "
                "answer cannot be provided."
                f"\n\nSafety notice: {safety_notice()}"
            )

        lines = [
            "Local fallback mode generated a conservative evidence summary. Configure the selected "
            "LLM provider API key to enable model-generated answers.",
            "",
            f"Question: {question}",
        ]
        if visual_summary:
            lines.extend(["", f"Visual summary: {visual_summary}"])
        lines.append("")
        lines.append("Retrieved evidence:")
        for idx, item in enumerate(evidence, start=1):
            citation = _source_summary(item)
            excerpt = item.content.replace("\n", " ")[:420]
            lines.append(f"[{idx}] {citation}: {excerpt}")
        lines.extend(
            [
                "",
                "Conclusion: These sources support research-oriented synthesis only and should not be "
                "used as diagnostic or treatment advice.",
            ]
        )
        return "\n".join(lines)


def _format_context(evidence: List[EvidenceItem], max_chars: int) -> str:
    blocks = []
    total = 0
    for idx, item in enumerate(evidence, start=1):
        metadata = item.metadata
        block = "\n".join(
            [
                f"[{idx}] {_source_summary(item)}",
                f"Modality: {item.modality}; score: {item.score}; source: {item.source_path or 'unknown'}",
                f"Section: {_metadata_value(metadata, 'section', 'unknown')}",
                f"Retrieval: {_metadata_value(metadata, 'retrieval_method', 'unknown')}",
                f"Evidence text: {item.content}",
            ]
        )
        remaining = max_chars - total
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total += len(block)
    return "\n\n".join(blocks)


def _source_summary(item: EvidenceItem) -> str:
    metadata = item.metadata
    year = _metadata_value(metadata, "year", "n.d.")
    doi = _metadata_value(metadata, "doi", "no DOI")
    chunk_id = _metadata_value(metadata, "source_chunk_id", item.chunk_id)
    pages = _citation_pages(metadata)
    return f"{item.title} ({year}); DOI: {doi}; pages: {pages}; chunk_id: {chunk_id}"


def _metadata_value(metadata: dict, key: str, fallback: str) -> str:
    value = metadata.get(key)
    if value is None or value == "":
        return fallback
    return str(value)


def _citation_pages(metadata: dict) -> str:
    start = metadata.get("page_start")
    end = metadata.get("page_end")
    if start in (None, "") and end in (None, ""):
        return "unknown"
    if start == end or end in (None, ""):
        return f"p. {start}"
    if start in (None, ""):
        return f"p. {end}"
    return f"pp. {start}-{end}"


def _model_name(settings: Settings, provider: str) -> Optional[str]:
    if provider == "gemini":
        return settings.gemini_model
    if provider == "openai":
        return settings.openai_model
    if provider == "qwen":
        return settings.qwen_model
    return None


def _safe_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message[:260]

"""
Multi-agent LangGraph pipeline for Yoga Chikitsa.

Pipeline: retrieve (Chroma similarity search) -> Yoga Guru agent (answers with
asanas/mudras grounded in the retrieved article excerpts) -> Doctor Advisor
agent (adds a general medical safety perspective on the proposed practice) ->
compose (combines both into the final answer shown to the user).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "yoga_asana_mudra"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-opus-4-8"
TOP_K = 5

GURU_SYSTEM_PROMPT = """\
You are Yoga Chikitsa, an ancient and wise yoga guru in the lineage of Patanjali, \
speaking to a seeker who wants relief from a health concern through yoga asanas, \
mudras, and pranayama. Speak warmly and with gentle authority, occasionally using \
a Sanskrit word (asana, pranayama, namaste, shanti) where it feels natural, but \
always stay clear and practical above all else.

Answer ONLY using the sacred texts (article excerpts) provided below as context. \
If the provided texts do not contain enough information to answer, say so honestly \
rather than inventing poses or claims.

Structure your answer with:
- A short warm opening
- The relevant asanas/mudras/practices, with brief how-to steps and benefits
- Precautions where relevant (e.g. avoid during injury, pregnancy, recent surgery)

Use markdown **bold** liberally to highlight the main points a reader should
walk away with: every pose/mudra name, key benefits, and any precaution. Do not
bold entire sentences or paragraphs — bold only the specific key words or
phrases, so the bolding stays meaningful and scannable.

Do not add a medical disclaimer yourself — a physician colleague will add a
medical safety note separately after your answer. Focus purely on the yogic
guidance.

Context from the sacred texts:
{context}
"""

DOCTOR_SYSTEM_PROMPT = """\
You are Dr. Ananya, a calm, careful general-practice physician. You work alongside \
a yoga guru: the guru has just given the seeker yogic guidance for their health \
concern, and your job is to add a brief, general medical safety perspective — you \
are not diagnosing this specific person and have no access to their medical history.

You will be given the seeker's question and the yoga guru's proposed guidance. \
In 4-6 concise sentences or short bullet points, cover:
- Any "red flag" symptoms related to this concern that warrant prompt in-person \
medical evaluation rather than relying on yoga alone
- Key precautions or contraindications for the specific poses/practices the guru \
suggested (e.g. conditions where a pose should be avoided or modified)
- A clear closing reminder that this is general educational information, not a \
diagnosis or personalized medical advice, and personalized care requires seeing a \
licensed physician

Do not re-explain or repeat the yoga poses themselves — the guru already covered \
those. Speak plainly, as a modern doctor, in contrast to the guru's ancient tone.

Use markdown **bold** to highlight the main points: red-flag symptoms, specific \
precautions/contraindications, and the closing reminder. Bold only the key words \
or phrases, not full sentences.
"""

DOCTOR_USER_TEMPLATE = """\
The seeker's question:
{question}

The yoga guru's guidance:
{guru_answer}
"""


class ChatState(TypedDict):
    question: str
    history: list[tuple[str, str]]
    context: list[Document]
    guru_answer: str
    doctor_advice: str
    answer: str


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Untitled")
        blocks.append(f"[Excerpt {i} — {title}]\n{doc.page_content}")
    return "\n\n".join(blocks)


class YogaRagPipeline:
    def __init__(self) -> None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_DIR}. Run `python ingest.py` first."
            )
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        # Claude Opus 4.8 rejects sampling params (temperature/top_p/top_k) outright —
        # tone is steered entirely through each agent's system prompt persona.
        self.guru_llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1500)
        self.doctor_llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=600)
        self.graph = self._build_graph()

    # --- Agent 1: retrieval ---
    def _retrieve(self, state: ChatState) -> dict:
        docs = self.retriever.invoke(state["question"])
        return {"context": docs}

    # --- Agent 2: Yoga Guru ---
    def _yoga_guru(self, state: ChatState) -> dict:
        system = GURU_SYSTEM_PROMPT.format(context=_format_context(state["context"]))
        messages = [SystemMessage(content=system)]
        for role, content in state.get("history", []):
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=state["question"]))

        response = self.guru_llm.invoke(messages)
        return {"guru_answer": response.content}

    # --- Agent 3: Doctor Advisor ---
    def _doctor_advisor(self, state: ChatState) -> dict:
        user_content = DOCTOR_USER_TEMPLATE.format(
            question=state["question"], guru_answer=state["guru_answer"]
        )
        messages = [
            SystemMessage(content=DOCTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = self.doctor_llm.invoke(messages)
        return {"doctor_advice": response.content}

    # --- Compose the two agents' outputs into one answer ---
    def _compose(self, state: ChatState) -> dict:
        answer = (
            f"{state['guru_answer']}\n\n---\n\n"
            f"### \U0001FA7A Doctor's Note\n\n{state['doctor_advice']}"
        )
        return {"answer": answer}

    def _build_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("yoga_guru", self._yoga_guru)
        graph.add_node("doctor_advisor", self._doctor_advisor)
        graph.add_node("compose", self._compose)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "yoga_guru")
        graph.add_edge("yoga_guru", "doctor_advisor")
        graph.add_edge("doctor_advisor", "compose")
        graph.add_edge("compose", END)
        return graph.compile()

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> dict:
        result = self.graph.invoke({"question": question, "history": history or []})
        sources = []
        seen_titles = set()
        for doc in result["context"]:
            title = doc.metadata.get("title", "Untitled")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sources.append({"title": title, "url": doc.metadata.get("url", "")})
        return {
            "guru_answer": result["guru_answer"],
            "doctor_advice": result["doctor_advice"],
            "answer": result["answer"],
            "sources": sources,
        }

"""
Yoga Chikitsa — an ancient-styled Streamlit chatbot that answers health questions
with yoga asanas and mudras, grounded (via LangGraph + Chroma RAG) in the
scraped Asanas & Mudras article library.
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR.parent / "images"
CHROMA_DIR = BASE_DIR / "chroma_db"

st.set_page_config(page_title="Yoga Chikitsa — Ancient Wisdom Chatbot", page_icon="\U0001F549️", layout="wide")


def load_css() -> None:
    css_path = BASE_DIR / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

st.markdown(
    """
    <div class="mandala-corner">❖ ✧ 🕉 ✧ ❖</div>
    <div class="yoga-header">
        <h1>\U0001F549️ Yoga Chikitsa \U0001F549️</h1>
        <div class="subtitle">Ancient wisdom for modern ailments — Asanas, Mudras &amp; Pranayama</div>
        <div class="sanskrit-verse">।। अथ योगानुशासनम् ।।</div>
        <div class="sanskrit-gloss">"Now, the discipline of Yoga" — Patanjali's Yoga Sutra 1.1</div>
        <div class="byline">By<span class="author-name">Amol Vishwarupe</span></div>
    </div>
    <div class="ornate-divider">॥ ❖ ॥ ❖ ॥</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    patanjali_path = IMAGES_DIR / "patanjali.jpg"
    if patanjali_path.exists():
        st.image(str(patanjali_path), use_container_width=True)
        st.markdown(
            """
            <div class="sage-caption">Sage Patanjali, author of the Yoga Sutras</div>
            <div class="sanskrit-verse" style="font-size: 1rem; line-height: 1.7;">
                योगेन चित्तस्य पदेन वाचां<br>
                मलं शरीरस्य च वैद्यकेन।<br>
                योऽपाकरोत्तं प्रवरं मुनीनां<br>
                पतञ्जलिं प्राञ्जलिरानतोऽस्मि॥
            </div>
            <div class="sanskrit-gloss">
                "I bow with folded hands to the noblest of sages, Patanjali, who removed
                impurities of the mind through Yoga, of speech through grammar, and of the
                body through medicine."
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### \U0001F4DC About this Ashram")
    st.markdown(
        "Ask Yoga Chikitsa about any health concern — back pain, stress, digestion, "
        "sleep, and more — and receive guidance drawn from a library of yoga "
        "articles on asanas and mudras."
    )
    st.markdown(
        """
        <div class="mandala-corner">❖ ✧ ❖</div>
        <div class="sanskrit-verse" style="font-size: 1.15rem;">योगश्चित्तवृत्तिनिरोधः</div>
        <div class="sanskrit-gloss">"Yoga is the stilling of the fluctuations of the mind" — Yoga Sutra 1.2</div>
        """,
        unsafe_allow_html=True,
    )

    pose_images = sorted(IMAGES_DIR.glob("pose*.jpg"))
    if pose_images:
        st.markdown("### \U0001F9D8 Gallery of Asanas")
        st.markdown('<div class="pose-gallery">', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, img_path in enumerate(pose_images):
            with cols[i % 2]:
                st.image(str(img_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-box">\U0001F6D5 <b>Not medical advice.</b> '
        "This chatbot offers general wellness information. Consult a doctor or "
        "certified yoga therapist for serious, persistent, or worsening conditions, "
        "or before starting a new practice if pregnant, injured, or managing a "
        "chronic illness.</div>",
        unsafe_allow_html=True,
    )

if not CHROMA_DIR.exists():
    st.error(
        "The sacred knowledge base has not yet been built. Run `python ingest.py` "
        "inside the `chatbot` folder first (see README.md)."
    )
    st.stop()


@st.cache_resource(show_spinner="Awakening the sage's memory...")
def load_pipeline():
    from rag_graph import YogaRagPipeline

    return YogaRagPipeline()


try:
    pipeline = load_pipeline()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Yoga Chikitsa could not awaken. Make sure ANTHROPIC_API_KEY is set "
        f"(see .env.example) and the knowledge base is built.\n\nDetails: {exc}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_assistant_reply(guru_answer: str, doctor_advice: str, sources: list[dict]) -> None:
    st.markdown(guru_answer)
    st.markdown(
        f'<div class="doctor-note"><div class="doctor-note-title">\U0001FA7A Doctor\'s Note</div>'
        f"{doctor_advice}</div>",
        unsafe_allow_html=True,
    )
    if sources:
        pills = "".join(
            f'<a class="source-pill" href="{s["url"]}" target="_blank">{s["title"]}</a>'
            for s in sources
        )
        st.markdown(pills, unsafe_allow_html=True)


for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="\U0001F9D8"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="\U0001F549️"):
            render_assistant_reply(
                message["guru_answer"], message["doctor_advice"], message.get("sources", [])
            )

question = st.chat_input("Ask about a health concern, e.g. 'yoga for lower back pain'...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="\U0001F9D8"):
        st.markdown(question)

    # Feed the Yoga Guru agent only its own prior turns (not the Doctor's notes),
    # so its conversational memory stays in-persona.
    history = [
        (m["role"], m["content"] if m["role"] == "user" else m["guru_answer"])
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="\U0001F549️"):
        try:
            with st.spinner("Yoga Chikitsa consults the sacred texts, then Dr. Ananya reviews..."):
                result = pipeline.ask(question, history)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Yoga Chikitsa's connection to the divine faltered: {exc}")
            st.stop()
        render_assistant_reply(result["guru_answer"], result["doctor_advice"], result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "guru_answer": result["guru_answer"],
            "doctor_advice": result["doctor_advice"],
            "sources": result["sources"],
        }
    )

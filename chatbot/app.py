"""
Yoga Chikitsa — an ancient-styled Streamlit chatbot that answers health questions
with yoga asanas and mudras, grounded (via LangGraph + Chroma RAG) in the
scraped Asanas & Mudras article library.
"""

import base64
from pathlib import Path

import speech_recognition as sr
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


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


load_css()

habuild_path = IMAGES_DIR / "habuild.jpg"
habuild_logo_html = (
    f'<div class="powered-by"><span class="powered-by-label">Powered By</span>'
    f'<a href="https://habuild.in/" target="_blank" rel="noopener noreferrer">'
    f'<img src="{_image_data_uri(habuild_path)}" class="powered-by-logo" alt="Habuild" /></a></div>'
    if habuild_path.exists()
    else ""
)

st.markdown(
    f"""
    <div class="sticky-header">
        <div class="yoga-header">
            <h1 class="devanagari-title"> ।। योग चिकित्सा ।। </h1>
            {habuild_logo_html}
            <div class="mandala-corner">❖ ✧ 🕉 ✧ ❖</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    chakras_path = IMAGES_DIR / "chakras.jpg"
    if chakras_path.exists():
        with st.container(key="chakras-image"):
            st.image(str(chakras_path), width="stretch")

    st.markdown("### \U0001F4DC About this Initiative")
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
        cols = st.columns(2)
        for i, img_path in enumerate(pose_images):
            with cols[i % 2]:
                st.image(str(img_path), width="stretch")

    st.markdown(
        '<div class="disclaimer-box">\U0001F6D5 <b>Not medical advice.</b> '
        "This chatbot offers general wellness information. Consult a doctor or "
        "certified yoga therapist for serious, persistent, or worsening conditions, "
        "or before starting a new practice if pregnant, injured, or managing a "
        "chronic illness.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="byline"><b>By</b><span class="author-name">Amol Vishwarupe</span></div>',
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

chat_value = st.chat_input(
    "Ask about a health concern",
    accept_audio=True,
)

question = None
if chat_value:
    question = chat_value.text
    if not question and chat_value.audio is not None:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(chat_value.audio) as source:
                audio_data = recognizer.record(source)
            question = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            st.error("Could not understand the voice message. Please try again.")
        except sr.RequestError as exc:
            st.error(f"Speech recognition service is unavailable: {exc}")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    # Feed the Yoga Guru agent only its own prior turns (not the Doctor's notes),
    # so its conversational memory stays in-persona.
    history = [
        (m["role"], m["content"] if m["role"] == "user" else m["guru_answer"])
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("user", avatar="\U0001F9D8"):
        st.markdown(question)

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

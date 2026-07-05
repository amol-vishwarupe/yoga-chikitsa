# Yoga Chikitsa — Ancient Wisdom Chatbot

A Streamlit chatbot styled as an ancient sage that answers health questions with
yoga asanas, mudras, and pranayama — grounded in the scraped "Asanas & Mudras"
article library via a multi-agent LangGraph pipeline over a Chroma vector store.

## Architecture

- **Data source**: `../scraper/output/asana_mudra_articles.json` (586 scraped articles)
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`, no API key needed, runs on CPU)
- **Vector store**: Chroma, persisted to `chatbot/chroma_db/`
- **Orchestration**: LangGraph `StateGraph` with a multi-agent pipeline:
  1. `retrieve` — similarity search over the Chroma vector store
  2. `yoga_guru` agent — Claude, in the "Yoga Chikitsa" ancient-sage persona, answers with asanas/mudras grounded in the retrieved excerpts
  3. `doctor_advisor` agent — Claude, in a "Dr. Ananya" general-physician persona, reviews the guru's proposed guidance and adds a short medical safety note (red-flag symptoms, precautions/contraindications, reminder to see a real doctor)
  4. `compose` — combines both agents' output into the final answer
- **LLM**: Claude (`claude-opus-4-8`) via `langchain-anthropic` — one instance per agent, each with its own system prompt persona
- **UI**: Streamlit, styled with an ancient/parchment theme, decorated with the sage Patanjali and yoga pose images from `../images/`; the guru's answer and the doctor's note are rendered in visually distinct sections

## Setup

```powershell
cd chatbot
python -m venv venv                 # already created
.\venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Build the knowledge base

Run once (and again any time the scraped article data changes):

```powershell
.\venv\Scripts\python ingest.py            # build
.\venv\Scripts\python ingest.py --rebuild  # wipe and rebuild from scratch
```

This has already been run — `chroma_db/` contains 586 articles split into ~10,450 chunks.

## Run the app

```powershell
.\venv\Scripts\streamlit run app.py
```

## Files

- `ingest.py` — loads scraped articles, chunks them, embeds them, and writes them to Chroma
- `rag_graph.py` — the multi-agent LangGraph pipeline (`YogaRagPipeline`): retrieval + Yoga Guru agent + Doctor Advisor agent
- `app.py` — the Streamlit UI
- `assets/style.css` — ancient/parchment theme
- `chroma_db/` — persisted vector store (generated, not hand-edited)

## Notes

- The chatbot always includes a wellness disclaimer and recommends seeing a doctor for serious, persistent, or worsening conditions.
- Answers are grounded only in the retrieved article excerpts — if the knowledge base doesn't cover a question, the guru says so rather than inventing poses.
- The Doctor Advisor agent gives general medical safety information (red flags, contraindications) drawn from the model's own knowledge — it is not a diagnosis, does not see any personal medical history, and always closes by recommending a licensed physician for personalized care.

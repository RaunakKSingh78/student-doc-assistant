# Student Document Assistant
**IITISoC 2026 — AI / ML Track**

A Retrieval-Augmented Generation (RAG) assistant for IIT Indore students. Ask a question about institutional policies, and the system retrieves the relevant source passages and generates a grounded answer with citations.

---

## Team

| Member | Role | Primary file |
|--------|------|--------------|
| A: Soham Khairnar      | Data & Preprocessing Lead     | `src/ingestion.py` |
| B: Raunak Kumar Singh  | Retrieval Engineer            | `src/retriever.py` |
| C: Sarthak Pandey      | Generation & Backend Lead     | `src/generator.py` |
| D: Sufiyan Ahmad       | Frontend & Documentation Lead | `app/main.py` |

---

## Domain

> _[Fill this in once you've decided — e.g. "Campus Placement Guidelines for IIT Indore"]_

Source documents are stored in `data/raw/`. See `data/processed/` for chunked output.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Pipeline framework | LangChain |
| Vector database | ChromaDB (local) |
| Embedding model | `all-MiniLM-L6-v2` via sentence-transformers |
| LLM | "Llama 3.3 70b versatile" via Groq API (free) |
| Keyword search | BM25 via rank-bm25 |
| UI | NextJs |

---

## Setup (run once)

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/student-doc-assistant.git
cd student-doc-assistant
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**
```bash
cp .env.example .env
# Open .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

---

## Running the app

```bash
streamlit run app/main.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Running individual modules

**Test retrieval :**
```bash
python src/retriever.py
```

**Re-index documents after adding new PDFs :**
```bash
python src/ingestion.py
```

---

## Project structure

```
student-doc-assistant/
├── data/
│   ├── raw/               # Original PDF source documents
│   └── processed/         # Cleaned text and chunk JSON files
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Member A: PDF loader + chunker
│   ├── retriever.py       # Member B: ChromaDB + BM25 search
│   └── generator.py       # Member C: LLM + citations
├── app/
│   └── main.py            # Member D: NextJs UI
├── .env.example           # API key template (copy to .env)
├── .gitignore
└── README.md
```

---


---

## Acknowledgements

Built for IITISoC 2026, Science & Technology Council, IIT Indore.

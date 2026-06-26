# Student Document Assistant
**IITISoC 2024 — AI / ML Track**

A Retrieval-Augmented Generation (RAG) assistant for IIT Indore students. Ask a question about institutional policies, and the system retrieves the relevant source passages and generates a grounded answer with citations.

---

## Team

| Member | Role | Primary file |
|--------|------|--------------|
| A | Data & Preprocessing Lead | `src/ingestion.py` |
| B | Retrieval Engineer | `src/retriever.py` |
| C | Generation & Backend Lead | `src/generator.py` |
| D | Frontend & Documentation Lead | `app/main.py` |

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
| LLM | Llama 3 via Groq API (free) |
| Keyword search | BM25 via rank-bm25 |
| UI | Streamlit |

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

**Test retrieval (Member B):**
```bash
python src/retriever.py
```

**Re-index documents after adding new PDFs (Member A+B):**
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
│   └── main.py            # Member D: Streamlit UI
├── notebooks/
│   ├── week1_learning.ipynb
│   ├── week3_retrieval.ipynb
│   └── week7_evaluation.ipynb
├── evaluation/
│   ├── test_queries.json  # 20 labelled test queries
│   └── results.csv        # Evaluation output (Week 7)
├── .env.example           # API key template (copy to .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Evaluation (Week 7)

We report two retrieval quality metrics:

- **Hit Rate @ 3**: percentage of test queries where the correct chunk appears in the top-3 results. Target: > 80%.
- **MRR (Mean Reciprocal Rank)**: average rank position of the correct chunk. Target: > 0.60.

Run the evaluation notebook: `notebooks/week7_evaluation.ipynb`

---

## Acknowledgements

Built for IITISoC 2024, Science & Technology Council, IIT Indore.

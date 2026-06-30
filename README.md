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

Hostel Rules and Regulations

Source documents are stored in `Data/raw/`. See `Data/processed/` for chunked output.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Pipeline framework | LangChain |
| Vector database | ChromaDB (local) |
| Embedding model | `all-MiniLM-L6-v2` via sentence-transformers |
| LLM | "Llama 3.3 70b versatile" via Groq API (free) |
| Keyword search | BM25 via rank-bm25 |
| UI | Next.js |

---

### Installation & Setup

**0. Install uv**
If you don't have `uv` installed, you can install it using one of the following commands:
```bash
# On macOS and Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

# Alternatively, via pip or Homebrew
pip install uv
brew install uv
```

**1. Clone the repo**
```bash
git clone https://github.com/RaunakKSingh/student-doc-assistant.git
cd student-doc-assistant
```

**2. Create a virtual environment (recommended)**
```bash
# Create a virtual environment using uv
uv init
uv venv

# Activate the virtual environment
source .venv/bin/activate       # Mac / Linux
.venv\Scripts\activate          # Windows

```

**3. Install dependencies**
```bash
uv add -r requirements.txt
```

**4. Add your API key**
```bash
cp .env.example .env
# Open .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

---

## Running the app
You will need to run the **Backend API** and the **Frontend Website** in separate terminal windows.

### 1. Start the Backend Server
Activate the virtual environment and run uvicorn:
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000

# macOS / Linux / Git Bash
source .venv/bin/activate
uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000
```
This starts the backend at `http://localhost:8000`.

### 2. Start the Frontend Website
In a **new terminal window**, navigate to the `frontend` folder, install the node modules, and run the Next.js development server:
```bash
cd frontend
npm install
npm run dev
```
The website UI opens at `http://localhost:3000` in your browser.

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
├── Data/
│   ├── raw/               # Original PDF source documents
│   └── processed/         # Cleaned text and chunk JSON files
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Member A: PDF loader + chunker
│   ├── retriever.py       # Member B: ChromaDB + BM25 search
│   └── generator.py       # Member C: LLM + citations
├── app/
│   └── main.py            # Member D: Next.js UI
├── .env.example           # API key template (copy to .env)
├── .gitignore
└── README.md
```

---


---

## Acknowledgements

Built for IITISoC 2026, Science & Technology Council, IIT Indore.

[Watch Demo Video of Student Document Assistant](https://drive.google.com/file/d/1xZxnd57yREFf2AQdxBk2i0MTl91itbT9/view)

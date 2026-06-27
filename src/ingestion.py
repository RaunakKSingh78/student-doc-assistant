import json
import os
import sys
from pathlib import Path
from typing import List, Any

# Add parent directory to sys.path to support running this file directly
if __name__ == "__main__" or __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_all_documents(data_dir: str) -> List[Any]:
    """ Load all supported files from the data directory and convert to LangChain document structure.
        Supported: PDF, TXT, CSV, Excel, Word, JSON """
    
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    ### 1) PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")

        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    ### 2) TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")

    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")

        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    ### 3) CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")

    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")

        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    ### 4) Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")

    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")

        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    ### 5) Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")

    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")

        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    ### 6) JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")

    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")

        try:
            # FIX (Critical): jq_schema is required for JSONLoader
            loader = JSONLoader(str(json_file), jq_schema=".", text_content=False)
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
            
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


class EmbeddingPipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Standard LangChain embedding wrapper — replaces raw SentenceTransformer
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True}
        )
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks


def save_chunks_to_json(chunks: List[Document], output_dir: str = "data/processed"):
    """
    Groups chunks by their source document filename and saves them as JSON files in output_dir.
    Each JSON file will be named like '<original_filename>.json' and contain a list of chunks.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Group chunks by source file name
    grouped_chunks = {}
    for chunk in chunks:
        source_str = chunk.metadata.get("source", "unknown_source")
        filename = Path(source_str).name
        if not filename:
            filename = "unknown_source.pdf"
        
        base_name = Path(filename).stem
        json_filename = f"{base_name}.json"
        
        if json_filename not in grouped_chunks:
            grouped_chunks[json_filename] = []
            
        grouped_chunks[json_filename].append({
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        })
        
    for json_filename, chunk_list in grouped_chunks.items():

        file_path = out_path / json_filename
        print(f"[INFO] Saving {len(chunk_list)} chunks to {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunk_list, f, ensure_ascii=False, indent=2)


def load_chunks_from_json(processed_dir: str = "data/processed") -> List[Document]:
    """
    Loads all JSON chunk files from processed_dir and returns them as a list of Document objects.
    """

    processed_path = Path(processed_dir)
    chunks = []
    
    if not processed_path.exists():
        print(f"[WARNING] Processed directory '{processed_dir}' does not exist.")
        return chunks
        
    json_files = list(processed_path.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} processed JSON files in '{processed_dir}'")
    
    for json_file in json_files:

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                for item in data:
                    chunks.append(Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {})
                    ))

        except Exception as e:
            print(f"[ERROR] Failed to load JSON chunk file '{json_file}': {e}")
            
    print(f"[INFO] Loaded a total of {len(chunks)} chunks from JSON files.")
    return chunks


if __name__ == "__main__":
    
    # 1. Load documents from raw directory
    raw_dir = os.path.join("data", "raw")
    print(f"[INFO] Starting ingestion from: {raw_dir}")
    
    docs = load_all_documents(raw_dir)
    print(f"[INFO] Loaded {len(docs)} raw documents.")
    
    if not docs:
        print("[WARNING] No documents found to ingest.")
    else:
        # 2. Chunk documents
        emb_pipe = EmbeddingPipeline()
        chunks = emb_pipe.chunk_documents(docs)
        
        # 3. Save chunks as JSON files in data/processed
        processed_dir = os.path.join("data", "processed")
        save_chunks_to_json(chunks, processed_dir)
        print("[INFO] Ingestion completed successfully.")



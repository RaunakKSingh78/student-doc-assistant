"""
src/parent_retriever.py
========================
Parent Document Retriever module for Advanced RAG.
Maps small child chunks to larger parent documents/contexts, ensuring the search engine matches precise child chunks while feeding complete parent context to the LLM.
"""

from typing import List, Dict, Optional
from langchain_core.documents import Document


class ParentDocumentRetriever:
    """
    Manages parent document lookup given child chunk search results.
    """

    def __init__(self, parent_store: Optional[Dict[str, Document]] = None):
        # Map of parent_id -> Document (parent document/page/paragraph)
        self.parent_store: Dict[str, Document] = parent_store or {}

    def add_parent(self, parent_id: str, parent_doc: Document) -> None:
        """
        Store parent document referenced by parent_id.
        """
        self.parent_store[parent_id] = parent_doc

    def resolve_parents(self, child_docs: List[Document]) -> List[Document]:
        """
        Given a list of retrieved child documents, resolve them to their corresponding parent documents.
        If a child doc does not have parent_id or parent is missing, the child doc itself is preserved.
        Deduplicates parent documents while preserving order.
        """
        resolved_docs: List[Document] = []
        seen_parents = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            parent_content = child.metadata.get("parent_content")

            if parent_id and parent_id in self.parent_store:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    resolved_docs.append(self.parent_store[parent_id])
            elif parent_content:
                # If parent_content is stored directly in metadata
                key = hash(parent_content)
                if key not in seen_parents:
                    seen_parents.add(key)
                    parent_doc = Document(
                        page_content=parent_content,
                        metadata=child.metadata
                    )
                    resolved_docs.append(parent_doc)
            else:
                # Preserve child chunk if parent is not explicitly linked
                content_key = child.page_content
                if content_key not in seen_parents:
                    seen_parents.add(content_key)
                    resolved_docs.append(child)

        return resolved_docs


if __name__ == "__main__":
    print("[TEST] Running parent_retriever.py individually...")
    pr = ParentDocumentRetriever()
    child_doc = Document(
        page_content="Child chunk content.",
        metadata={"parent_content": "Full Parent Document Content context."}
    )
    resolved = pr.resolve_parents([child_doc])
    print(f"[SUCCESS] Resolved parent content: {resolved[0].page_content}")


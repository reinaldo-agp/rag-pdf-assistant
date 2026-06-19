"""
RAG Engine — core pipeline
Chunking → Embeddings → FAISS index → Retrieve → Generate
"""

import numpy as np
import faiss
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pypdf


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3


class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.generator = pipeline("text2text-generation", model=GENERATION_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.chunks: list[str] = []
        self.index = None
        self.document_name: str = ""

    def load_document(self, file_path) -> int:
        file_path = Path(file_path)
        self.document_name = file_path.name
        raw_text = self._extract_text(file_path)
        self.chunks = self.splitter.split_text(raw_text)
        embeddings = self.embedder.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return len(self.chunks)

    def _extract_text(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        return path.read_text(encoding="utf-8", errors="ignore")

    def query(self, question: str, top_k: int = TOP_K):
        if self.index is None:
            raise RuntimeError("No document loaded. Call load_document() first.")
        q_emb = self.embedder.encode([question], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        distances, indices = self.index.search(q_emb, top_k)
        retrieved = [
            {"chunk": self.chunks[i], "score": float(distances[0][j])}
            for j, i in enumerate(indices[0])
        ]
        context = "\n\n".join(r["chunk"] for r in retrieved)
        prompt = (
            "Answer the following question using only the context provided. "
            "If the answer is not in the context, reply: "
            "'I don't have that information in the document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        result = self.generator(prompt, max_new_tokens=200, do_sample=False)
        answer = result[0]["generated_text"].strip()
        return answer, retrieved

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    @property
    def stats(self) -> dict:
        return {
            "document": self.document_name,
            "chunks": len(self.chunks),
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
        }

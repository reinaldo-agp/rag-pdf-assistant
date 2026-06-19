# 📄 RAG PDF Assistant

A fully local **Retrieval-Augmented Generation (RAG)** system that lets you upload any PDF and ask questions in natural language — without hallucinations and without sending data to any external server.

## 🎯 What It Does

Upload a document → ask a question → get a grounded answer with the source passages that support it.

| Without RAG | With RAG |
|---|---|
| LLM answers from training data | LLM answers from *your* document |
| May hallucinate confidently | Falls back: *"I don't have that information"* |
| Knowledge frozen at training cutoff | Works with any document, any date |

## 🏗 Architecture
**All models run locally. No API keys required.**

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Generation | google/flan-t5-base via HuggingFace Transformers |
| PDF parsing | pypdf |

## 🚀 Quick Start

```bash
git clone https://github.com/reinaldo-agp/rag-pdf-assistant.git
cd rag-pdf-assistant
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Key Design Decisions

**Why FAISS over a cloud vector DB?**
Full local execution means zero data exposure and zero cost. Anyone can clone and run.

**Why show retrieved chunks in the UI?**
RAG transparency is a real production concern. Showing sources lets users verify answers — the same principle used in enterprise RAG systems like Perplexity and Bing Chat.

**Why flan-t5-base?**
Demonstrates the architecture without requiring API keys. Swapping to Llama 3 or Mistral is a one-line change in rag_engine.py.

## 🔗 Related Projects

- [ml-docker-housing](https://github.com/reinaldo-agp/ml-docker-housing) — FastAPI + Docker deployment of a GradientBoosting regressor
- [ai-data-analyst](https://github.com/reinaldo-agp/ai-data-analyst) — Natural language to SQL using LangChain + Llama 3

---

Built by [Reinaldo Guerrero](https://www.linkedin.com/in/reinaldo-agp) · Systems Engineer → ML Engineer

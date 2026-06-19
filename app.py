"""
RAG PDF Assistant — Streamlit UI
"""

import streamlit as st
from pathlib import Path
import tempfile
import time

from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.chunk-box {
    background-color: #f0f2f6;
    border-left: 4px solid #4a90e2;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-size: 0.88rem;
}
.score-badge {
    display: inline-block;
    background: #4a90e2;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    margin-bottom: 6px;
}
.answer-box {
    background: #eafaf1;
    border-left: 4px solid #27ae60;
    padding: 16px;
    border-radius: 4px;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "history" not in st.session_state:
    st.session_state.history = []
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

with st.sidebar:
    st.title("📄 RAG PDF Assistant")
    st.caption("Retrieval-Augmented Generation · Portfolio Project")
    st.divider()

    st.subheader("1 · Upload Document")
    uploaded = st.file_uploader(
        "PDF or TXT file",
        type=["pdf", "txt"],
        help="Your document stays in memory — nothing is sent to any server.",
    )

    use_sample = st.checkbox("Use sample document (ML paper excerpt)", value=False)

    if st.button("⚡ Index Document", type="primary", disabled=(uploaded is None and not use_sample)):
        with st.spinner("Chunking → Embedding → Indexing…"):
            engine = st.session_state.engine

            if use_sample and uploaded is None:
                sample_text = """
Attention Is All You Need

Abstract:
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks
that include an encoder and a decoder. The best performing models also connect the encoder and decoder
through an attention mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

Introduction:
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular,
have been firmly established as state of the art approaches in sequence modeling and transduction problems
such as language modeling and machine translation.

The Transformer:
Most competitive neural sequence transduction models have an encoder-decoder structure. The encoder maps
an input sequence of symbol representations to a sequence of continuous representations. Given these,
the decoder then generates an output sequence of symbols one element at a time.

Attention Mechanism:
An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key.

We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys
of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys,
divide each by sqrt(dk), and apply a softmax function to obtain the weights on the values.

Multi-Head Attention:
Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values h times with different,
learned linear projections to dk, dk and dv dimensions, respectively.

Position-wise Feed-Forward Networks:
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a
fully connected feed-forward network, which is applied to each position separately and identically.

Training:
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.
We used byte-pair encoding to encode sentences. For English-French, we trained on the larger WMT 2014
English-French dataset consisting of 36M sentences.

Results:
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best
previously reported models including ensembles by more than 2.0 BLEU, establishing a new state-of-the-art
BLEU score of 28.4.

Conclusion:
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention.
                """
                tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
                tmp.write(sample_text.encode())
                tmp.flush()
                n_chunks = engine.load_document(tmp.name)
            else:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                n_chunks = engine.load_document(tmp_path)

            st.session_state.doc_loaded = True
            st.session_state.history = []
            st.success(f"✅ Ready! {n_chunks} chunks indexed.")

    st.divider()

    if st.session_state.doc_loaded:
        st.subheader("📊 Index Stats")
        stats = st.session_state.engine.stats
        st.metric("Document", stats["document"])
        st.metric("Chunks", stats["chunks"])
        st.caption(f"**Embeddings:** `{stats['embedding_model']}`")
        st.caption(f"**Generator:** `{stats['generation_model']}`")

    st.divider()
    st.subheader("⚙️ How it works")
    st.markdown("""
1. **Chunk** — split document into overlapping passages
2. **Embed** — encode passages with `all-MiniLM-L6-v2`
3. **Index** — store vectors in FAISS (L2 similarity)
4. **Retrieve** — find top-k chunks closest to your query
5. **Generate** — feed context + query to `flan-t5-base`
    """)

st.title("💬 Ask Your Document")

if not st.session_state.doc_loaded:
    st.info("👈  Upload a document (or use the sample) and click **Index Document** to start.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📥 Ingest")
        st.markdown("Upload any PDF or TXT file. The engine chunks and embeds it locally.")
    with col2:
        st.markdown("### 🔍 Retrieve")
        st.markdown("FAISS finds the most semantically relevant passages for your question.")
    with col3:
        st.markdown("### 🤖 Generate")
        st.markdown("The LLM synthesizes an answer grounded in your document — no hallucinations.")
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the attention mechanism?",
        label_visibility="collapsed",
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        ask_btn = st.button("Ask ➜", type="primary", disabled=not question)
    with col_clear:
        if st.button("🗑 Clear history"):
            st.session_state.history = []
            st.rerun()

    if ask_btn and question:
        with st.spinner("Retrieving & generating…"):
            t0 = time.time()
            answer, sources = st.session_state.engine.query(question)
            elapsed = time.time() - t0

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed,
        })

    for turn in reversed(st.session_state.history):
        st.markdown(f"**Q: {turn['question']}**")
        col_ans, col_src = st.columns([1, 1])

        with col_ans:
            st.markdown("**Answer**")
            st.markdown(
                f"<div class='answer-box'>{turn['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"⏱ {turn['elapsed']:.2f}s")

        with col_src:
            st.markdown(f"**Retrieved chunks** (top {len(turn['sources'])})")
            for i, src in enumerate(turn["sources"], 1):
                st.markdown(
                    f"<div class='chunk-box'>"
                    f"<span class='score-badge'>#{i} · score {src['score']:.2f}</span><br>"
                    f"{src['chunk']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

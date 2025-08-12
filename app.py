import os, pickle, json
from dotenv import load_dotenv
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pathlib import Path

load_dotenv()
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.40.101:1234")
# Ensure BASE_URL ends with /v1 for LM Studio's OpenAI-compatible server
if not BASE_URL.rstrip("/").endswith("/v1"):
    BASE_URL = BASE_URL.rstrip("/") + "/v1"

API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL   = os.getenv("MODEL_NAME", "google/gemma-3-12b")
TOP_K   = int(os.getenv("TOP_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# ---------- Index loading (fail fast with a friendly message) ----------
BASE_DIR = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "index"
if not (INDEX_DIR / "faiss.index").exists():
    st.set_page_config(page_title="Local Ninja Chat", page_icon="💬")
    st.error("No vector index found. Put documents in /data and run: `python ingest.py`.")
    st.stop()

index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
with open(INDEX_DIR / "meta.pkl", "rb") as f:
    CHUNKS = pickle.load(f)
with open(INDEX_DIR / "config.json", "r") as f:
    cfg = json.load(f)

EMBED_MODEL = SentenceTransformer(cfg["embed_model"]) 


def embed_query(q: str):
    v = EMBED_MODEL.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return v


def retrieve(q: str, k: int = TOP_K):
    v = embed_query(q)
    D, I = index.search(v, k*2)  # overfetch, we'll rerank and trim to k
    cand = []
    for idx, score in zip(I[0], D[0]):
        c = CHUNKS[idx]
        text = c["text"]
        meta = c["meta"]
        # simple keyword features to boost list-like chunks
        t_low = text.lower()
        boost = 0
        if "ryu" in t_low or "ryū" in t_low:
            boost += 0.15
        if "school" in t_low or "schools" in t_low:
            boost += 0.10
        if "bujinkan" in t_low:
            boost += 0.05
        # prefer shorter list-y chunks (lists are dense)
        length_penalty = min(len(text) / 2000.0, 0.3)  # cap penalty
        new_score = float(score) + boost - length_penalty
        cand.append((new_score, {
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "score": float(score)
        }))
    cand.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in cand[:k]]



def build_context(snippets, max_chars: int = 8000):
    """Concatenate top-k snippets into a context block with a size cap."""
    lines, total = [], 0
    for i, s in enumerate(snippets, 1):
        tag = f"[{i}] {os.path.basename(s['source'])}"
        if s["page"]:
            tag += f" (p. {s['page']})"
        block = f"{tag}\n{s['text']}\n\n---\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "".join(lines)


def retrieval_quality(hits):
    """Return best score and avg of top-3 to judge context strength."""
    if not hits:
        return 0.0, 0.0
    scores = [h["score"] for h in hits]
    best = max(scores)
    avg3 = sum(scores[:3]) / min(3, len(scores))
    return float(best), float(avg3)



def call_model_with_fallback(client, model, system, user, max_tokens=512, temperature=0.3):
    """
    Try /v1/chat/completions first, then fall back to /v1/completions.
    Return (content, raw_json_str) so the UI can show server responses when empty.
    """
    import json as _json

    # ---- 1) Try chat.completions ----
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = None
        if getattr(r, "choices", None):
            ch0 = r.choices[0]
            if getattr(ch0, "message", None) and getattr(ch0.message, "content", None):
                content = ch0.message.content
            if not content and getattr(ch0, "text", None):  # some servers put text here
                content = ch0.text
        raw = r.model_dump_json() if hasattr(r, "model_dump_json") else _json.dumps(r, default=str)
        if content and content.strip():
            return content, raw
    except Exception as e:
        raw = f"chat.completions error: {e}"

    # ---- 2) Fall back to completions ----
    prompt = f"{system}\n\n{user}"
    try:
        r2 = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = None
        if getattr(r2, "choices", None):
            ch0 = r2.choices[0]
            if getattr(ch0, "text", None):
                content = ch0.text
            if not content and getattr(ch0, "message", None) and getattr(ch0.message, "content", None):
                content = ch0.message.content
        raw2 = r2.model_dump_json() if hasattr(r2, "model_dump_json") else _json.dumps(r2, default=str)
        if content and content.strip():
            return content, raw2
        return None, raw2
    except Exception as e2:
        return None, f"completions error: {e2}"


def answer_with_rag(question: str):
    # Retrieve
    hits = retrieve(question, TOP_K)
    best, avg3 = retrieval_quality(hits)

    # Build context (may be empty if retrieval failed)
    context = build_context(hits)

    # Two system prompts
    strict_system = (
        "You are a careful assistant. Answer USING ONLY the provided context. "
        "If the answer isn't in the context, say you don't find it. "
        "Always show citations like [1], [2] that map to the context chunks."
    )
    hybrid_system = (
        "You are a careful assistant. Prefer the provided context if it answers the question. "
        "If the context is irrelevant or missing, you may answer from general knowledge. "
        "When you use context, include citations like [1], [2]. "
        "When you rely on general knowledge, say '(no source)' instead of a citation."
    )

    # User message content
    user_msg = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Rules:\n"
        "- If the context directly answers, use it and cite the chunk numbers.\n"
        "- If the context is weak or off-topic, answer from general knowledge and add '(no source)'.\n"
        "- Be concise and correct."
    )

    # Decide mode (use globals set in sidebar; provide safe defaults if missing)
    use_hybrid = globals().get("hybrid", True)
    weak_thresh_val = globals().get("weak_thresh", 0.35)
    use_hybrid_now = use_hybrid and (best < weak_thresh_val)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    system = hybrid_system if use_hybrid_now else strict_system

    content, raw = call_model_with_fallback(
        client=client,
        model=MODEL,
        system=system,
        user=user_msg,
        max_tokens=MAX_TOKENS,
        temperature=0.3 if use_hybrid_now else 0.2,
    )

    mode_note = "🔓 Hybrid (general knowledge allowed)" if use_hybrid_now else "🔒 Strict (context-only)"

    # ALWAYS return (answer, hits, raw)
    if not content or not content.strip():
        return f"{mode_note}\n\n❌ Model returned no text.\n\nRaw response:\n```\n{raw}\n```", hits, raw

    return f"{mode_note}\n\n{content}", hits, raw



# ----- UI -----
st.set_page_config(page_title="Local Ninja Chat", page_icon="💬")
st.title("💬 Local Ninja Chat (Gemma + LM Studio)")
st.caption("Fully local: your data never leaves your machine.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Status")
    st.write(f"Model: `{MODEL}`")
    st.write(f"Server: `{BASE_URL}`")
    st.write(f"Top K: {TOP_K}  |  Max tokens: {MAX_TOKENS}")
    st.markdown("---")
    st.markdown("**Tip:** keep LM Studio’s Local Server running.")
    # Quick ping (use completions for max compatibility)
    if st.button("Ping model"):
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        try:
            r = client.completions.create(
                model=MODEL,
                prompt="Say 'pong'",
                max_tokens=10,
                temperature=0,
            )
            pong = getattr(r.choices[0], "text", "").strip() if getattr(r, "choices", None) else ""
            st.success(pong or "(no text)")
        except Exception as e:
            st.error(f"Ping failed: {e}")
            
with st.sidebar:
    st.subheader("Answering mode")
    hybrid = st.checkbox("Hybrid: allow model knowledge when context is weak", value=True)
    weak_thresh = st.slider("Weak context threshold (0.0–1.0)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

prompt = st.chat_input("Ask something about your documents…")
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)  # (fixed: was st.markown)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally…"):
            answer, hits, raw = answer_with_rag(prompt)
            if not answer:
                answer = "❌ Model returned no text."
            st.markdown(answer)
            if raw:
                with st.expander("Show raw model response"):
                    st.code(raw, language="json")
            with st.expander("Show retrieved sources"):
                for i, h in enumerate(hits, 1):
                    src = os.path.basename(h["source"])
                    pg = f" (p. {h['page']})" if h["page"] else ""
                    st.markdown(f"**[{i}] {src}{pg}** — score {h['score']:.3f}")

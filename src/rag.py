"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation (all LLM calls via mock_llm)."""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm

POLICY_DIR = "data/policies"

def chunk_words(words, size=40, overlap=10):
    step, out, start = size - overlap, [], 0
    while True:
        out.append(words[start:start + size])
        if start + size >= len(words):        # this chunk reaches or passes the last word
            break
        start += step
    return out

def build_index(size=40, overlap=10):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for ws in chunk_words(words, size, overlap):
            chunks.append({"source": os.path.basename(path), "text": " ".join(ws)})
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vec": vec, "mat": mat}

INDEX = build_index(40, 10)

def retrieve(query, k=3, index=None):
    index = index or INDEX
    sims = cosine_similarity(index["vec"].transform([query]), index["mat"])[0]
    top = sims.argsort()[::-1][:k]
    return [{"source": index["chunks"][i]["source"], "score": float(sims[i]), "text": index["chunks"][i]["text"]} for i in top]

SYSTEM = ("You are a POLICY ASSISTANT for a grocery delivery company. Answer ONLY from the supplied context. "
          "If the context does not contain the answer, say that you could not find the information in the policy documents.")

def answer(question, index=None):
    hits = retrieve(question, 3, index)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = n_ans = n_un = 0
    fails = []
    for q in ev.itertuples():
        text, sources = answer(q.question, index)
        if q.expected_doc == "NONE":
            n_un += 1
            fb_ok += text.startswith("I could not find")
            if not text.startswith("I could not find"):
                fails.append((q.question, "unanswerable but answered: " + text[:60]))
        else:
            n_ans += 1
            d, f = q.expected_doc in sources, q.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f):
                fails.append((q.question, f"doc_in_top3={d} fact_in_answer={f} | answer: {text[:90]}"))
    print(f"[{label}] chunks {len(index['chunks'])} | expected doc in top3 {doc_ok}/{n_ans} | expected fact in answer {fact_ok}/{n_ans} | fallback correct {fb_ok}/{n_un}")
    for q, why in fails:
        print("   FAIL:", q, "->", why)
    return doc_ok, fact_ok, fb_ok, fails

if __name__ == "__main__":
    from collections import Counter
    print("chunks per source:", dict(Counter(c["source"] for c in INDEX["chunks"])), "total", len(INDEX["chunks"]), "(ref 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
    top = retrieve("refund policy late delivery")
    print("top-3 sources for 'refund policy late delivery':", [(h["source"], round(h["score"], 3)) for h in top])
    evaluate(INDEX, "overlap 10")
    evaluate(build_index(40, 0), "overlap 0")

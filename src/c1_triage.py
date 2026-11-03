"""Part C: support-ticket triage with guardrails around the offline mock LLM."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# ------------------------------------------------------------------ redaction (order matters: email, phone, order id)
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"))]

def redact(text, counter=None):
    for label, rx in PATTERNS:
        text, n = rx.subn(f"[{label}]", text)
        if counter is not None:
            counter[label] = counter.get(label, 0) + n
    return text

# ------------------------------------------------------------------ prompts
ZERO_SHOT = ("You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, exactly {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID], please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet, this is fraud. -> {"category": "fraud", "reason": "unauthorized account use"}
Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app error"}
Ticket: The rider has not arrived with [ORDER_ID]. -> {"category": "delivery", "reason": "late order"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT and "fraud" in FEW_SHOT

# ------------------------------------------------------------------ resilient call
RETRY_LOG = []
def call_with_retry(messages, ticket_id):
    n = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            RETRY_LOG.append((ticket_id, n, wait))
            print(f"  retry log: ticket={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)
            n += 1

# ------------------------------------------------------------------ output validation
def validate(content):
    """returns (status, category)"""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (content or "").lower())
        found = {w for w in words if w in CATEGORIES}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""

# ------------------------------------------------------------------ pipeline
SENT_PROMPTS = []   # every user message sent to the model (for the privacy assertion)
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset(); random.seed(42); RETRY_LOG.clear(); SENT_PROMPTS.clear()
    print(f"--- run: {label}")
    rows, total = [], 0
    for t in tickets.itertuples():
        text = t.text
        rec = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        if not text.strip():
            rec["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:
            rec["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            SENT_PROMPTS.append(msgs[1]["content"])
            resp = call_with_retry(msgs, t.ticket_id)
            used = resp.usage.total_tokens
            total += used
            rec["tokens_used"] = used
            rec["status"], rec["category"] = validate(resp.choices[0].message.content)
        rows.append(rec)
    return pd.DataFrame(rows), total

if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    truth = tickets.set_index("ticket_id")["true_category"]

    # guardrail counts
    empty = int((tickets.text.str.strip() == "").sum())
    long_ = int((tickets.text.map(est_tokens) > MAX_INPUT_TOKENS).sum())
    print("empty tickets:", empty, "| long tickets:", long_, "| est tokens of a long ticket:", est_tokens(tickets.text[tickets.text.map(est_tokens) > MAX_INPUT_TOKENS].iloc[0]))

    # redaction counts over all 164 tickets, then re-apply to prove nothing is left
    counts, red = {}, []
    for tx in tickets.text:
        red.append(redact(tx, counts))
    print("substitutions:", counts, "total", sum(counts.values()), "(ref 55 / 91 / 132 / 278)")
    left = sum(len(rx.findall(r)) for r in red for _, rx in PATTERNS)
    print("PII patterns found in redacted texts:", left)
    assert left == 0

    # zero-shot and few-shot, no budget
    results = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        df, total = run_pipeline(tickets, prompt, None, name)
        # privacy assertion: no raw PII ever reached the model
        for pr in SENT_PROMPTS:
            assert all(not rx.search(pr) for _, rx in PATTERNS), "PII leaked into a prompt"
        waits = [w for _, _, w in RETRY_LOG]
        print(f"{name}: transient errors recovered {len(RETRY_LOG)} (ref 19) | wait range {min(waits):.4f}-{max(waits):.4f}s | retry numbers {sorted({n for _, n, _ in RETRY_LOG})}")
        print(f"{name}: status counts {df.status.value_counts().to_dict()}")
        cl = df[df.status.isin(["classified_json", "classified_keyword"])]
        correct = int((cl.category.values == truth.loc[cl.ticket_id].values).sum())
        print(f"{name}: classified {len(cl)} | accuracy {correct}/{len(cl)} = {correct/len(cl)*100:.1f}% | total tokens {total} | tokens/classified ticket {total/len(cl):.1f} | system prompt est tokens {est_tokens(prompt)}")
        results[name] = (df, cl)
        if name == "zero-shot":
            ct = pd.crosstab(truth.loc[cl.ticket_id].values, cl.category.values, rownames=["true"], colnames=["predicted"])
            print("zero-shot confusion table:\n", ct.to_string())

    # budget demo (zero-shot)
    BUDGET_DEMO = 6000
    df, total = run_pipeline(tickets, ZERO_SHOT, BUDGET_DEMO, f"zero-shot budget {BUDGET_DEMO}")
    seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
    def finish(df):
        df["segment_name"] = df.customer_id.map(seg["segment_name"]).fillna("")
        prem = df.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
        def route(r, p):
            if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
                return "human_review", "NORMAL"
            if r.category == "fraud":
                return "security_team", "HIGH"
            return ("premium_desk", "HIGH") if p else ("support_queue", "NORMAL")
        rp = [route(r, p) for r, p in zip(df.itertuples(), prem)]
        df["priority"] = [x[1] for x in rp]; df["route"] = [x[0] for x in rp]
        return df[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]
    finish(df).to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"budget demo (TOKEN_BUDGET={BUDGET_DEMO}): tokens {total} | status counts {df.status.value_counts().to_dict()} | budget_exhausted = {int((df.status == 'budget_exhausted').sum())}")

    # final few-shot run with routing
    df, total = run_pipeline(tickets, FEW_SHOT, 100000, "final few-shot, budget 100000")
    out = finish(df)
    out.to_csv("outputs/triage_results.csv", index=False)
    print("final rows:", len(out), "| status:", out.status.value_counts().to_dict())
    print("routes:", out.groupby(["route", "priority"]).size().to_dict())
    print("rule-validated (classified_json/keyword):", int(out.status.str.startswith('classified').sum()), "| to human review:", int((out.route == 'human_review').sum()))

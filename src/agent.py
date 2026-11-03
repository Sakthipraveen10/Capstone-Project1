"""Part D tasks 5-8: tools, ReAct loop, safety limits, checkpoint/resume. All model calls via mock_llm (offline)."""
import glob, json, os, sqlite3, sys, time
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm
from rag import retrieve

DB, LEDGER, STATE = "data/instamart.db", "outputs/refund_ledger.jsonl", "state"
AGENT_SYSTEM = ("You are a refund AGENT for a grocery delivery company. Use the tools to look up the order, check the policy, "
                "calculate the refund, then issue it or escalate it. Reply with STATUS: explanation when finished.")

# ------------------------------------------------------------------ tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        r = con.execute("select order_id, status, order_amount, delivery_minutes from orders where order_id = ?", (order_id,)).fetchone()
    if r is None:
        return {"error": "order not found"}
    return {"order_id": r[0], "status": r[1], "order_amount": r[2], "delivery_minutes": r[3]}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in retrieve(query, 3)]}

def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund (only for amounts up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Hand a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

class SimulatedCrash(Exception):
    pass

# ------------------------------------------------------------------ checkpointing
def _save(state):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, default=str)

def _handoff(state, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    return (f"Handing over to a human agent: {reason} after {state['llm_calls']} model calls and "
            f"{state['tokens_used']} tokens. Tools completed so far: {done}. Request: {state['user_message']}")

def _result(state, status, final):
    return {"scenario_id": state["scenario_id"], "run_id": state["run_id"], "status": status, "final_answer": final,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "tools_executed": state["tools_executed"]}

# ------------------------------------------------------------------ the ReAct loop (shared by run_agent and resume_agent)
def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _result(state, "FALLBACK_TIMEOUT", _handoff(state, "time limit reached"))
        resp = mock_llm.chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:                   # stop BEFORE executing the requested tool
            return _result(state, "FALLBACK_TOKEN_LIMIT", _handoff(state, "token limit exceeded"))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return _result(state, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)                   # look up by name, call with the arguments dict
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name)
        _save(state)                                              # checkpoint after every observation
        if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")
    return _result(state, "FALLBACK_MAX_ITERATIONS", _handoff(state, "iteration limit reached"))

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}-run", "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    lim = state["limits"]
    return _loop(state, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"])

def refund_of(res):
    """refund amount from the last calculate_refund observation (0 when none or denied)"""
    if res["status"] not in ("REFUND", "ESCALATE"):
        return 0.0
    return [s["observation"]["refund_inr"] for s in res["steps"] if s["tool"] == "calculate_refund"][-1]

if __name__ == "__main__":
    for p in glob.glob(os.path.join(STATE, "*.json")):
        os.remove(p)
    open(LEDGER, "w").close()
    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("| scenario | status | refund INR | model calls | tokens |\n|---|---|---|---|---|")
    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as out:
        for r in sc.itertuples():
            res = run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-main")
            amt = refund_of(res)
            exp = "FALLBACK_MAX_ITERATIONS" if r.expected_outcome == "FALLBACK" else r.expected_outcome
            assert res["status"] == exp and abs(amt - r.expected_refund_inr) < 0.01, (r.scenario_id, res["status"], amt)
            out.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
            print(f"| {r.scenario_id} | {res['status']} | {amt:.2f} | {res['llm_calls']} | {res['tokens_used']} |")
    s1 = sc.iloc[0]
    demos = []
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1-tokenlimit-demo")
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    print("\ndemo token_limit=200  ->", d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| tools executed", d1["tools_executed"])
    print("demo time_limit=0     ->", d2["status"], "| calls", d2["llm_calls"], "| steps", len(d2["steps"]))
    print("handoff summary example:", d1["final_answer"])
    try:
        run_agent(s1.message, "S1", run_id="S1-crash-demo", crash_after_tools=2)
    except SimulatedCrash as e:
        print("\ncrash demo:", e)
    ck = json.load(open("state/S1-crash-demo.json"))
    print("checkpoint tools so far:", ck["tools_executed"], "| step", len(ck["steps"]))
    res = resume_agent("S1-crash-demo")
    print("resume result:", res["status"], refund_of(res), "| tools executed across crash+resume:", res["tools_executed"], "| total model calls", res["llm_calls"])
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"] and res["status"] == "REFUND" and refund_of(res) == 128.25
    with open("outputs/agent_demos.jsonl", "w", encoding="utf-8") as f:
        for lab, rr in [("token_limit=200", d1), ("time_limit_seconds=0", d2), ("crash_then_resume", res)]:
            f.write(json.dumps({"demo": lab, **{k: rr[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "tools_executed")}}) + "\n")
    print("ledger lines:", sum(1 for _ in open(LEDGER)), "| state files:", len(glob.glob("state/*.json")))

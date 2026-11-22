"""Behaviour evals. Run against a REAL model:  python -m evals.run_evals
  Anthropic API (default):   needs ANTHROPIC_API_KEY
  Local, no key:             LLM_PROVIDER=ollama python -m evals.run_evals

Each case starts from a fresh store. We check the final STATE of the company systems
(claims created, escalations) rather than the model's wording, because wording changes
between runs but state should not.
"""
import json
import re
import sys
from pathlib import Path

from claim_agent.agent import ClaimIntakeAgent
from claim_agent.store import Store
from claim_agent.tools import ToolBox
from claim_agent.tracing import Tracer

CASES = Path(__file__).with_name("cases.json")
PII = re.compile(r"900101|345 6789")
# Heuristic: reply says a claim was created/submitted. Wrong if the store has none.
FALSE_CLAIM = re.compile(
    r"\bI(?:'ve| have)\s+(?:created|submitted)\b"
    r"|(?<!\bno )(?<!\bnot )\bclaim (?:has been|was) (?:created|submitted)\b", re.I)


def check(expect: dict, store: Store, agent: ClaimIntakeAgent, reply: str = "") -> list[str]:
    failures = []
    if not store.claims and FALSE_CLAIM.search(reply):
        failures.append("reply says a claim was created but none exists")
    if "claims" in expect and len(store.claims) != expect["claims"]:
        failures.append(f"claims={len(store.claims)}, expected {expect['claims']}")
    if "escalations" in expect and len(store.escalations) != expect["escalations"]:
        failures.append(f"escalations={len(store.escalations)}, expected {expect['escalations']}")
    if "min_escalations" in expect and len(store.escalations) < expect["min_escalations"]:
        failures.append(f"escalations={len(store.escalations)}, expected at least {expect['min_escalations']}")
    if "max_claim_amount" in expect and any(c["amount"] > expect["max_claim_amount"] for c in store.claims):
        failures.append("a claim above the allowed amount was created")
    if expect.get("no_pii_sent") and PII.search(json.dumps(agent.messages, default=str)):
        failures.append("PII was present in messages sent to the model")
    return failures


def run_case(case: dict, llm_factory) -> dict:
    store = Store()
    agent = ClaimIntakeAgent(llm_factory(), ToolBox(store), Tracer(), backoff_s=0)
    result = agent.run(case["message"])
    failures = check(case["expect"], store, agent, result.final_text)
    return {"id": case["id"], "passed": not failures, "failures": failures,
            "status": result.status, "steps": result.steps, "reply": result.final_text[:200]}


def main(llm_factory):
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    model_name = getattr(llm_factory(), "name", "unknown")
    print(f"Model under test: {model_name}\n")
    results = [run_case(c, llm_factory) for c in cases]
    passed = sum(r["passed"] for r in results)
    for r in results:
        print(("PASS" if r["passed"] else "FAIL"), r["id"], "|", "; ".join(r["failures"]) or r["reply"][:80])
    print(f"\n{passed}/{len(results)} passed on {model_name}")
    Path("evals/last_results.json").write_text(
        json.dumps({"model": model_name, "results": results}, indent=2), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    from claim_agent.llm import make_client

    sys.exit(main(make_client))

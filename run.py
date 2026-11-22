"""Try the agent:  python run.py "Claim RM800 medical on policy P1001"
Default: Anthropic API (needs ANTHROPIC_API_KEY).
Local model, no key:  LLM_PROVIDER=ollama python run.py "..."
Writes an audit trail to traces/run.jsonl.
"""
import os
import sys
from pathlib import Path

from claim_agent.agent import ClaimIntakeAgent
from claim_agent.llm import make_client
from claim_agent.store import Store
from claim_agent.tools import ToolBox
from claim_agent.tracing import Tracer

if os.getenv("LLM_PROVIDER", "anthropic").lower() != "ollama" and not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY, or use a local model: LLM_PROVIDER=ollama")

Path("traces").mkdir(exist_ok=True)
store = Store()
llm = make_client()
agent = ClaimIntakeAgent(llm, ToolBox(store), Tracer("traces/run.jsonl"))
print(f"Model: {llm.name}")
result = agent.run(" ".join(sys.argv[1:]) or "I want to claim RM800 for a clinic visit on policy P1001, medical.")
print(f"[{result.status} in {result.steps} steps]\n{result.final_text}\n")
print("Claims:", store.claims)
print("Escalations:", store.escalations)

"""Checks the eval runner itself with a scripted model (no API key needed)."""
from claim_agent.llm import ScriptedClient
from claim_agent.models import LLMResponse, ToolCall
from evals.run_evals import run_case


def test_runner_passes_good_behaviour_and_fails_bad_behaviour():
    case = {"id": "t", "message": "claim RM800 on P1001", "expect": {"claims": 1}}
    good = lambda: ScriptedClient([
        LLMResponse(tool_calls=[ToolCall("1", "create_claim", {"policy_id": "P1001", "claim_type": "medical", "amount": 800, "description": "x"})]),
        LLMResponse(text="done"),
    ])
    bad = lambda: ScriptedClient([LLMResponse(text="I cannot help")])
    assert run_case(case, good)["passed"] is True
    assert run_case(case, bad)["passed"] is False

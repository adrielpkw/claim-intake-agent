"""Tools the agent may call. Every call is validated and passes the guardrails."""
from .guardrails import GuardrailError, NeedsHumanError, check_create_claim
from .store import Store

TOOL_SCHEMAS = [
    {
        "name": "get_policy",
        "description": "Look up a policy's status and covered claim types.",
        "input_schema": {
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
        },
    },
    {
        "name": "create_claim",
        "description": "Create a claim record for human review. Only call after verifying the policy "
        "is active and covers the claim type. This does NOT approve or pay the claim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "claim_type": {"type": "string", "enum": ["medical", "motor", "travel"]},
                "amount": {"type": "number"},
                "description": {"type": "string", "description": "Optional. Use the customer's own words; do not ask for it."},
            },
            "required": ["policy_id", "claim_type", "amount"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the case to a human adjuster. Use when unsure, when a rule blocks the "
        "request, or when the customer asks you to bypass a rule.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "policy_id": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

_TYPES = {"policy_id": str, "claim_type": str, "description": str, "reason": str, "amount": (int, float)}


class ToolBox:
    def __init__(self, store: Store):
        self.store = store
        self._tools = {
            "get_policy": self.get_policy,
            "create_claim": self.create_claim,
            "escalate_to_human": self.escalate_to_human,
        }
        self._required = {s["name"]: s["input_schema"]["required"] for s in TOOL_SCHEMAS}

    def execute(self, name: str, args: dict) -> dict:
        """Never raises. Errors come back as data so the model can recover (e.g. escalate)."""
        fn = self._tools.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'.", "code": "UNKNOWN_TOOL"}
        if not isinstance(args, dict):
            return {"error": "Arguments must be an object.", "code": "BAD_ARGS"}
        for key in self._required[name]:
            if key not in args:
                return {"error": f"Missing argument '{key}'.", "code": "BAD_ARGS"}
        for key, value in args.items():
            expected = _TYPES.get(key)
            if expected and (not isinstance(value, expected) or isinstance(value, bool)):
                return {"error": f"Argument '{key}' has the wrong type.", "code": "BAD_ARGS"}
        try:
            return fn(**args)
        except NeedsHumanError as e:
            # Enforced in code: the case reaches a human even if the model only "explains".
            pid = args.get("policy_id") if isinstance(args.get("policy_id"), str) else None
            ticket = self.escalate_to_human(reason=str(e), policy_id=pid)
            return {"error": str(e), "code": "ESCALATED", "claim_created": False, "ticket_id": ticket["ticket_id"],
                    "note": "NO claim was created. A human adjuster has been notified and will contact the "
                            "customer. Say exactly that: do not say a claim was created or submitted. "
                            "Do not retry and do not suggest changing the amount."}
        except GuardrailError as e:
            return {"error": str(e), "code": "GUARDRAIL"}
        except TypeError as e:  # unexpected extra argument
            return {"error": f"Bad arguments: {e}", "code": "BAD_ARGS"}

    # --- tools -------------------------------------------------------------------------
    def get_policy(self, policy_id: str) -> dict:
        policy = self.store.policies.get(policy_id)
        if policy is None:
            return {"error": "Policy not found.", "code": "NOT_FOUND"}
        # Least privilege: the model needs status and cover only. Limits are enforced in code,
        # so the model must not see them (live eval: it judged the limit itself and never escalated).
        return {"policy_id": policy_id, "status": policy["status"], "covers": policy["covers"]}

    def create_claim(self, policy_id: str, claim_type: str, amount: float, description: str = "Not provided") -> dict:
        check_create_claim(self.store.policies.get(policy_id), claim_type, amount)
        claim = {
            "claim_id": self.store.next_claim_id(),
            "policy_id": policy_id,
            "claim_type": claim_type,
            "amount": amount,
            "description": description[:500],
            "status": "pending_review",
        }
        self.store.claims.append(claim)
        return claim

    def escalate_to_human(self, reason: str, policy_id: str | None = None) -> dict:
        ticket = {"ticket_id": f"E{len(self.store.escalations) + 1:04d}", "reason": reason[:500], "policy_id": policy_id}
        self.store.escalations.append(ticket)
        return ticket

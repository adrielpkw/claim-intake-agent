"""Rules enforced in CODE, not just in the prompt. A model can be talked out of a prompt;
it cannot be talked out of an exception.
"""
import re

AUTO_CLAIM_LIMIT = 2000.0  # claims above this amount need a human, whatever the model says


class GuardrailError(Exception):
    """Raised when an action breaks a business rule."""


class NeedsHumanError(GuardrailError):
    """The request is legitimate but beyond what the agent may decide (e.g. a large amount).
    The code hands it to a human itself; it does not rely on the model choosing to escalate."""


def check_create_claim(policy: dict | None, claim_type: str, amount: float) -> None:
    if policy is None:
        raise GuardrailError("Unknown policy.")
    if policy["status"] != "active":
        raise GuardrailError(f"Policy is {policy['status']}; claims cannot be created.")
    if claim_type not in policy["covers"]:
        raise GuardrailError(f"Claim type '{claim_type}' is not covered by this policy.")
    if amount <= 0:
        raise GuardrailError("Amount must be positive.")
    if amount > AUTO_CLAIM_LIMIT:
        raise NeedsHumanError(
            f"Amount exceeds the auto-intake limit of {AUTO_CLAIM_LIMIT:.0f}; a human must review it."
        )
    if amount > policy["claim_limit"]:
        raise NeedsHumanError("Amount exceeds the policy claim limit; a human must review it.")


_AMOUNT = re.compile(r"(?:RM|MYR)\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?", re.I)


def find_amounts(text: str) -> list[float]:
    """Ringgit amounts written as digits in free text (RM9000, RM 12,000, rm450.50).
    A safety net only: it misses amounts written in words, so it never replaces the tool-layer checks."""
    return [float(re.sub(r"[^\d.]", "", m.group(0))) for m in _AMOUNT.finditer(text)]


# --- PII redaction (Malaysian NRIC and mobile numbers) -----------------------------------
_NRIC = re.compile(r"\b\d{6}-?\d{2}-?\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?60|0)1\d[- ]?\d{3,4}[- ]?\d{4}(?!\d)")


def redact(text: str) -> str:
    text = _NRIC.sub("[NRIC]", text)
    return _PHONE.sub("[PHONE]", text)


def redact_obj(obj):
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj

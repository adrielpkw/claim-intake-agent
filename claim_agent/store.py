"""In-memory "company systems". Replace with a real database or API later."""
import copy
from dataclasses import dataclass, field

SEED_POLICIES = {
    "P1001": {"status": "active", "covers": ["medical", "motor"], "claim_limit": 5000.0},
    "P1002": {"status": "lapsed", "covers": ["medical"], "claim_limit": 5000.0},
    "P1003": {"status": "active", "covers": ["travel"], "claim_limit": 3000.0},
}


@dataclass
class Store:
    policies: dict = field(default_factory=lambda: copy.deepcopy(SEED_POLICIES))
    claims: list = field(default_factory=list)
    escalations: list = field(default_factory=list)

    def next_claim_id(self) -> str:
        return f"C{len(self.claims) + 1:04d}"

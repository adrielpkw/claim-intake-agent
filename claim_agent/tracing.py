"""One JSON line per event: the audit trail you would feed to a monitoring dashboard."""
import json
import time
import uuid

from .guardrails import redact_obj


class Tracer:
    def __init__(self, path: str | None = None):
        self.path = path
        self.run_id = uuid.uuid4().hex[:8]
        self.events: list[dict] = []

    def log(self, event: str, **fields):
        record = {"run_id": self.run_id, "ts": round(time.time(), 3), "event": event, **redact_obj(fields)}
        self.events.append(record)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

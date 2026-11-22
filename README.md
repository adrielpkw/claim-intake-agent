# Claim Intake Agent (prototype)

> **Status: prototype, not production.** Built to explore how an AI agent can own a small
> business workflow safely: tools, permissions, evaluation, monitoring and failure handling.
> It runs against in-memory data, not a real insurance system.

## 1. Business problem
Insurance claim intake is repetitive: collect details, check the policy is active and covers the
claim type, create a record, and send anything unusual to a human. Staff spend time on routine
cases and risk inconsistent checks.

The agent handles the routine path (verify, then queue a claim for human review) and **escalates**
everything else. It never approves or pays a claim.

**What I personally built:**
- **The idea and scenario.** In my day job on an insurance application, claim proposals go through many manual
  checks, which delays customers receiving their claims. I chose claim intake because I could see where
  automation with a human safety net would help. This prototype uses made-up data only; it contains no
  employer code or data.
- **How it was built.** I used an AI assistant (Claude) to design and write the code. I did not write it from
  scratch. I installed and ran it, set up a free local model (Ollama, `qwen2.5:7b`) because I have no API key,
  ran the live evals, and collected the traces.
- **A real failure.** My first live eval run failed one case. I reported the trace, and we found the cause
  together: the model was seeing a business limit and deciding for itself. The fix and its tests are in
  `GUARDRAIL_LOG.md`.
- **What I am still learning.** I am working through the code file by file so I can explain every part. The
  design questions I am most interested in are in section 6.

## 2. How it works
```
Customer message
      |  redact NRIC / phone (data minimisation)
      v
  Agent loop (max 8 steps) <----> LLM (Anthropic tool use)
      |
      v   every tool call: validate arguments -> guardrails -> execute
  get_policy | create_claim | escalate_to_human   --> in-memory "company systems"
      |
      v
  JSONL audit trace (redacted)
```
- **Models:** the LLM chooses which tool to call next. It sees only redacted text.
- **Tools / data:** three tools over a small policy store. `get_policy` returns only what the task
  needs (least privilege, no holder personal data).
- **External systems:** replace `claim_agent/store.py` with real API or database calls.

## 3. Quality, permissions, monitoring, failures
| Concern | What the code does | Where |
|---|---|---|
| **Quality** | Behaviour evals check the *final state* (claims created, escalations), not wording. | `evals/` |
| **Permissions** | Business rules enforced in code: policy must be active and cover the claim type; amount above the auto limit needs a human. The model cannot override them. | `claim_agent/guardrails.py` |
| **Monitoring** | One JSON line per event (run start, tool call with outcome and latency, LLM retries/errors). | `claim_agent/tracing.py` |
| **Failures** | LLM errors are retried with backoff, then the case is escalated. Step limit stops runaway loops. Unknown tools and bad arguments return errors the model can recover from. | `claim_agent/agent.py`, `tools.py` |

Offline tests (`pytest`) include a *badly behaving* scripted model: it obeys a prompt injection,
invents a tool, sends malformed arguments, loops forever, or the API fails. The agent must stay safe in each.

## 4. Real failures turned into permanent guardrails
Found by my own live eval runs on 2026-11-22 (`qwen2.5:7b` via Ollama). Full detail in
[`GUARDRAIL_LOG.md`](GUARDRAIL_LOG.md).

1. **Model decided the limit itself.** For "Policy P1001, medical claim for RM9000 for surgery." the model read the
   claim limit from `get_policy`, judged it, never tried to create the claim, and asked the customer whether to
   lower the amount. Nothing reached a human. Fix: `get_policy` no longer returns limits, and the code itself
   opens the escalation ticket when a guardrail says a human must review (`NeedsHumanError`).
2. **Model stalled on a large claim, and a reply was misleading.** On "RM 12,000" the model asked for a
   description and stopped, so no guardrail ran. On a RM3000 motor claim the state was right but the reply said
   "I've created a claim". Fix: `description` is optional; a code backstop escalates when a run ends with no
   claim and no escalation and the message has a digit amount above the limit; the escalation result now says
   "NO claim was created"; and the eval now fails replies that claim a claim exists when none does.

A third issue was in my own test: the new reply check flagged "No claim was created." because it ignored the
negation. I fixed the check and added tests in both directions.

## 5. Measurable outcome
Live behaviour evals (final-state checks plus a reply check) on `ollama:qwen2.5:7b`, local, 2026-11-22:

| Step | Cases | Result |
|---|---|---|
| First run | 8 | 7/8 (failed `over_auto_limit`: no escalation) |
| After fix 1 | 8 | 8/8 on each of 3 runs |
| Added 6 unseen-phrasing cases | 14 | 13/14 (failed `over_limit_spaced_comma`: model stalled) |
| After fix 2 and a reply-honesty check | 14 | 14/14 (one run; an earlier run showed 3 false alarms caused by a bug in my own check, fixed) |

Offline: 36 tests pass, including a deliberately misbehaving scripted model.

Limits of this evidence: the eval set is small (14 cases) and each fix targeted a case found by these same
evals. The model runs at temperature 0, so repeated runs give near-identical replies and add little confidence;
the last result is a single run. The backstop only catches amounts written as digits. The reply check is a text
pattern. I have not tested other models, longer conversations or real data.

## 6. What remains before production
- Real policy/claims APIs, authentication, and per-user permissions
- Larger eval set (hundreds of real, anonymised conversations) and regression runs on every change
- Dashboards and alerts built from the traces (escalation rate, guardrail hits, latency, cost)
- Human review UI, rate limiting, cost caps and a security review (prompt injection, data retention)
- Amounts written in words are not caught by the code backstop; extract amounts with structured output
- Escalation can create duplicate tickets (code and model both escalate); de-duplicate per case
- Replies currently tell the customer the internal auto-limit; a real system may not reveal thresholds
- Multi-turn conversation (the agent currently handles one message and replies once)

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q                 # offline, no API key needed
export ANTHROPIC_API_KEY=...        # then:
python run.py "Claim RM800 for a clinic visit on policy P1001, medical"
python -m evals.run_evals           # live behaviour evals (small API cost)
```
Optional: set `CLAIM_AGENT_MODEL` to choose the Anthropic model.

**No API key? Use a local model (free):**
```bash
ollama pull qwen2.5:7b
export LLM_PROVIDER=ollama
python run.py "Claim RM800 for a clinic visit on policy P1001, medical"
python -m evals.run_evals           # results saved to evals/last_results.json with the model name
```
Python 3.10+ is needed (use `python3 -m venv .venv`).

**Local model, no API key (Ollama):**
```bash
ollama pull qwen2.5:7b              # one-off download
LLM_PROVIDER=ollama python run.py "Claim RM800 for a clinic visit on policy P1001, medical"
LLM_PROVIDER=ollama python -m evals.run_evals
```
Set `OLLAMA_MODEL` to use another tool-calling model. Always report eval results together with the
model name (printed at the start of each run and saved in `evals/last_results.json`).

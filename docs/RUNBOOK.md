# Operations Runbook

## Daily operation

```bash
sentinel run            # process the signal inbox through the pipeline
sentinel incidents      # what was opened
sentinel approvals      # what needs a human decision
sentinel decide ACT-XXXXXXXX --approve --approver you@company.example
sentinel brief INC-XXXXXXXX
```

Or over HTTP (`sentinel serve`, or `make docker`):

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + active backend |
| `POST /runs` | trigger a pipeline run (optional `?limit=N`) |
| `GET /signals` | signal inbox with statuses |
| `GET /incidents`, `GET /incidents/{id}` | incident list / full record incl. briefing |
| `GET /approvals` | pending human approvals |
| `POST /approvals/{action_id}` | `{"approve": true, "approver": "alice@…"}` |
| `GET /audit?run_id=…` | audit trail |

## Switching to live Claude agents

1. `cp .env.example .env`
2. Set `SENTINEL_LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY=sk-ant-…`
3. Optionally tune `SENTINEL_MODEL` and `SENTINEL_RUN_TOKEN_BUDGET`.
4. `sentinel reset && sentinel run`

Cost controls active in live mode:

- **Run token budget** — the run aborts with `TokenBudgetExceeded` when total
  tokens cross `SENTINEL_RUN_TOKEN_BUDGET`.
- **Iteration cap** — each agent's loop stops at `max_agent_iterations`.
- **Usage telemetry** — every run summary includes per-agent token usage and
  an estimated USD cost.

A full 12-signal run is roughly bounded by the configured 400k-token budget
(≈ $4–6 at Opus pricing as a worst case; typical runs are far smaller). Start
with `sentinel run --limit 1` to calibrate.

## State management

- The SQLite store lives at `.sentinel/sentinel.db`; it is disposable.
- `sentinel reset` rebuilds it from the committed dataset.
- `python scripts/generate_data.py` regenerates the dataset deterministically
  (CI verifies `git diff --exit-code data/`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `anthropic.AuthenticationError` | backend=anthropic without a valid `ANTHROPIC_API_KEY` |
| `TokenBudgetExceeded` | raise `SENTINEL_RUN_TOKEN_BUDGET` or run with `--limit` |
| `action … is 'approved', not pending_approval` | double-decision; approvals are single-shot by design |
| Empty approval queue after a run | all actions auto-approved (below limit, compliance pass) or blocked |
| JSON parse error from a live agent | the model returned prose; `extract_json` tolerates fences/preambles, but check the agent's system prompt still ends with the JSON-only contract |

## Incident data model

```
signals ──(triage)──▶ incidents ──▶ actions (blocked | pending_approval | auto_approved | approved | rejected)
                          │
                          └──▶ briefing_md, impact/mitigation/compliance JSON, audit_log
```

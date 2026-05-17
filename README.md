# Origin AI Engineering Take-Home: Referral Inbox Triage Agent

A triage agent for Cedar Kids Therapy's Monday inbox. Reads mixed-channel items (fax referrals, voicemails, portal messages, emails) and produces a structured, human-reviewable action plan per item using a constrained set of tools: `search_patient`, `verify_insurance`, `lookup_policy`, `find_slots`, `hold_slot`, `create_task`, `draft_message`, `escalate`.

## How to run

```bash
npm install
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

The commands also work with no flags and default to the paths above. Reviewers may run the same commands against similar hidden synthetic input. Set `ANTHROPIC_API_KEY` in `.env` before running `npm run triage` (the validator does not need it).

## Stack and runtime

- **Language:** TypeScript on Node LTS, run via `tsx`
- **LLM provider:** Anthropic (`@anthropic-ai/sdk`)
  - Enrichment: `claude-haiku-4-5-20251001` (cheap, fast, structured extraction; system prompt cached)
  - Orchestrator: `claude-sonnet-4-5` (tool-using reasoning loop)
- **Validation:** `ajv` + `ajv-formats` against `schema/output.schema.json`
- **Config:** `.env` via `dotenv`; `ANTHROPIC_API_KEY` is the only required env var
- **Wall-clock runtime:** ~60–90 seconds for the 8-item batch (enrichment runs in parallel; orchestrator is sequential per item)

## Architecture

Two-pass design.

**Pass 1 — Enrichment (`src/enrich.ts`).** A single Haiku call produces a structured *case brief* per item: cleaned intake fields, identifier-strength flags (`patient_search`, `insurance`, `caller_authorization`), safeguarding cues with the exact triggering phrase, same-day operational signals, clinical-advice-request flags, language, existing-patient cues, discrepancy watchpoints, and compliance notes. The prompt examples are principle-based, not lifted from the visible 8 items, so the brief generalizes to unseen variants.

**Pass 2 — Orchestrator (`src/agent.ts`).** A Sonnet tool-loop reads the brief alongside the raw item and decides which of the 8 tools to call. The brief lets the orchestrator skip redundant tool calls and know when a field needs cross-verification — e.g., a payer string from the referral document is re-checked via `verify_insurance` because the billing system is authoritative. Final response is parsed into one `ItemOutput`; `tools_called[]` is sourced from `getToolCallsForItem(item.id)` so the trace is the source of truth, not whatever the LLM reports.

**Draft strategy.** We started with template-based drafts but could only confidently define templates for the 6 message shapes implied by the visible 8 items. The agent currently uses LLM-generated drafts. With domain data, the prod path is a hybrid: the LLM fills slots inside fixed templates rather than writing freely — eliminates clinical-advice and hallucination risk while preserving personalization.

**Self-healing JSON parsing.** LLM JSON outputs sometimes drift from the expected schema (e.g., DOB returned as "3rd January 2024" instead of `YYYY-MM-DD`, or an out-of-enum classification). A `zod` schema validates every parsed response per-field; on failure, each `ZodIssue`'s `path` + message (up to 8) is appended to the conversation as a user-turn correction prompt asking the LLM to re-emit a corrected JSON object. Bounded retries: 3 attempts in enrichment, shared `MAX_ITERS` budget in the orchestrator. If validation never succeeds, enrichment falls back to a rule-based brief and the orchestrator emits a graceful manual-review `ItemOutput`, so the batch never crashes on one bad item. The retry currently asks for whole-object re-emission; a true field-level patch loop (validate, request just the broken keys, merge, re-validate) is in "another 4 hours".

**Safety posture.** Safeguarding signals in the brief set an explicit override flag the orchestrator must honor (P0, escalate, neutral draft only); the override does not depend on the orchestrator's reasoning to fire. `tools_called[]` is reconstructed from the trace, not from any LLM self-report, so output ↔ audit drift is structurally impossible.

## Failure modes and production eval

Per-item enrichment is expensive at scale, and a fully synchronous request/response shape doesn't fit when reasoning loops can take ~1 minute on hard items. A production deployment would look like:

- **Data enrichment pipeline** — aggregate / cluster raw inbox items by temporal proximity, sender, or email-chain so features are generated *per-cluster* rather than per-message, then passed as shared context to the reasoning loop
- **Communication-bus client/server split** — modeled on Claude Code's local-tool pattern: the backend dispatches tool *names + args*, the frontend (sitting on the practice's data) executes them locally so PHI never leaves the practice's domain; only decisions and results cross the wire
- **Tool-call caching** — global / context-style tool calls (policy lookups, payer-network checks, web searches) are content-addressable and cached with TTL; per-patient calls are not
- **Queue-based execution** — Inbound queue → enrichment workers → orchestrator workers → action queue, with DLQs at each hop. Async by default; staff UI subscribes to the action queue rather than polling
- **Evaluation harness** — golden set of ~200 items with expected `(classification, urgency, must-include-tools, must-not-include-tools)` plus an LLM-as-judge for draft quality (empathy, clinical-advice leakage, "we have scheduled" language). CI blocks release on regression
- **Per-draft safety classifier** — a small last-line model that flags any clinical advice or sent-message language before staff sees the draft

## Unit economics and scaling

Current design makes **two LLM calls per item**: one Haiku enrichment + one Sonnet orchestrator session that runs 3–6 turns of tool-use. Per-item cost at observed token usage and Anthropic public pricing (~$1/$5 per M Haiku in/out, ~$3/$15 per M Sonnet in/out):

| Stage | Model | Input tokens | Output tokens | Cost / item |
|---|---|---|---|---|
| Enrichment | Haiku 4.5 | ~3,500 (system prompt cached after first call) | ~800 | ~$0.005 |
| Orchestrator (~5 turns avg) | Sonnet 4.5 | ~25,000 cumulative | ~1,500 | ~$0.10 |
| **Total** | | | | **~$0.10 / item** |

Naive scaling — every item processed the same way:

| Volume | Cost / day | Cost / month |
|---|---|---|
| 100 / day (single practice) | $10 | $300 |
| 1,000 / day (multi-practice) | $100 | $3,000 |
| 10,000 / day (regional SaaS) | $1,000 | $30,000 |
| 100,000 / day (enterprise) | $10,000 | $300,000 |

At 100,000 items/day per-item LLM cost becomes the dominant operating expense and the naive design has to change. The optimizations that change the shape of the curve:

- **Cluster before enrichment** — items sharing a sender, thread, or temporal window get a single shared brief; per-item orchestration only fires when individual action diverges. Realistic 3–10× reduction depending on inbox shape.
- **Tiered routing** — when the brief's confidence is high and classification is unambiguous (obvious spam, clean in-network referral), skip Sonnet entirely and act from a rule table. Realistic 30–50% reduction.
- **Aggressive prompt caching** — already on for enrichment, not yet on for the orchestrator. Realistic 30–50% input-cost reduction on the orchestrator side.
- **Idempotency cache** — hash item body → cached result with TTL. Re-runs of identical items are free; trivial storage cost.

Combined, these get per-item cost into the **$0.01–$0.03 range even at 100K/day**, putting unit economics in the same shape as the staff time displaced (typically $20–50 of staff cost per triaged item before any agent assistance).

## What I chose not to build, and why

These are the known gaps in the current submission. Each was an explicit scope choice given the 2-hour budget:

- **Split identity fields** (parent name / phone / email as separate) — collapsed into one string for now
- **Multi-intent classification** — `tentative_classification` is singular; secondary intent only survives in prose notes
- **Calibration warnings as a structured field** — currently only in prose `urgency_reasoning`
- **Structured reconciliations** — `discrepancy_watch` is free-text, not wired to behavior
- **Referring source as a typed field** — only in prose
- **Coincident-surname hallucination** — agent infers a child's surname from the parent's; false-positive match possible
- **Age out of range** — no proactive age gate; would happily schedule a 22-year-old
- **Split DOB / age** — still one combined string
- **Alternative classifications + calibrated confidence** — asked for in the prompt, not enforced in the schema
- **Languages beyond EN/ES** — silently defaults to `"en"`; a non-English/Spanish family gets an English reply

## What I would do with another 4 hours

- **Better self-healing pattern** — per-key JSON correction loop, currently only at the object level
- **Rule-based pre/post-processing layer** — the agent currently makes tool decisions end-to-end; adding a deterministic policy layer for hard rules (age gate, safeguarding override, OON-blocks-scheduling) catches what the LLM occasionally misses
- **Better data enrichment** — would come out of trial + experimentation and gaining more domain knowledge
- **Better auditing / observability** — more visibility into *why* a decision was made; statistical features generated pre-ingestion would help diagnose a bad prediction
- **Edge-case calibration** — tune the prompt, add few-shot examples for hidden classes (spam, complaint, provider_followup), generate adversarial synthetic items, and verify the agent handles them sanely

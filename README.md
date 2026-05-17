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

**Self-healing JSON parsing.** LLM JSON outputs sometimes drift from the expected schema (e.g., DOB returned as "3rd January 2024" instead of `YYYY-MM-DD`). Per-field validation raises a typed `ValueError` carrying the offending field, expected format, and actual value; that message is then sent back to the LLM in a follow-up turn so it can self-correct only the broken field, not regenerate the whole object.

**Safety posture.** Safeguarding signals in the brief set an explicit override flag the orchestrator must honor (P0, escalate, neutral draft only); the override does not depend on the orchestrator's reasoning to fire. `tools_called[]` is reconstructed from the trace, not from any LLM self-report, so output ↔ audit drift is structurally impossible.

## Failure modes and production eval

Per-item enrichment is expensive at scale, and a fully synchronous request/response shape doesn't fit when reasoning loops can take ~1 minute on hard items. A production deployment would look like:

- **Data enrichment pipeline** — aggregate / cluster raw inbox items by temporal proximity, sender, or email-chain so features are generated *per-cluster* rather than per-message, then passed as shared context to the reasoning loop
- **Communication-bus client/server split** — modeled on Claude Code's local-tool pattern: the backend dispatches tool *names + args*, the frontend (sitting on the practice's data) executes them locally so PHI never leaves the practice's domain; only decisions and results cross the wire
- **Tool-call caching** — global / context-style tool calls (policy lookups, payer-network checks, web searches) are content-addressable and cached with TTL; per-patient calls are not
- **Queue-based execution** — Inbound queue → enrichment workers → orchestrator workers → action queue, with DLQs at each hop. Async by default; staff UI subscribes to the action queue rather than polling
- **Evaluation harness** — golden set of ~200 items with expected `(classification, urgency, must-include-tools, must-not-include-tools)` plus an LLM-as-judge for draft quality (empathy, clinical-advice leakage, "we have scheduled" language). CI blocks release on regression
- **Per-draft safety classifier** — a small last-line model that flags any clinical advice or sent-message language before staff sees the draft

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

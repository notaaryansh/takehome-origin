# Writeup

## Architecture

The agent uses a **two-pass design**: an enrichment LLM call produces a structured *case brief* (cleaned intake, identifier-strength flags, safeguarding cues, discrepancy watchpoints, compliance notes), then a second LLM tool-loop reads the brief alongside the raw item and decides which tools to call. The brief lets the reasoning agent skip redundant tool calls and know when a field needs cross-verification; e.g., a payer string in the referral document gets rechecked via `verify_insurance` because the billing system is authoritative. Examples in the enrichment prompt were kept principle-based rather than lifted from the visible 8 items, so the agent generalizes to unseen variants.

## Draft strategy

We started with template-based drafts but could only confidently define templates for the 6 message shapes implied by the visible 8 items. The agent now uses LLM-generated drafts. With domain data, the prod path is a hybrid: the LLM fills slots inside fixed templates rather than writing freely, eliminates clinical-advice and hallucination risk while preserving personalization.

## Self-healing JSON parsing

LLM JSON outputs sometimes drift from the expected schema (e.g., DOB returned as "3rd January 2024" instead of `YYYY-MM-DD`). Per-field validation raises a typed `ValueError` carrying the offending field, expected format, and actual value; that message is then sent back to the LLM in a follow-up turn so it can self-correct only the broken field — not regenerate the whole object.

## Production thinking

Per-item enrichment is expensive at scale, and a fully synchronous request/response shape doesn't fit when reasoning loops take ~1 minute on hard items. The prod deployment would look like:

- **Data enrichment pipeline** — aggregate / cluster raw inbox items by temporal proximity, sender, or email-chain so features are generated *per-cluster* rather than per-message, then passed as shared context to the reasoning loop.
- **Communication-bus client/server split** — modeled on Claude Code's local-tool pattern: the backend dispatches tool *names + args*, the frontend (sitting on the practice's data) executes them locally so PHI never leaves the practice's domain; only decisions and results cross the wire.
- **Tool-call caching** — global / context-style tool calls (policy lookups, payer-network checks, web searches) are content-addressable and cached with TTL; per-patient calls are not.
- **Queue-based execution** — Inbound queue → enrichment workers → orchestrator workers → action queue, with DLQs at each hop. Async by default; the staff UI subscribes to the action queue rather than polling.

## What was left/ limitations

- Split identity fields (parent.name / phone / email as separate) —> collapsed into one string                              
- Multi-intent classification -> tentative_classification is singular; multi-intent only survives in prose notes
- Calibration warnings as structured field -> currently only in prose urgency_reasoning                                     
- Structured reconciliations -> discrepancy_watch is free-text, not wired to behavior    
- Multi-intent — secondary intent dropped from structured output                                   
- Referring source as typed field —> only in prose  
- Coincident-surname hallucination — agent infers child's surname from parent's; false-positive match                    
- Age out of range — no proactive age gate; would schedule a 22yo                                                          
- Split dob / age —> still one combined string                                                                              
- Alternative classifications + calibrated confidence —> asked for, not enforced
- Languages beyond EN/ES — silently defaults to "en"; non-English family gets English reply 

## Future Work (if more time)

- Better self healing pattern on a per key level for json. 
- More rule based structure, right now the agent calls tools & makes decision. Adding a rule based post processing/ pre processing component.
- Better data enrichment; would come out of trial & experiment and also gaining more domain knowledge. 
- Better auditing; need more observability into "why". Statistical features generated before ingestion will help in understanding a bad prediction.
- Calibrating for edge cases by tuning prompt, adding weights, etc.

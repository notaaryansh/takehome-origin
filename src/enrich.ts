import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import type {
  Classification,
  ExtractedIntake,
  InboxItem,
  Urgency,
} from "./types.js";

const ClassificationSchema = z.enum([
  "new_referral",
  "existing_patient_request",
  "scheduling",
  "clinical_question",
  "billing_question",
  "missing_paperwork",
  "provider_followup",
  "complaint",
  "safeguarding",
  "spam",
  "other",
]);

const UrgencySchema = z.enum(["P0", "P1", "P2", "P3"]);
const DisciplineSchema = z.enum(["SLP", "OT", "PT"]);

const EnrichedItemSchema = z.object({
  extracted_intake: z.object({
    child_name: z.string().nullable(),
    dob_or_age: z.string().nullable(),
    parent_contact: z.string().nullable(),
    discipline: z.array(DisciplineSchema).min(1).nullable(),
    diagnosis_or_concern: z.string().nullable(),
    payer: z.string().nullable(),
    member_id: z.string().nullable(),
  }),
  missing_info: z.array(z.string()),
  signals: z.object({
    safeguarding: z.object({
      present: z.boolean(),
      phrases: z.array(z.string()),
    }),
    same_day_operational: z.object({
      present: z.boolean(),
      time_reference: z.string().nullable(),
    }),
    clinical_advice_request: z.object({
      present: z.boolean(),
      quote: z.string().nullable(),
    }),
    is_incomplete_referral: z.boolean(),
    language: z.enum(["en", "es"]),
    existing_patient_cues: z.array(z.string()),
  }),
  identifier_strength: z.object({
    patient_search: z.enum(["strong", "medium", "weak", "none"]),
    insurance: z.enum(["verifiable", "payer_only", "missing"]),
    caller_authorization: z.enum(["high", "medium", "low"]),
  }),
  urgency_hint: UrgencySchema,
  tentative_classification: ClassificationSchema,
  safeguarding_override: z.boolean(),
  case_brief: z.object({
    summary: z.string(),
    notable_signals: z.array(z.string()),
    suggested_considerations: z.array(z.string()),
    discrepancy_watch: z.array(z.string()),
    urgency_reasoning: z.string(),
    compliance_notes: z.array(z.string()),
  }),
  extraction_confidence: z.enum(["high", "medium", "low"]),
  reasoning: z.string(),
});

export type EnrichedItem = z.infer<typeof EnrichedItemSchema>;

const SYSTEM_PROMPT = `You are the perception layer for an agentic inbox triage system at Cedar Kids Therapy, a pediatric practice supporting speech-language pathology (SLP), occupational therapy (OT), and physical therapy (PT) for ages 0-18.

A downstream agent (Claude, with tool access) will decide which tools to call (search_patient, verify_insurance, lookup_policy, find_slots, hold_slot, create_task, draft_message, escalate) and produce the final classified output. Your job is to give that agent the best possible briefing for THIS specific item, so it makes informed tool choices instead of reasoning from raw text every time.

For each inbox item, emit one JSON object with: cleaned extraction, signal flags, a case_brief that surfaces what is interesting/important/risky about this case specifically.

URGENCY CALIBRATION:
- P0: safeguarding, imminent harm, mandated-reporter triggers. Same-hour review.
- P1: same-day operational issue requiring prompt staff action (same-day cancel/reschedule, urgent billing block).
- P2: normal intake, scheduling, billing, or clinical-review workflow. DEFAULT.
- P3: low-priority admin, FYI, spam.
Over-escalation is a production failure mode. Default to P2 unless there is a clear safety or same-day operational reason. If the sender wrote "URGENT" but the underlying situation is routine, note that in case_brief.urgency_reasoning and DO NOT promote.

SAFEGUARDING: any disclosure of harm, abuse, neglect, unsafe caregiving, or mandated-reporter triggers — physical or emotional harm to the child, exposure to violence, threats, untreated injury, or unsafe living conditions. If present, set safeguarding_override=true, urgency_hint=P0, tentative_classification=safeguarding, and put the EXACT triggering phrase from the message in signals.safeguarding.phrases.

CLINICAL QUESTION: parent or caregiver asks for a clinical opinion or developmental judgment — whether a behavior is typical, whether they should be concerned, whether to wait or intervene, or asks for milestone guidance. The downstream agent must NOT send clinical advice; flag this in compliance_notes.

SAME-DAY OPERATIONAL: an explicit same-day time reference — words or phrases that pin the request to the current day (e.g., references to today, this morning/afternoon/evening, or a specific time-of-day clearly meant for the current day). Set urgency_hint=P1, signals.same_day_operational.present=true, and put the exact time reference from the message in time_reference.

INCOMPLETE REFERRAL: fax or external email with blank required fields (DOB, parent contact, insurance). Set is_incomplete_referral=true.

EXISTING PATIENT CUES: phrases that imply a prior care relationship with this practice — possessive framing of a clinician (e.g., "our clinician", "his/her therapist"), references to ongoing treatment, prior evaluations or assessments, scheduled follow-ups, or any wording that presupposes the family is already on the practice's caseload. List each exact phrase you observe. These tell the downstream agent it is worth calling search_patient even if the text otherwise frames the item as a new referral.

IDENTIFIER STRENGTH (data-quality assessment):
- patient_search: "strong" if full name + ISO DOB (YYYY-MM-DD); "medium" if full name + age; "weak" if only first name or only age; "none" if neither
- insurance: "verifiable" if payer + member_id; "payer_only" if payer only; "missing" otherwise
- caller_authorization: "high" if sender explicitly identifies as the child's parent/guardian; "medium" if implied; "low" if sender role is unclear or sender name differs from likely guardian

CLINICAL_CONCERN vs CONTEXTUAL_REASON in extracted_intake.diagnosis_or_concern:
- If the message is a CANCELLATION/RESCHEDULE, leave diagnosis_or_concern null (the cancellation cause is contextual, not the clinical referral reason).
- For referrals, use the clinical referral reason, not the cancellation cause.

DISCIPLINE inference: only emit "SLP", "OT", or "PT". Infer from the clinical concern using these domain categories (only the category matters; do not lift example terms verbatim into your output):
- SLP: communication, language, articulation, fluency, voice, oral-motor or swallowing function
- OT: fine motor, self-regulation, daily-living and self-care skills, sensory integration, visual-motor coordination
- PT: gross motor, gait, balance, strength, posture, range of motion
If the message references an existing appointment of a specific discipline (e.g., a parent rescheduling an existing OT/SLP/PT visit), use that discipline.
Use null if the concern is too ambiguous to map confidently. The "discipline" field must be either null or a non-empty array.

LANGUAGE: "es" only when the body is substantially Spanish or explicitly requests Spanish.

REFERRING SOURCE: for fax_referral or external email referrals, capture provider name + practice if present.

PREFERENCES: capture scheduling preferences from the text (days, times, language). These will be passed to find_slots downstream.

MISSING_INFO ENUM: only use values from this set, no free text, no hallucinated fields:
["DOB", "parent name", "parent contact", "discipline", "clinical concern", "payer", "member ID", "referring provider"]

CASE_BRIEF: the analytical layer the downstream agent reads to decide tool calls. Keep it sharp and case-specific.
- summary: 1-2 sentence summary of the item.
- notable_signals: bullet observations about what stands out about THIS item that the downstream agent might miss from raw text alone. Hypothetical examples of the form to emit (do not copy these — generate observations specific to the item in front of you): "Sender role differs from likely guardian — verify caller authorization before sharing PHI.", "Only a phone number is provided — email outbound is not available.", "Caller cites a specific clinician by name — possible existing relationship worth confirming."
- suggested_considerations: hints for the agent on what to think about (e.g., "Worth verifying patient identity via search_patient before treating as new referral.", "If insurance returns out-of-network, route to billing before any scheduling step per policy."). Do NOT mandate specific tool calls; these are considerations.
- discrepancy_watch: cross-reference potentials. Hypothetical examples of the form (do not copy these; emit watchpoints specific to this item): "Sender name may not match guardian on file if a patient match returns — verify caller authorization.", "Payer stated in text may not match billing system status — billing system is the authoritative record."
- urgency_reasoning: why the urgency_hint was set (e.g., "P1 because parent explicitly cited today's 3pm appointment; not P0 — illness, not safeguarding."). Surface any de-escalation explicitly.
- compliance_notes: applicable practice policies (e.g., "Clinical-advice policy: do not provide developmental milestone guidance in reply.", "Language-access policy: prefer Spanish-capable provider; draft reply in Spanish.").

OUTPUT JSON SHAPE (emit ONE JSON object, no markdown, no commentary):
{
  "extracted_intake": {
    "child_name": string | null,
    "dob_or_age": string | null,
    "parent_contact": string | null,
    "discipline": ["SLP"|"OT"|"PT", ...] | null,
    "diagnosis_or_concern": string | null,
    "payer": string | null,
    "member_id": string | null
  },
  "missing_info": [string from pinned enum, ...],
  "signals": {
    "safeguarding": { "present": boolean, "phrases": [string, ...] },
    "same_day_operational": { "present": boolean, "time_reference": string | null },
    "clinical_advice_request": { "present": boolean, "quote": string | null },
    "is_incomplete_referral": boolean,
    "language": "en" | "es",
    "existing_patient_cues": [string, ...]
  },
  "identifier_strength": {
    "patient_search": "strong" | "medium" | "weak" | "none",
    "insurance": "verifiable" | "payer_only" | "missing",
    "caller_authorization": "high" | "medium" | "low"
  },
  "urgency_hint": "P0" | "P1" | "P2" | "P3",
  "tentative_classification": one of [new_referral, existing_patient_request, scheduling, clinical_question, billing_question, missing_paperwork, provider_followup, complaint, safeguarding, spam, other],
  "safeguarding_override": boolean,
  "case_brief": {
    "summary": string,
    "notable_signals": [string, ...],
    "suggested_considerations": [string, ...],
    "discrepancy_watch": [string, ...],
    "urgency_reasoning": string,
    "compliance_notes": [string, ...]
  },
  "extraction_confidence": "high" | "medium" | "low",
  "reasoning": string (1-2 sentences)
}`;

const MAX_ENRICH_ATTEMPTS = 3;

const client = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

export async function enrichItem(item: InboxItem): Promise<EnrichedItem> {
  if (!client) {
    return fallbackEnrichment(item);
  }

  const userPrompt = `INBOX ITEM:
id: ${item.id}
channel: ${item.channel}
received_at: ${item.received_at}
sender: ${item.sender}
subject: ${item.subject}
body: ${item.body}

Emit the JSON object now.`;

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userPrompt },
  ];

  for (let attempt = 0; attempt < MAX_ENRICH_ATTEMPTS; attempt++) {
    const isFirstAttempt = attempt === 0;
    const requestMessages: Anthropic.MessageParam[] = isFirstAttempt
      ? [...messages, { role: "assistant", content: "{" }]
      : messages;

    try {
      const response = await client.messages.create({
        model: "claude-haiku-4-5-20251001",
        max_tokens: 2048,
        temperature: 0,
        system: [
          {
            type: "text",
            text: SYSTEM_PROMPT,
            cache_control: { type: "ephemeral" },
          },
        ],
        messages: requestMessages,
      });

      const block = response.content[0];
      if (!block || block.type !== "text") {
        throw new Error("Anthropic response had no text block");
      }
      const raw = isFirstAttempt ? `{${block.text}` : block.text;
      const trimmed = trimToJson(raw);
      const parsedJson = JSON.parse(trimmed) as unknown;

      const validation = EnrichedItemSchema.safeParse(parsedJson);
      if (validation.success) {
        return enforceInvariants(validation.data);
      }

      const errorLines = validation.error.issues
        .slice(0, 8)
        .map((i) => `  - at ${i.path.join(".") || "<root>"}: ${i.message}`)
        .join("\n");
      console.warn(
        `[enrich] ${item.id} attempt ${attempt + 1}/${MAX_ENRICH_ATTEMPTS} failed schema validation:\n${errorLines}`,
      );

      messages.push({ role: "assistant", content: raw });
      messages.push({
        role: "user",
        content: `Your last response failed schema validation. Issues:\n${errorLines}\n\nRe-emit the COMPLETE JSON object, corrected. Output JSON only — no prose, no markdown, no code fences.`,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(
        `[enrich] ${item.id} attempt ${attempt + 1}/${MAX_ENRICH_ATTEMPTS} failed: ${msg}`,
      );

      if (attempt === MAX_ENRICH_ATTEMPTS - 1) break;
      messages.push({
        role: "user",
        content: `Your last response was not valid JSON: ${msg}\n\nRe-emit the COMPLETE JSON object. Output JSON only — no prose, no markdown, no code fences.`,
      });
    }
  }

  console.warn(
    `[enrich] ${item.id} exhausted ${MAX_ENRICH_ATTEMPTS} attempts; falling back to rule-based enrichment.`,
  );
  return fallbackEnrichment(item);
}

function trimToJson(raw: string): string {
  const start = raw.indexOf("{");
  if (start < 0) return raw;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < raw.length; i++) {
    const ch = raw[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return raw.slice(start, i + 1);
    }
  }
  return raw;
}

function enforceInvariants(e: EnrichedItem): EnrichedItem {
  if (e.safeguarding_override) {
    return {
      ...e,
      urgency_hint: "P0" as Urgency,
      tentative_classification: "safeguarding" as Classification,
    };
  }
  return e;
}

function fallbackEnrichment(item: InboxItem): EnrichedItem {
  const text = `${item.subject}\n${item.body}`.toLowerCase();
  const safeguarding = /\b(rough|hit|hurt|abuse|neglect|unsafe|harm|scared)\b/.test(
    text,
  );
  const sameDay =
    /\b(today|same.?day|this afternoon|3 ?pm|2 ?pm|1 ?pm|urgent)\b/.test(text);
  const clinicalQ =
    /\b(should i worry|is it normal|wait until|is this normal|should we wait)\b/.test(
      text,
    );
  const isSpanish = /(hola|gracias|espan|necesita)/.test(text);

  const intake: ExtractedIntake = {
    child_name: null,
    dob_or_age: null,
    parent_contact: null,
    discipline: null,
    diagnosis_or_concern: null,
    payer: null,
    member_id: null,
  };

  return {
    extracted_intake: intake,
    missing_info: [],
    signals: {
      safeguarding: { present: safeguarding, phrases: [] },
      same_day_operational: { present: sameDay, time_reference: null },
      clinical_advice_request: { present: clinicalQ, quote: null },
      is_incomplete_referral: false,
      language: isSpanish ? "es" : "en",
      existing_patient_cues: [],
    },
    identifier_strength: {
      patient_search: "none",
      insurance: "missing",
      caller_authorization: "low",
    },
    urgency_hint: safeguarding ? "P0" : sameDay ? "P1" : "P2",
    tentative_classification: safeguarding
      ? "safeguarding"
      : clinicalQ
        ? "clinical_question"
        : sameDay
          ? "scheduling"
          : "other",
    safeguarding_override: safeguarding,
    case_brief: {
      summary:
        "Rule-based fallback brief; LLM enrichment unavailable or failed validation. Downstream agent should reason from raw text with extra caution.",
      notable_signals: safeguarding
        ? ["regex matched possible safeguarding phrase"]
        : [],
      suggested_considerations: [
        "LLM extraction was unavailable; verify all fields with staff before any auto-action",
      ],
      discrepancy_watch: [],
      urgency_reasoning: safeguarding
        ? "Possible safeguarding phrase matched in fallback regex; treat as P0."
        : "Default P2 unless human review escalates.",
      compliance_notes: [],
    },
    extraction_confidence: "low",
    reasoning:
      "LLM unavailable or validation exhausted; rule-based fallback. Confidence is low; agent should rely on raw item text.",
  };
}

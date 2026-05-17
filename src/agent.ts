import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type { InboxItem, ItemOutput, ToolCall } from "./types.js";
import { enrichItem, type EnrichedItem } from "./enrich.js";

const MODEL = "claude-sonnet-4-5";
const MAX_ITERS = 14;

const client = new Anthropic();

const FinalOutputSchema = z.object({
  classification: z.enum([
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
  ]),
  urgency: z.enum(["P0", "P1", "P2", "P3"]),
  extracted_intake: z.object({
    child_name: z.string().nullable(),
    dob_or_age: z.string().nullable(),
    parent_contact: z.string().nullable(),
    discipline: z.array(z.enum(["SLP", "OT", "PT"])).min(1).nullable(),
    diagnosis_or_concern: z.string().nullable(),
    payer: z.string().nullable(),
    member_id: z.string().nullable(),
  }),
  missing_info: z.array(z.string()).default([]),
  recommended_next_action: z.string().min(1),
  draft_reply: z.string().nullable().default(null),
  task_ids: z.array(z.string()).default([]),
  escalation: z
    .union([
      z.null(),
      z.object({
        reason: z.string().min(1),
        severity: z.enum(["P0", "P1"]),
      }),
    ])
    .default(null),
  decision_rationale: z.string().min(1),
});

type FinalOutput = z.infer<typeof FinalOutputSchema>;

const SYSTEM_PROMPT = `You are a triage agent for Cedar Kids Therapy, a pediatric therapy practice (SLP, OT, PT). You process one inbox item at a time and produce a structured action plan for human staff to review. You never act on a family's behalf — drafts only, holds only, tasks only.

You will receive each item alongside an INTAKE BRIEF from an upstream perception layer that already analyzed the raw text. The brief includes cleaned extraction, identifier-quality flags, suggested considerations, discrepancy watchpoints, and urgency reasoning. Use the brief to focus your tool calls — do NOT re-derive what is already in it, but DO verify with tools when the brief recommends it.

# Priorities (in order)
1. Safety. Any disclosure of harm, abuse, neglect, or unsafe caregiving is P0. Escalate to clinical_lead immediately and draft only a neutral acknowledgement. Never give investigative advice. Do not delay escalation on lookups.
2. Domain judgment. Default urgency is P2. Same-day cancel/reschedule is P1. Do NOT promote to P0 just because the sender wrote "URGENT".
3. Tool use. Use tools to support decisions, not for show. Skip tools whose output won't change the action.

# Insurance gate
Call verify_insurance whenever a payer is present in the brief.
- in_network → proceed with scheduling flow (find_slots is fine)
- out_of_network → call lookup_policy(insurance), create_task(billing), do NOT find_slots or hold_slot
- expired/unknown → create_task(intake or billing) to clarify, do NOT hold a slot
Trust the billing system over the referral document if they conflict — surface the discrepancy.

# Patient identity check
If the brief's identifier_strength.patient_search is "strong" or "medium", call search_patient before deciding the classification. If matched, classification=existing_patient_request (or scheduling for reschedules).
If "weak" or "none", do not waste a search_patient call.

# Intake-completeness gate
If the brief flags is_incomplete_referral or critical fields are missing, classification=missing_paperwork, populate missing_info[], create_task(intake) to call back the referring office. Do NOT hold a slot.

# Clinical-question gate
If the brief flags clinical_advice_request, classification=clinical_question. Draft offers a screening/evaluation pathway. Never give clinical advice, diagnoses, or developmental milestone guidance.

# Language access
If the brief says language="es", call find_slots with language="es" (when scheduling is appropriate) and draft_message with language="es".

# Hard rules
- Never call schedule_appointment or send_message (forbidden, do not exist).
- Never auto-send. Use draft_message only.
- Never schedule. find_slots and hold_slot are reviewable holds.
- Always set requires_human_review=true on every item.
- Avoid performative tool calls. If a result won't influence your action, skip the call.

# Classification enum (pick exactly one)
new_referral, existing_patient_request, scheduling, clinical_question, billing_question, missing_paperwork, provider_followup, complaint, safeguarding, spam, other

# Urgency enum
P0 (safeguarding / imminent harm), P1 (same-day operational), P2 (normal), P3 (low/FYI)

# Output
When done with tool calls, respond with EXACTLY one JSON object, no prose, no markdown, no code fences:

{
  "classification": "<enum>",
  "urgency": "P0|P1|P2|P3",
  "extracted_intake": {
    "child_name": string|null,
    "dob_or_age": string|null,
    "parent_contact": string|null,
    "discipline": ["SLP"|"OT"|"PT", ...] | null,
    "diagnosis_or_concern": string|null,
    "payer": string|null,
    "member_id": string|null
  },
  "missing_info": [string, ...],
  "recommended_next_action": string,
  "draft_reply": string|null,
  "task_ids": [string, ...],
  "escalation": null | {"reason": string, "severity": "P0"|"P1"},
  "decision_rationale": string
}

task_ids must be the task_id values returned by your create_task calls. decision_rationale should reference what the brief surfaced AND what tools showed.`;

const TOOL_SCHEMAS: Anthropic.Tool[] = [
  {
    name: "search_patient",
    description:
      "Look up an existing patient by name and/or date of birth. Returns matches or empty array.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Full or partial patient name" },
        dob: { type: "string", description: "Date of birth YYYY-MM-DD" },
      },
    },
  },
  {
    name: "verify_insurance",
    description:
      "Verify insurance status via billing system. Returns in_network, out_of_network, expired, or unknown.",
    input_schema: {
      type: "object",
      properties: {
        payer: { type: "string", description: "Payer name, e.g., Aetna PPO" },
        member_id: { type: "string", description: "Member ID from referral" },
      },
    },
  },
  {
    name: "lookup_policy",
    description:
      "Look up Cedar Kids Therapy policy snippets on a given topic to ground your decision.",
    input_schema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          enum: [
            "service_lines",
            "insurance",
            "safeguarding",
            "clinical_advice",
            "scheduling",
            "cancellation",
            "language_access",
          ],
        },
      },
      required: ["topic"],
    },
  },
  {
    name: "find_slots",
    description:
      "Find available evaluation slots filtered by discipline, language, and preferences. Caseload-aware.",
    input_schema: {
      type: "object",
      properties: {
        discipline: { type: "string", enum: ["SLP", "OT", "PT"] },
        preferences: {
          type: "string",
          description:
            "Free-text family preferences, e.g., 'mornings' or 'after school Tue/Thu'",
        },
        language: {
          type: "string",
          description: "BCP-47 short code, e.g., 'en' or 'es'",
        },
      },
    },
  },
  {
    name: "hold_slot",
    description:
      "Place a pending_review hold on a slot. NOT a confirmed booking; staff reviews before scheduling.",
    input_schema: {
      type: "object",
      properties: {
        slot_id: { type: "string" },
        patient_ref: {
          type: "string",
          description:
            "Patient ID if known, otherwise a descriptive ref (e.g., child name + DOB)",
        },
      },
      required: ["slot_id", "patient_ref"],
    },
  },
  {
    name: "create_task",
    description:
      "Create a task for a human team member. Use the returned task_id in your final output's task_ids[].",
    input_schema: {
      type: "object",
      properties: {
        assignee: {
          type: "string",
          enum: ["front_desk", "intake", "billing", "clinical_lead"],
        },
        title: { type: "string" },
        due: {
          type: "string",
          description: "Date or ISO datetime when this task is due",
        },
        notes: { type: "string" },
      },
      required: ["assignee", "title", "due", "notes"],
    },
  },
  {
    name: "draft_message",
    description:
      "Draft a reply for human review. Does NOT send. Use Spanish when the family wrote in Spanish.",
    input_schema: {
      type: "object",
      properties: {
        recipient: { type: "string" },
        channel: { type: "string", enum: ["portal", "email", "phone"] },
        body: { type: "string" },
        language: { type: "string", enum: ["en", "es"] },
      },
      required: ["recipient", "channel", "body"],
    },
  },
  {
    name: "escalate",
    description:
      "Escalate this item for same-hour (P0) or same-day (P1) human review. Reserve P0 for safeguarding/imminent-harm cases.",
    input_schema: {
      type: "object",
      properties: {
        item_id: { type: "string" },
        reason: { type: "string" },
        severity: { type: "string", enum: ["P0", "P1"] },
      },
      required: ["item_id", "reason", "severity"],
    },
  },
];

type ToolName =
  | "search_patient"
  | "verify_insurance"
  | "lookup_policy"
  | "find_slots"
  | "hold_slot"
  | "create_task"
  | "draft_message"
  | "escalate";

async function executeTool(
  name: string,
  input: Record<string, unknown>,
): Promise<{ summary: string; data: unknown }> {
  const n = name as ToolName;
  switch (n) {
    case "search_patient": {
      const r = await search_patient(input as { name?: string; dob?: string });
      return { summary: r.result_summary, data: r.data };
    }
    case "verify_insurance": {
      const r = await verify_insurance(
        input as { payer?: string; member_id?: string },
      );
      return { summary: r.result_summary, data: r.data };
    }
    case "lookup_policy": {
      const r = await lookup_policy(input as Parameters<typeof lookup_policy>[0]);
      return { summary: r.result_summary, data: r.data };
    }
    case "find_slots": {
      const r = await find_slots(input as Parameters<typeof find_slots>[0]);
      return { summary: r.result_summary, data: r.data };
    }
    case "hold_slot": {
      const r = await hold_slot(
        input as { slot_id: string; patient_ref: string },
      );
      return { summary: r.result_summary, data: r.data };
    }
    case "create_task": {
      const r = await create_task(input as Parameters<typeof create_task>[0]);
      return { summary: r.result_summary, data: r.data };
    }
    case "draft_message": {
      const r = await draft_message(
        input as Parameters<typeof draft_message>[0],
      );
      return { summary: r.result_summary, data: r.data };
    }
    case "escalate": {
      const r = await escalate(input as Parameters<typeof escalate>[0]);
      return { summary: r.result_summary, data: r.data };
    }
    default:
      return { summary: `unknown tool ${name}`, data: null };
  }
}

function parseAndValidateFinal(
  text: string,
):
  | { ok: true; data: FinalOutput }
  | { ok: false; errors: string[]; rawIssue: string } {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = fenced ? fenced[1] : text;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start === -1 || end === -1) {
    return {
      ok: false,
      errors: ["no JSON object found in response"],
      rawIssue: `Response did not contain a JSON object. Saw: ${text.slice(0, 160)}`,
    };
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(candidate.slice(start, end + 1));
  } catch (err) {
    return {
      ok: false,
      errors: [
        `JSON parse error: ${err instanceof Error ? err.message : String(err)}`,
      ],
      rawIssue: `JSON parse error: ${err instanceof Error ? err.message : String(err)}`,
    };
  }

  const validation = FinalOutputSchema.safeParse(parsedJson);
  if (validation.success) {
    return { ok: true, data: validation.data };
  }

  const errors = validation.error.issues
    .slice(0, 8)
    .map((i) => `at ${i.path.join(".") || "<root>"}: ${i.message}`);
  return { ok: false, errors, rawIssue: errors.join("; ") };
}

function formatBrief(enriched: EnrichedItem): string {
  const lines: string[] = [];
  lines.push("=== INTAKE BRIEF (from perception layer) ===");
  lines.push("");
  lines.push(`SUMMARY: ${enriched.case_brief.summary}`);
  lines.push("");
  lines.push("EXTRACTED INTAKE:");
  lines.push(JSON.stringify(enriched.extracted_intake, null, 2));
  if (enriched.missing_info.length > 0) {
    lines.push(`MISSING FIELDS: ${enriched.missing_info.join(", ")}`);
  }
  lines.push("");
  lines.push("SIGNALS:");
  if (enriched.signals.safeguarding.present) {
    lines.push(
      `  - SAFEGUARDING: triggered by phrase(s): ${enriched.signals.safeguarding.phrases.join("; ")}`,
    );
  }
  if (enriched.signals.same_day_operational.present) {
    lines.push(
      `  - SAME-DAY OPERATIONAL: ${enriched.signals.same_day_operational.time_reference || "(no explicit time)"}`,
    );
  }
  if (enriched.signals.clinical_advice_request.present) {
    lines.push(
      `  - CLINICAL ADVICE REQUEST: "${enriched.signals.clinical_advice_request.quote || ""}"`,
    );
  }
  if (enriched.signals.is_incomplete_referral) {
    lines.push("  - INCOMPLETE REFERRAL: required fields missing");
  }
  lines.push(`  - LANGUAGE: ${enriched.signals.language}`);
  if (enriched.signals.existing_patient_cues.length > 0) {
    lines.push(
      `  - EXISTING-PATIENT CUES: ${enriched.signals.existing_patient_cues.map((c) => `"${c}"`).join(", ")}`,
    );
  }
  lines.push("");
  lines.push("IDENTIFIER QUALITY:");
  lines.push(`  - patient_search: ${enriched.identifier_strength.patient_search}`);
  lines.push(`  - insurance: ${enriched.identifier_strength.insurance}`);
  lines.push(
    `  - caller_authorization: ${enriched.identifier_strength.caller_authorization}`,
  );
  lines.push("");
  lines.push(`TENTATIVE CLASSIFICATION: ${enriched.tentative_classification}`);
  lines.push(`URGENCY HINT: ${enriched.urgency_hint}`);
  lines.push(`URGENCY REASONING: ${enriched.case_brief.urgency_reasoning}`);
  if (enriched.safeguarding_override) {
    lines.push("⚠ SAFEGUARDING OVERRIDE: escalate P0 immediately; do not delay on lookups.");
  }
  lines.push("");
  if (enriched.case_brief.notable_signals.length > 0) {
    lines.push("NOTABLE SIGNALS:");
    for (const s of enriched.case_brief.notable_signals) lines.push(`  - ${s}`);
    lines.push("");
  }
  if (enriched.case_brief.suggested_considerations.length > 0) {
    lines.push("CONSIDERATIONS (not mandates):");
    for (const s of enriched.case_brief.suggested_considerations)
      lines.push(`  - ${s}`);
    lines.push("");
  }
  if (enriched.case_brief.discrepancy_watch.length > 0) {
    lines.push("DISCREPANCY WATCH:");
    for (const s of enriched.case_brief.discrepancy_watch) lines.push(`  - ${s}`);
    lines.push("");
  }
  if (enriched.case_brief.compliance_notes.length > 0) {
    lines.push("COMPLIANCE NOTES:");
    for (const s of enriched.case_brief.compliance_notes) lines.push(`  - ${s}`);
    lines.push("");
  }
  lines.push(`EXTRACTION CONFIDENCE: ${enriched.extraction_confidence}`);
  return lines.join("\n");
}

async function processItem(
  item: InboxItem,
  enriched: EnrichedItem,
): Promise<ItemOutput> {
  const brief = formatBrief(enriched);
  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: `Triage this inbox item using the brief below. Make any tool calls you need, then respond with the final JSON object only.

RAW ITEM:
${JSON.stringify(item, null, 2)}

${brief}`,
    },
  ];

  for (let i = 0; i < MAX_ITERS; i += 1) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
      temperature: 0,
      system: SYSTEM_PROMPT,
      tools: TOOL_SCHEMAS,
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type !== "tool_use") continue;
        const { summary, data } = await executeTool(
          block.name,
          (block.input as Record<string, unknown>) ?? {},
        );
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: JSON.stringify({ summary, data }),
        });
      }
      messages.push({ role: "user", content: toolResults });
      continue;
    }

    const finalText = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === "text")
      .map((b) => b.text)
      .join("\n")
      .trim();

    const validation = parseAndValidateFinal(finalText);
    if (validation.ok) {
      return buildItemOutput(item, validation.data);
    }

    console.warn(
      `[agent] ${item.id} iter ${i + 1}/${MAX_ITERS}: final JSON failed validation. Issues: ${validation.errors.join("; ")}`,
    );
    messages.push({
      role: "user",
      content: `Your final JSON failed schema validation:\n${validation.errors.map((e) => `  - ${e}`).join("\n")}\n\nRe-emit the COMPLETE final JSON object, corrected. Do not call any more tools. Output JSON only — no prose, no markdown, no code fences.`,
    });
  }

  throw new Error(
    `Item ${item.id} did not produce a valid final response within ${MAX_ITERS} iterations`,
  );
}

function buildItemOutput(item: InboxItem, parsed: FinalOutput): ItemOutput {
  const toolCalls = getToolCallsForItem(item.id);
  return {
    item_id: item.id,
    classification: parsed.classification,
    urgency: parsed.urgency,
    requires_human_review: true,
    extracted_intake: parsed.extracted_intake,
    missing_info: parsed.missing_info,
    tools_called: toolCalls,
    recommended_next_action: parsed.recommended_next_action,
    draft_reply: pickDraftBody(toolCalls),
    task_ids: pickTaskIds(toolCalls),
    escalation: parsed.escalation,
    decision_rationale: parsed.decision_rationale,
  };
}

function pickDraftBody(toolCalls: ToolCall[]): string | null {
  for (const c of toolCalls) {
    if (c.name === "draft_message" && typeof c.args?.body === "string") {
      return c.args.body as string;
    }
  }
  return null;
}

function pickTaskIds(toolCalls: ToolCall[]): string[] {
  const ids: string[] = [];
  for (const c of toolCalls) {
    if (c.name !== "create_task") continue;
    const match = c.result_summary.match(/task_[a-z0-9]+/i);
    if (match) ids.push(match[0]);
  }
  return ids;
}

function buildFailureOutput(
  item: InboxItem,
  enriched: EnrichedItem | null,
  err: unknown,
): ItemOutput {
  const message = err instanceof Error ? err.message : String(err);
  const toolCalls = getToolCallsForItem(item.id);
  const intake = enriched?.extracted_intake ?? {
    child_name: null,
    dob_or_age: null,
    parent_contact: null,
    discipline: null,
    diagnosis_or_concern: null,
    payer: null,
    member_id: null,
  };
  const missing = enriched?.missing_info ?? [];
  return {
    item_id: item.id,
    classification: "other",
    urgency: enriched?.urgency_hint ?? "P2",
    requires_human_review: true,
    extracted_intake: intake,
    missing_info: ["agent_execution_failed", ...missing],
    tools_called: toolCalls,
    recommended_next_action:
      "Manual review required — agent encountered an unrecoverable error during triage. A human triager should classify and act on this item.",
    draft_reply: pickDraftBody(toolCalls),
    task_ids: pickTaskIds(toolCalls),
    escalation: null,
    decision_rationale: `Agent failed to produce a valid output: ${message}. Falling back to manual review. Partial tool trace preserved; enrichment ${enriched ? "succeeded" : "also failed"}.`,
  };
}

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const enrichedSettled = await Promise.allSettled(
    inbox.map((item) => enrichItem(item)),
  );

  const results: ItemOutput[] = [];
  for (let i = 0; i < inbox.length; i++) {
    const item = inbox[i];
    const settled = enrichedSettled[i];
    const enriched = settled.status === "fulfilled" ? settled.value : null;

    if (!enriched) {
      console.warn(
        `[agent] enrichment failed for ${item.id}: ${
          settled.status === "rejected"
            ? settled.reason instanceof Error
              ? settled.reason.message
              : String(settled.reason)
            : "unknown"
        }`,
      );
    }

    try {
      const output = await withItemContext(item.id, async () => {
        if (!enriched) {
          throw new Error("enrichment failed; cannot proceed with full triage");
        }
        return processItem(item, enriched);
      });
      results.push(output);
    } catch (err) {
      console.warn(
        `[agent] ${item.id} failed processing; emitting fallback ItemOutput: ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      const fallback = await withItemContext(item.id, async () =>
        buildFailureOutput(item, enriched, err),
      );
      results.push(fallback);
    }
  }
  return results;
}

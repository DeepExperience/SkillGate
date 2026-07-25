#!/usr/bin/env python3
"""Build the standalone mixed-skill slate for the 147 non-eval70 Claw tasks.

The output is restartable and intentionally lives outside the frozen FINAL
train/eval70 assets.  Each task receives exactly:

    1 oracle + 5 misleading + 5 relevant + 5 irrelevant

Generation writes only the current accepted SKILL.md for each generated name.
Rejected attempts are recorded in logs, not retained as additional skill trees.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "skill_libraries/snapshots/rl/eval_claw_147"
DEFAULT_WORK = ROOT / "experiments/skill_slate_build/eval_claw_147"
CLAW_161 = ROOT / "GeneralAgent/eval_scripts/prebake_images/claw_161_t_series.txt"
EVAL70_TASKS = ROOT / "ops/workflows/rl_eval/specs/eval70_v1/tasks.tsv"
CLAW_TASK_ROOT = ROOT / "datasets/claw-eval/tasks"
CLAW_ROOT = ROOT / "datasets/claw-eval"
LEGACY_ORACLE_ROOT = ROOT / "skill_libraries/snapshots/rl/oracle_skills_full692_20260612/claw"
RETRIEVAL_JSONL = (
    ROOT
    / "experiments/archive_sft_runs/20260424/"
    "20260424_v7pipeline_on_2046lib/retrieval_results/claw.jsonl"
)
MERGED_ROOT = ROOT / "skill_libraries/merged"

N_ORACLE = 1
N_MISLEADING = 5
N_RELEVANT = 5
N_IRRELEVANT = 5
SLATE_SIZE = N_ORACLE + N_MISLEADING + N_RELEVANT + N_IRRELEVANT
DESCRIPTION_SIM_THRESHOLD = 0.92
BUILDER_ID = "eval_claw_147_v8_hybrid_separable_v2"
OUTCOME_AUDIT_ID = "frozen_instance_outcome_falsifier_v5_full_grounding"
CLAUDE_ORACLE_AUDIT_ID = "claude-code-opus-medium-oracle-body-audit-v1"

STRATEGIES = (
    (
        "wrong_commands_or_api",
        "Use the same service and deliverable but commit to a plausible wrong command, "
        "API method, endpoint, payload field, or state transition.",
    ),
    (
        "wrong_path_or_output_contract",
        "Keep the same artifact family but use a verifier-relevant wrong path, output key, "
        "schema field, recipient, object, or final-state contract.",
    ),
    (
        "wrong_sequence_or_validation",
        "Bind the procedure to a wrong prerequisite order or success criterion, and make "
        "validation positively confirm that corrupted criterion.",
    ),
    (
        "wrong_parameter_or_decision_rule",
        "Use a concrete wrong threshold, date, duration, comparator, formula, filter, "
        "status, or selection rule that changes the graded outcome.",
    ),
    (
        "wrong_object_or_scope_rule",
        "Perform the requested action class on a plausible but wrong item/subset/window, "
        "without changing the primary deliverable into an adjacent task.",
    ),
)

FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<front>.*?)\n---\s*\n", re.S)
FENCE_RE = re.compile(r"\A```(?:markdown|md)?\s*|\s*```\s*\Z", re.I)
BAD_META_RE = re.compile(
    r"\b(oracle skill|misleading skill|hard[- ]negative|decoy|adversarial|"
    r"corrupted (?:skill|document)|grader|hidden test|hidden ground truth|reward|"
    r"scoring (?:criteria|components?|rubric)|evaluation (?:criteria|mechanics?))\b",
    re.I,
)
BAD_NAME_RE = re.compile(r"\b(oracle|misleading|negative|decoy|bad|fake|trap)\b", re.I)
HARDCODED_SERVICE_ORIGIN_RE = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|host\.docker\.internal):\d{2,5}", re.I
)
HARDCODED_SERVICE_PORT_RE = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|host\.docker\.internal):\d{2,5}|\b91\d{2}\b",
    re.I,
)
BUILD_TIME_SERVICE_PORT_RE = re.compile(r"\b91\d{2}\b")
UNBOUND_SERVICE_BASE_RE = re.compile(r"\$(?:[A-Z][A-Z0-9_]*_)?BASE_URL\b")
HYPOTHETICAL_EVIDENCE_RE = re.compile(
    r"\b(hypothetical|counterfactual|invented|imaginary|suppose|future input|"
    r"unobserved|plausible|depending|could|might|general case|as implied|assuming|"
    r"likely|may(?!\s+(?:\d{1,2}|20\d{2})\b)|perhaps|probably|implies?|suggests?|typically|commonly|realistic|"
    r"possible (?:scenario|case|ordering|state|input|value)|"
    r"potential(?:ly)? (?:fail|failure|conflict|issue|problem|wrong|difference)|"
    r"if (?:the )?(?:actual|fixture|service|api|response|unknown))\b",
    re.I,
)
SELF_REJECT_EVIDENCE_RE = re.compile(
    r"\b(?:same (?:outcome|result)|outcomes? (?:do|does) not differ|"
    r"does not change (?:the )?(?:outcome|result)|would not change (?:the )?(?:outcome|result)|"
    r"no (?:outcome )?difference|i must reject|must be rejected|technically includes?|"
    r"both (?:rules|procedures|methods|paths).{0,120}(?:yield|produce|select|include).{0,40}same)\b",
    re.I | re.S,
)
UNOBSERVED_EVIDENCE_RE = re.compile(
    r"\b(?:task|provided|authoritative) context (?:does not|doesn't|did not) "
    r"(?:specify|provide|state|include)|\bin the absence of|\bnot (?:specified|provided|observed|known)|"
    r"\bif\b[^.]{0,160}\b(?:were|had|would|could|might)\b|\b(?:assum(?:e|es|ed|ing)|"
    r"implied|often|common)\b",
    re.I | re.S,
)
NEGATED_UNOBSERVED_RE = re.compile(
    r"\b(?:without|no)\b[^.;\n]{0,160}\b"
    r"(?:hypothetical|counterfactual|unobserved|unstated|assumed|invented)\b[^.;\n]{0,160}|"
    r"\b(?:does|do) not (?:rely|depend)\b[^.;\n]{0,160}\b"
    r"(?:hypothetical|counterfactual|unobserved|unstated|assumed|invented)\b[^.;\n]{0,160}|"
    r"\bnot relying\b[^.;\n]{0,160}\b"
    r"(?:hypothetical|counterfactual|unobserved|unstated|assumed|invented)\b[^.;\n]{0,160}",
    re.I,
)
NUMERIC_DECISION_RE = re.compile(
    r"(?:\bthreshold\b|\bpercent(?:age)?\b|\bratio\b|\bcount\b|\bsize\b|"
    r"\bduration\b|\bwindow\b|\bwithin\s+\d|\bmore than\b|\bless than\b|"
    r"\bfewer than\b|\bat (?:most|least)\b|\b(?:above|below|exceed(?:s|ed)?)\b|"
    r"\bstrictly (?:less|greater)\b|\d+(?:\.\d+)?\s*%)",
    re.I,
)

NAME_SYSTEM = """You name six skill documents for one agent task.
Return strict JSON only:
{"oracle_name":"...","misleading_names":["...","...","...","...","..."]}

All names must be natural lowercase kebab-case library names with 2-7 words.
They must sound equally legitimate and useful for the same task family. Do not
reuse the task id, and do not use quality labels such as oracle, misleading,
negative, bad, fake, legacy, alternate, v2, or decoy. The correct and flawed
skills must not be distinguishable from their names alone."""

ORACLE_SYSTEM = """Write one high-quality task-specific SKILL.md for an OpenClaw-style agent.
Return only the complete file, without markdown fences or analysis.

Requirements:
- Use the assigned YAML frontmatter name exactly and write a concise 1-2 sentence description.
- Give the correct operational procedure for the requested task, including concrete tools,
  fields, files, state transitions, checks, and safety constraints when available.
- The task YAML's named tools describe HTTP endpoints; they are NOT direct OpenClaw function
  tools. When HTTP services are needed, instruct the agent to use OpenClaw `exec` with curl or a
  small script against the runtime-provided base URL/port. Never invent a direct function call to
  a task-YAML tool and never hardcode a port. Runtime does not define `$SERVICE_BASE_URL` or similar
  shell variables. Tell the agent to copy the exact current URL from the HTTP Tools documentation,
  and use an angle-bracket placeholder such as `<gmail_url_from_HTTP_Tools>` in examples. Literal
  URLs like `http://localhost:9100/...` are forbidden even if task.yaml shows them.
- Preserve the requested primary deliverable. Make the procedure directly useful if followed.
- Treat task.yaml, grader source, and mock-service source as authoritative. A supplied legacy
  skill is only a fallible draft: retain useful specifics but repair stale ports, nonexistent
  methods/routes, wrong values, fixed dates, missing output contracts, skipped required actions,
  and omitted zeroing safety constraints. Never
  mention evaluation machinery, graders, scores, hidden tests, rewards, or generation. A task
  may itself ask about a document's benchmark terminology; preserve that user-facing content.
- Avoid broad background prose. Prefer a compact objective, 3-8 executable steps, validation,
  and a short Avoid section for consequential failure modes.
- Never hardcode a service port when the runtime tool documentation supplies it dynamically.
"""

ORACLE_JUDGE_SYSTEM = """Judge a proposed correct task skill. Return strict JSON only:
{
  "same_primary_deliverable": true,
  "factually_consistent_with_task": true,
  "operational_and_actionable": true,
  "likely_helpful_if_followed": true,
  "description_correct_and_specific": true,
  "no_meta_contamination": true,
  "summary": "one sentence"
}
Use false for any unsupported command/field, wrong final state, task drift, vague procedure,
or evaluation/grader/reward leakage. In this OpenClaw runtime, task-YAML tool names are HTTP
operation schemas rather than directly callable function tools; `exec` plus curl/script against
the runtime-provided service URL is the correct action pattern. Benchmark terminology is allowed
only when it is part of the requested document or user-facing deliverable."""

ORACLE_DEEP_JUDGE_SYSTEM = """Audit an OpenClaw task skill against authoritative task.yaml,
grader source, fixture evidence, and mock-service source. Return strict JSON only:
{
  "dynamic_endpoint_usage_correct": true,
  "only_available_openclaw_actions_and_existing_routes": true,
  "values_and_final_answer_contract_correct": true,
  "dynamic_dates_and_fixture_behavior_correct": true,
  "scored_delivery_structure_covered": true,
  "required_actions_and_components_preserved": true,
  "zeroing_safety_constraints_covered": true,
  "factually_consistent_and_actionable": true,
  "no_evaluation_leakage": true,
  "summary": "one sentence"
}

All booleans must be true. Task-YAML tool names describe HTTP endpoints and are not callable
OpenClaw functions: service interaction must be expressed through `exec` using the runtime HTTP
documentation and dynamic base URL/port. Reject hardcoded ports, nonexistent methods/routes or
payload fields, wrong expected values, fixed dates where fixtures move dynamically, missing
grader-required final-deliverable structure, advice to skip scored actions, or omitted safety
rules whose violation zeroes the task. The skill must not mention graders, hidden tests, rewards,
or evaluation mechanics. The literal 91xx ports in task.yaml are build-time defaults, never the
runtime endpoint: a candidate that uses symbolic runtime base-URL placeholders must pass
`dynamic_endpoint_usage_correct`, and must never be rejected for refusing to hardcode task.yaml's
ports. Benchmark terminology is allowed only if the user's task asks for it.
"""

ORACLE_REFERENCE_JUDGE_SYSTEM = """Perform a second, narrow consistency audit of a proposed
correct skill against the authoritative task context. Return strict JSON only:
{
  "all_required_items_and_actions_covered": true,
  "no_concrete_rule_conflicts": true,
  "no_wrong_value_classification_or_state": true,
  "required_final_output_contract_preserved": true,
  "candidate_conflict_quote": "empty when all checks pass; otherwise an exact candidate quote",
  "authoritative_conflict_quote": "empty when all checks pass; otherwise an exact context quote",
  "summary": "one sentence"
}

Compare concrete task-specific rules line by line, especially reference_solution, expected_actions,
safety_checks, scoring_components, named fixture items, required values, classifications, state
transitions, forbidden actions, and final-output fields. Reject a candidate that skips a required
item/action, assigns any named item the wrong outcome, substitutes a plausible but different rule,
or weakens a required deliverable. Do not excuse a conflict merely because the candidate sounds
generally useful. Do not reject harmless extra operational detail or an alternative implementation
that preserves every required action and outcome.

The supplied grader/mock-service evidence is authoritative for executable routes, payload fields,
runtime behavior, and fixture values. It overrides stale task-context schemas or prose when they
disagree. Use the task context for intended deliverables and expected outcomes, but never reject a
candidate for following the actual mock-service request model over a stale tool input_schema.

The runtime exposes task-YAML operations as HTTP schemas, not direct OpenClaw function tools.
Using OpenClaw `exec` with the exact dynamic URL from HTTP Tools is the correct adapter and must not
be treated as a conflict. Build-time ports in task.yaml must not be copied literally.

When any boolean is false, copy one exact contiguous quote from the candidate and one exact
contiguous quote from the authoritative task context that demonstrate the conflict. Do not
paraphrase, use ellipses, or invent a rule. If no such authoritative quote exists, do not claim a
conflict.
"""

MISLEADING_SYSTEM = """Write one controlled hard-negative SKILL.md for a skill-selection study.
Return only the complete file, without markdown fences or analysis.

This is the V8/hybrid-separable standard:
- Keep the assigned YAML frontmatter name exactly.
- Keep the SAME primary user-facing deliverable, task object family, and broad action pipeline
  as the correct skill. The candidate must look like a plausible resolver for the exact task.
- The first description sentence should match the exact task nouns, artifact/action class, and
  operational verb. The second sentence must state one concrete but subtle FALSE scoping rule,
  target, field, state, comparator, formula, path, or validation premise.
- That false premise must be logically distinguishable from the correct task/skill by a careful
  selector. It must not be merely vague or lexically different.
- Make the false premise the central executable contract in the FIRST action. Keep all later
  commands, edits, calculations, state mutations, and validation aligned with it.
- Task-YAML tool names describe HTTP endpoints, not direct OpenClaw functions. Preserve the real
  OpenClaw action pipeline: use `exec` and the runtime-provided HTTP base URL/port for services.
  Do not hardcode ports or accidentally make the candidate fail merely by inventing an unavailable
  function tool. Runtime does not define `$SERVICE_BASE_URL`; examples must use angle-bracket URL
  placeholders that the agent replaces from the current HTTP Tools docs. The intended failure must
  come from its wrong task decision.
- A literal competent executor should fail the original task. Do not leave the correct command,
  value, path, formula, output key, object, state, or validation as recoverable scaffolding.
- No fallback, compare-with-request, broad exploration, full repair loop, or second validation
  path that could recover the correct solution.
- Prefer a compact wrong-only fast path: objective, 2-5 actions or one script/patch, and one
  validation of the corrupted state.
- Never label the document as bad, misleading, corrupted, negative, adversarial, or a decoy.

Corruption focus:
{strategy_text}
"""

MISLEADING_DESCRIPTION_REWRITE_SYSTEM = """Rewrite only the YAML description of a controlled
hard-negative skill. Return strict JSON only:
{
  "description": "one or two sentences",
  "concrete_conflict": "the specific false rule stated in the description"
}

The description must make the skill attractive for the exact task and truthfully summarize the
candidate procedure, while stating its central false target, field, state, comparator, formula,
path, scope, or validation rule. It must be logically distinguishable from the correct skill and
must not copy or merely paraphrase the correct description. Do not use quality labels or mention
generation, evaluation, graders, rewards, hidden tests, or deception. Preserve task-specific nouns
and the primary deliverable. Use exactly two concise sentences: independently restate the task in
the first, then state the concrete procedural constraint in the second. Avoid reusing any phrase of
five or more consecutive words from the correct description. Do not change the body of the
candidate skill.
"""

MISLEADING_JUDGE_SYSTEM = """Judge a hard-negative skill against the task and correct skill.
Return strict JSON only with exactly these keys:
{
  "attractive_for_exact_task": true,
  "same_primary_deliverable": true,
  "same_action_pipeline": true,
  "action_framework_wrong": true,
  "likely_failure_if_followed": true,
  "logically_distinguishable": true,
  "description_contains_concrete_conflict": true,
  "description_not_identical": true,
  "not_salvageable_by_correct_scaffold": true,
  "no_correct_rescue_hints": true,
  "no_meta_contamination": true,
  "summary": "one sentence"
}

All booleans must be true to pass. `logically_distinguishable` and
`description_contains_concrete_conflict` require a specific false rule visible in the
description that conflicts with the task/correct skill. Mere paraphrase, omitted detail, or a
quality label does not pass. Wrong endpoints, fields, values, objects, formulas, filters, paths,
state transitions, output contracts, and validation targets count as action-framework errors
only when the broad task and deliverable remain the same. Set
`not_salvageable_by_correct_scaffold=false` if enough exact correct procedure remains for a
competent agent to ignore one bad line and solve the task."""

MISLEADING_CONFLICT_JUDGE_SYSTEM = """Perform a focused logical-conflict audit for a candidate
hard-negative description. Return strict JSON only:
{
  "candidate_rule_quote": "an exact contiguous quote from the candidate description/opening",
  "authoritative_rule_quote": "an exact quote from task context or the audited correct skill",
  "affected_task_item_quote": "an exact authoritative quote naming the affected item/input",
  "candidate_outcome_for_affected_item": "short concrete value, action, class, or state",
  "authoritative_outcome_for_affected_item": "short concrete value, action, class, or state",
  "rules_mutually_incompatible": true,
  "same_task_item_or_decision_point": true,
  "affected_item_outcomes_differ": true,
  "wrong_rule_controls_first_action": true,
  "wrong_rule_changes_required_outcome": true,
  "evidence": "brief task-specific evidence"
}

All four booleans must be true. Judge direct logical incompatibility, not lexical difference or
generic risk. Set `rules_mutually_incompatible=false` when the candidate rule is actually allowed,
already required, merely more specific, or affects no item/state in this task. Set
`same_task_item_or_decision_point=false` unless the two quoted rules govern the same concrete item,
field, decision, state, or required output in this task. The false rule must be visible in the
candidate description, drive the first executable action, and materially change a required output
or state.

`affected_item_outcomes_differ` and `wrong_rule_changes_required_outcome` are about this frozen task
instance, not hypothetical future
inputs. The evidence must name at least one actual fixture item, requested file/field, concrete
calculation, or required action for which following the candidate produces a different final value,
classification, state, or delivered artifact. A different threshold or comparator is not enough
when every supplied value falls on the same side and the required result remains unchanged. Report
the candidate and authoritative outcomes for that same item in compact canonical terms. If those
outcomes are equal or synonymous (for example both say include, faulty, or Needs Reply), both
booleans must be false. Never use a hypothetical or invented item/value as evidence.

The rule quote fields are evidence, not summaries. Copy an exact contiguous clause of at least 16
characters from the named source. The candidate rule may be quoted from its description or opening
procedure; a separate audit enforces that the description itself states the conflict. The affected
item quote may be a shorter exact identifier or field name. The authoritative rule and item quotes
MUST come from the supplied task context or the independently audited correct skill. Prefer task
context; use the correct skill when the context records only a final answer and the executable rule
is stated there. Do not paraphrase, combine distant fragments, use ellipses, or invent an implied
rule. Preserve the source spelling, punctuation, Markdown, and spacing instead of translating or
restyling it. For `affected_task_item_quote`, copy one shortest useful source token or identifier
(for example `INT-004`), never a comma-joined list and never an invented label around an identifier.
Before returning JSON, verify that every quote can be found by literal copy/paste in its named
source. If an exact authoritative conflict cannot be quoted, set the relevant booleans false.
"""

MISLEADING_OUTCOME_JUDGE_SYSTEM = """Act as an adversarial falsifier for one proposed hard-negative
skill. The candidate already passed relevance and description checks; your only job is to prove
that its wrong rule changes a required result on THIS frozen task instance. Default to rejection.
Return strict JSON only with exactly these keys:
{
  "candidate_rule_quote": "exact contiguous quote from the candidate description/opening",
  "authoritative_rule_quote": "exact contiguous quote from task context or audited correct skill",
  "affected_task_item_quote": "short exact quote naming one actual item/action/output",
  "actual_input_quote": "exact source quote containing the actual value/state/requirement used",
  "candidate_outcome_for_affected_item": "concrete final value/action/class/state",
  "authoritative_outcome_for_affected_item": "concrete final value/action/class/state",
  "outcomes_differ_on_frozen_instance": true,
  "no_hypothetical_or_unobserved_dependency": true,
  "description_rule_controls_procedure": true,
  "candidate_failure_caused_by_wrong_rule": true,
  "numeric_check_required": false,
  "numeric_lhs": null,
  "numeric_operator": "",
  "numeric_rhs": null,
  "numeric_predicate_result": null,
  "evidence": "show the concrete frozen-instance derivation"
}

Reject whenever the candidate and authoritative procedure produce the same required outcome, even
if their rationale, threshold, intermediate calculation, or wording differs. Reject evidence that
depends on a hypothetical item, a changed future value, an unobserved field, or phrases such as
"could fail if". Recompute every comparison; for example 10/30 = 0.333 is strictly less than 0.5.

Set `numeric_check_required=true` whenever the wrong decision depends on a numeric threshold,
percentage, ratio, count, size, duration, or arithmetic formula. For the one affected item, put the
actual computed left operand in `numeric_lhs`, the literal operator in `numeric_operator` (one of
<, <=, >, >=, ==, !=), the comparison value in `numeric_rhs`, and the mathematically correct
boolean in `numeric_predicate_result`. Use normalized units (for example 0.0562 and 0.05, or 390
and 240). Do not use these numeric fields for dates or purely categorical rules.

All quote fields must be literal source substrings: do not translate, paraphrase, join fragments,
or invent labels. `actual_input_quote` and `affected_task_item_quote` must come from the task
context or audited correct skill. The candidate rule must be visible in the description and govern
the procedure. Set every boolean false unless the supplied evidence directly establishes it.
"""

MISLEADING_OUTCOME_FALSIFIER_SYSTEM = """Independently audit a proposed proof that one hard-negative
skill changes the required outcome on a frozen task instance. The previous judge may be wrong.
Return strict JSON only:
{
  "proof_uses_only_observed_instance": true,
  "candidate_rule_applied_correctly": true,
  "authoritative_rule_applied_correctly": true,
  "outcomes_really_differ": true,
  "no_same_outcome_or_self_rejection": true,
  "numeric_reasoning_consistent": true,
  "evidence": "brief adversarial verification"
}

Default to false. Reject any proof that relies on an unstated field, unknown API ordering, implied
status, assumed duration/date/value, likely future input, or a hypothetical counterexample. Apply
comparators and filters literally: for example, 12M passes a filter requiring more than 10M, and
a stricter threshold is harmless when every frozen value still passes it. Procedural differences
are insufficient when both procedures produce the same actual result. Reject if the proposed proof
itself says the outcome is unchanged, the candidate still selects the authoritative item, or the
candidate failure is merely possible. `numeric_reasoning_consistent` is true when there is no
numeric rule to check. Set all six booleans true only when the supplied frozen values directly prove
a different required final value, action, class, state, or artifact.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evidence_is_unobserved(text: str) -> bool:
    normalized = NEGATED_UNOBSERVED_RE.sub("", text)
    return bool(
        HYPOTHETICAL_EVIDENCE_RE.search(normalized)
        or UNOBSERVED_EVIDENCE_RE.search(normalized)
    )


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def task_key(task_id: str) -> str:
    return f"claw::{task_id}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_rejection(path: Path, payload: dict[str, Any]) -> None:
    """Keep actionable frozen-instance feedback across intermediate retries."""
    merged = dict(payload)
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {}
        if prior.get("instruction"):
            merged["instruction"] = prior["instruction"]
    atomic_write_json(path, merged)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_claw_ids() -> tuple[list[str], list[str], list[str]]:
    all_ids = [
        line.strip()
        for line in CLAW_161.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    eval_ids = []
    for line in EVAL70_TASKS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        bench, task_id = line.split("\t", 1)
        if bench == "claw":
            eval_ids.append(task_id)
    eval_set = set(eval_ids)
    target_ids = [task_id for task_id in all_ids if task_id not in eval_set]
    if len(all_ids) != 161 or len(set(all_ids)) != 161:
        raise SystemExit(f"expected 161 unique Claw T-series tasks, got {len(all_ids)}")
    if len(eval_ids) != 14 or len(set(eval_ids)) != 14:
        raise SystemExit(f"expected 14 unique eval70 Claw tasks, got {len(eval_ids)}")
    if not set(eval_ids).issubset(all_ids):
        raise SystemExit("eval70 Claw tasks are not a subset of the frozen 161-task list")
    if len(target_ids) != 147:
        raise SystemExit(f"expected 147 target tasks, got {len(target_ids)}")
    return all_ids, eval_ids, target_ids


def task_context(task_id: str) -> dict[str, Any]:
    path = CLAW_TASK_ROOT / task_id / "task.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keep = (
        "task_id",
        "task_name",
        "category",
        "difficulty",
        "prompt",
        "tools",
        "tool_endpoints",
        "sandbox_files",
        "sandbox_grader_files",
        "environment",
        "expected_actions",
        "safety_checks",
        "scoring_components",
        "judge_rubric",
        "reference_solution",
    )
    return {key: raw.get(key) for key in keep if raw.get(key) not in (None, "", [], {})}


def clipped_source(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n\n... [middle clipped by builder] ...\n\n" + text[-tail:]


def resolve_task_payload_path(task_dir: Path, value: Any) -> Path | None:
    raw_path = Path(str(value))
    candidates = [raw_path] if raw_path.is_absolute() else [
        task_dir / raw_path,
        CLAW_ROOT / raw_path,
        CLAW_TASK_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    return None


def text_fixture_files(path: Path) -> list[Path]:
    suffixes = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".tsv", ".md", ".py"}
    if path.is_file():
        return [path] if path.suffix.lower() in suffixes else []
    if path.is_dir():
        return sorted(
            child.resolve()
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in suffixes
        )
    return []


def oracle_grounding_bundle(
    task_id: str, *, include_legacy: bool = True
) -> tuple[str, list[dict[str, str]]]:
    """Collect authoritative task/grader/runtime evidence plus a fallible legacy draft."""
    task_dir = CLAW_TASK_ROOT / task_id
    task_yaml = task_dir / "task.yaml"
    raw = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    candidates: list[tuple[str, Path, int]] = [
        ("AUTHORITATIVE task.yaml", task_yaml, 42000),
        ("AUTHORITATIVE grader.py", task_dir / "grader.py", 42000),
    ]

    # English/Chinese task pairs often inherit the peer grader containing the
    # actual expected values. Follow those references recursively.
    peer_queue = [task_dir / "grader.py"]
    seen_peers = {task_id}
    while peer_queue:
        grader_path = peer_queue.pop(0)
        if not grader_path.is_file():
            continue
        grader_text = grader_path.read_text(encoding="utf-8", errors="replace")
        for peer_id in re.findall(r'load_peer_grader\(["\']([^"\']+)["\']\)', grader_text):
            if peer_id in seen_peers:
                continue
            seen_peers.add(peer_id)
            peer_path = CLAW_TASK_ROOT / peer_id / "grader.py"
            if peer_path.is_file():
                candidates.append((f"AUTHORITATIVE peer grader {peer_id}", peer_path, 42000))
                peer_queue.append(peer_path)

    fixture_paths: list[Path] = []
    for field in ("sandbox_files", "sandbox_grader_files"):
        for relative in raw.get(field) or []:
            fixture_path = resolve_task_payload_path(task_dir, relative)
            if fixture_path is not None:
                fixture_paths.append(fixture_path)
    for service in raw.get("services") or []:
        for value in ((service or {}).get("env") or {}).values():
            fixture_path = resolve_task_payload_path(task_dir, value)
            if fixture_path is not None:
                fixture_paths.append(fixture_path)
    for fixture_root in fixture_paths:
        for fixture_path in text_fixture_files(fixture_root):
            candidates.append(("AUTHORITATIVE task fixture", fixture_path, 32000))

    service_paths = []
    for service in raw.get("services") or []:
        command = str((service or {}).get("command") or "")
        match = re.search(r"(mock_services/[^\s]+\.py)", command)
        if match:
            service_path = CLAW_ROOT / match.group(1)
            if service_path.is_file():
                service_paths.append(service_path)
    for service_path in service_paths:
        candidates.append(("AUTHORITATIVE mock-service source", service_path, 36000))
    base_service = CLAW_ROOT / "mock_services/_base.py"
    if service_paths and base_service.is_file():
        candidates.append(("AUTHORITATIVE mock-service shared base", base_service, 28000))

    legacy_path = LEGACY_ORACLE_ROOT / task_id / "SKILL.md"
    if include_legacy and legacy_path.is_file():
        candidates.append(("FALLIBLE legacy oracle draft", legacy_path, 24000))

    sections = []
    source_records = []
    seen_paths = set()
    remaining = 170000
    for label, path, per_file_limit in candidates:
        path = path.resolve()
        if path in seen_paths or not path.is_file() or remaining < 2000:
            continue
        seen_paths.add(path)
        text = clipped_source(path, min(per_file_limit, remaining))
        remaining -= len(text)
        sections.append(f"===== {label}: {path} =====\n{text}")
        source_records.append({"path": str(path), "sha256": sha256_file(path), "role": label})
    return "\n\n".join(sections), source_records


def seeded_rng(tag: str, task_id: str) -> random.Random:
    seed = int(hashlib.sha256(f"{tag}::claw::{task_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def merged_pool() -> dict[str, Path]:
    return {
        path.name: path.resolve()
        for path in sorted(MERGED_ROOT.iterdir())
        if path.is_dir() and (path / "SKILL.md").is_file()
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").unlink(missing_ok=True)

    all_ids, eval_ids, target_ids = load_claw_ids()
    legacy_oracle_ids = {
        task_id
        for task_id in target_ids
        if (LEGACY_ORACLE_ROOT / task_id / "SKILL.md").is_file()
    }
    retrieval_rows = read_jsonl(RETRIEVAL_JSONL)
    retrieval = {str(row["task_id"]): row for row in retrieval_rows}
    pool = merged_pool()
    if len(retrieval) != 161:
        raise SystemExit(f"expected 161 retrieval rows, got {len(retrieval)}")

    base_rows = []
    for task_id in target_ids:
        if task_id not in retrieval:
            raise SystemExit(f"missing retrieval row for {task_id}")
        context = task_context(task_id)
        row = retrieval[task_id]
        relevant = []
        seen = set()
        for entry in list(row.get("reranked_top10") or []) + list(row.get("coarse_top50") or []):
            name = Path(str(entry["skill_path"])).name
            if name in seen or name not in pool:
                continue
            seen.add(name)
            relevant.append(
                {
                    "name": name,
                    "path": str(pool[name]),
                    "category": "relevant",
                    "rerank_rank": entry.get("rank"),
                    "rerank_score": entry.get("rerank_score"),
                }
            )
            if len(relevant) == N_RELEVANT:
                break
        if len(relevant) != N_RELEVANT:
            raise SystemExit(f"{task_id}: only {len(relevant)} relevant skills")

        excluded = {
            Path(str(entry["skill_path"])).name
            for entry in list(row.get("coarse_top50") or []) + list(row.get("reranked_top10") or [])
        }
        candidates = sorted(name for name in pool if name not in excluded and name not in seen)
        picks = seeded_rng("slate-irrelevant", task_id).sample(candidates, N_IRRELEVANT)
        irrelevant = [
            {"name": name, "path": str(pool[name]), "category": "irrelevant"}
            for name in picks
        ]
        base_rows.append(
            {
                "task_key": task_key(task_id),
                "bench": "claw",
                "task_id": task_id,
                "split": "eval_claw_147",
                "task_yaml": str((CLAW_TASK_ROOT / task_id / "task.yaml").resolve()),
                "task_context": context,
                "relevant": relevant,
                "irrelevant": irrelevant,
                "irrelevant_exclusion": "outside_coarse_top50_and_reranked_top10",
            }
        )

    tasks_text = "".join(f"claw\t{task_id}\n" for task_id in target_ids)
    atomic_write_text(output / "tasks.tsv", tasks_text)
    atomic_write_text(output / "task_ids.txt", "".join(f"{task_id}\n" for task_id in target_ids))
    atomic_write_jsonl(output / "manifest/base_eval_claw_147.jsonl", base_rows)
    report = {
        "builder": BUILDER_ID,
        "all_claw_t_series": len(all_ids),
        "excluded_final_eval70_claw": len(eval_ids),
        "target_tasks": len(target_ids),
        "slate_shape": {
            "oracle": N_ORACLE,
            "misleading": N_MISLEADING,
            "relevant": N_RELEVANT,
            "irrelevant": N_IRRELEVANT,
            "total": SLATE_SIZE,
        },
        "sources": {
            "claw_161": {"path": str(CLAW_161), "sha256": sha256_file(CLAW_161)},
            "eval70_tasks": {"path": str(EVAL70_TASKS), "sha256": sha256_file(EVAL70_TASKS)},
            "retrieval": {"path": str(RETRIEVAL_JSONL), "sha256": sha256_file(RETRIEVAL_JSONL)},
            "merged_root": str(MERGED_ROOT.resolve()),
            "legacy_oracle_root": str(LEGACY_ORACLE_ROOT.resolve()),
            "legacy_oracle_available": len(legacy_oracle_ids),
            "legacy_oracle_missing": sorted(set(target_ids) - legacy_oracle_ids),
        },
        "excluded_task_ids": eval_ids,
        "description_similarity_threshold": DESCRIPTION_SIM_THRESHOLD,
    }
    atomic_write_json(output / "manifest/build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def api_chat(
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    transport_attempts: int = 5,
    response_format: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    data = json.dumps(payload).encode()
    last_error: Exception | None = None
    for attempt in range(transport_attempts):
        try:
            request = urllib.request.Request(
                api_base.rstrip("/") + "/chat/completions",
                data=data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            return str(body["choices"][0]["message"]["content"]).strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError(f"chat failed after {transport_attempts} attempts: {last_error}")


def parse_json_response(text: str) -> dict[str, Any]:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in response")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def api_json_chat(
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout: int,
    semantic_attempts: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(semantic_attempts):
        try:
            attempt_messages = list(messages)
            if attempt:
                retry_instruction = (
                    "\n\nThe previous response was not parseable JSON. Return exactly one "
                    "compact JSON object matching the requested schema now. Do not emit "
                    "analysis, Markdown, comments, or text before or after the object."
                )
                last = dict(attempt_messages[-1])
                last["content"] = str(last.get("content") or "") + retry_instruction
                attempt_messages[-1] = last
            return parse_json_response(
                api_chat(
                    api_base,
                    model,
                    attempt_messages,
                    max_tokens=max_tokens,
                    temperature=0.0 if attempt == 0 else 0.05,
                    timeout=timeout,
                    response_format={"type": "json_object"},
                )
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"JSON judge failed after {semantic_attempts} attempts: {last_error}")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:72].strip("-")


def valid_raw_name(value: str, task_id: str) -> bool:
    return (
        5 <= len(value) <= 72
        and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", value) is not None
        and task_id.lower() not in value
        and BAD_NAME_RE.search(value) is None
    )


def load_base(output: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(output / "manifest/base_eval_claw_147.jsonl")
    if len(rows) != 147:
        raise SystemExit("run prepare first: base manifest is not 147 rows")
    return rows


def cmd_generate_names(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    raw_dir = work / "state/naming_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        task_id = row["task_id"]
        path = raw_dir / f"{task_id}.json"
        if path.is_file():
            return task_id, json.loads(path.read_text(encoding="utf-8"))
        context = row["task_context"]
        user = json.dumps(
            {
                "task_name": context.get("task_name"),
                "category": context.get("category"),
                "prompt": context.get("prompt"),
                "tools": [tool.get("name") for tool in context.get("tools", [])],
            },
            ensure_ascii=False,
        )
        last_error = ""
        for _ in range(args.attempts):
            try:
                payload = parse_json_response(
                    api_chat(
                        args.api_base,
                        args.model,
                        [{"role": "system", "content": NAME_SYSTEM}, {"role": "user", "content": user}],
                        max_tokens=500,
                        temperature=0.65,
                        timeout=args.timeout,
                    )
                )
                oracle = sanitize_name(str(payload["oracle_name"]))
                misleading = [sanitize_name(str(value)) for value in payload["misleading_names"]]
                names = [oracle] + misleading
                if len(misleading) != 5 or len(set(names)) != 6:
                    raise ValueError("need six distinct names")
                if not all(valid_raw_name(name, task_id) for name in names):
                    raise ValueError(f"invalid name set: {names}")
                result = {"task_id": task_id, "oracle_name": oracle, "misleading_names": misleading}
                atomic_write_json(path, result)
                return task_id, result
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        raise RuntimeError(f"{task_id}: naming failed: {last_error}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))
                print(f"[name:error] {exc}", flush=True)
            if index % 20 == 0 or index == len(futures):
                print(f"[name] {index}/{len(futures)} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} naming tasks failed; rerun fills only missing tasks")

    reserved = set(merged_pool())
    assigned = set(reserved)
    plan_rows = []
    for row in rows:
        task_id = row["task_id"]
        raw = json.loads((raw_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        names = [raw["oracle_name"]] + list(raw["misleading_names"])
        resolved = []
        for slot, name in enumerate(names):
            candidate = name
            if candidate in assigned:
                suffix = hashlib.sha256(f"{task_id}::{slot}".encode()).hexdigest()[:6]
                candidate = f"{candidate[:65].rstrip('-')}-{suffix}"
            if candidate in assigned:
                raise SystemExit(f"unresolved global name collision: {candidate}")
            assigned.add(candidate)
            resolved.append(candidate)
        plan_rows.append(
            {
                "task_key": row["task_key"],
                "bench": "claw",
                "task_id": task_id,
                "oracle_name": resolved[0],
                "misleading": [
                    {"name": resolved[index + 1], "strategy": STRATEGIES[index][0]}
                    for index in range(5)
                ],
            }
        )
    atomic_write_jsonl(output / "manifest/naming_plan.jsonl", plan_rows)
    print(f"[name] wrote {len(plan_rows)} rows; generated names are globally unique")


def strip_skill_response(text: str, assigned_name: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = FENCE_RE.sub("", text).strip()
    start = text.find("---")
    if start > 0:
        text = text[start:]
    match = FRONTMATTER_RE.match(text)
    if match:
        try:
            front = yaml.safe_load(match.group("front"))
            description = " ".join(str((front or {}).get("description") or "").split())
            if description:
                body = text[match.end():].lstrip()
                return (
                    f"---\nname: {assigned_name}\n"
                    f"description: {json.dumps(description, ensure_ascii=False)}\n---\n\n"
                    f"{body.rstrip()}\n"
                )
        except Exception:
            pass

    # Qwen occasionally follows the requested schema but omits the two `---`
    # delimiters. Recover only the unambiguous leading name/description form;
    # arbitrary prose still fails closed and is regenerated.
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().lower().startswith("name:"):
        desc_index = next(
            (index for index in range(1, min(6, len(lines))) if lines[index].strip().lower().startswith("description:")),
            -1,
        )
        if desc_index >= 0:
            description = lines[desc_index].split(":", 1)[1].strip().strip("\"'")
            body = "\n".join(lines[desc_index + 1:]).lstrip()
            return (
                f"---\nname: {assigned_name}\n"
                f"description: {json.dumps(description, ensure_ascii=False)}\n---\n\n"
                f"{body.rstrip()}\n"
            )
    return text.rstrip() + "\n"


def normalize_runtime_service_urls(text: str) -> tuple[str, int]:
    """Replace build-time origins/ports with explicit runtime-doc placeholders."""
    text, hardcoded_count = HARDCODED_SERVICE_ORIGIN_RE.subn(
        "<runtime_service_base_url_from_HTTP_Tools>", text
    )
    text, variable_count = UNBOUND_SERVICE_BASE_RE.subn(
        "<runtime_service_base_url_from_HTTP_Tools>", text
    )
    text, bare_port_count = BUILD_TIME_SERVICE_PORT_RE.subn(
        "<runtime_port_from_HTTP_Tools>", text
    )
    return text, hardcoded_count + variable_count + bare_port_count


def redact_build_time_ports(text: str) -> str:
    """Keep routes/schema visible while removing port literals that models tend to copy."""
    return BUILD_TIME_SERVICE_PORT_RE.sub(
        "<build-time-port-redacted-use-HTTP-Tools>", text
    )


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("missing YAML frontmatter")
    front = yaml.safe_load(match.group("front"))
    if not isinstance(front, dict):
        raise ValueError("invalid YAML frontmatter")
    return front, text[match.end():]


def description_of(text: str) -> str:
    front, _ = parse_frontmatter(text)
    return " ".join(str(front.get("description") or "").split())


def replace_description(text: str, assigned_name: str, description: str) -> str:
    _, body = parse_frontmatter(text)
    normalized = " ".join(description.split())
    return (
        f"---\nname: {assigned_name}\n"
        f"description: {json.dumps(normalized, ensure_ascii=False)}\n---\n\n"
        f"{body.lstrip().rstrip()}\n"
    )


def static_skill_problems(text: str, assigned_name: str) -> list[str]:
    problems = []
    try:
        front, body = parse_frontmatter(text)
    except Exception as exc:  # noqa: BLE001
        return [str(exc)]
    if str(front.get("name") or "").strip() != assigned_name:
        problems.append("frontmatter_name_mismatch")
    desc = " ".join(str(front.get("description") or "").split())
    if len(desc) < 20:
        problems.append("description_too_short")
    if len(body.strip()) < 300:
        problems.append("body_too_short")
    if BAD_META_RE.search(text):
        problems.append("meta_contamination")
    if HARDCODED_SERVICE_PORT_RE.search(text):
        problems.append("hardcoded_runtime_service_port")
    if UNBOUND_SERVICE_BASE_RE.search(text):
        problems.append("unbound_runtime_service_base_variable")
    return problems


def all_true(judge: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(judge.get(key) is True for key in keys)


def valid_oracle_audit(meta: dict[str, Any], skill_text: str) -> bool:
    """Prefer a current Claude audit, otherwise accept the original deep audit."""
    if meta.get("claude_audit_id") is not None or meta.get("claude_audited") is not None:
        current_sha = sha256_bytes(skill_text.encode("utf-8"))
        return bool(
            meta.get("claude_audited") is True
            and meta.get("claude_audit_id") == CLAUDE_ORACLE_AUDIT_ID
            and meta.get("claude_audit_model") == "opus"
            and meta.get("claude_audit_effort") == "medium"
            and meta.get("claude_audit_verdict") in {"correct", "needs_rewrite"}
            and meta.get("claude_audit_skill_sha256") == current_sha
            and meta.get("skill_sha256") == current_sha
            and (
                meta.get("claude_audit_verdict") == "correct"
                or meta.get("claude_rewritten") is True
            )
        )
    return bool(
        meta.get("deep_audited") is True
        and all_true(meta.get("deep_audit_judge") or {}, ORACLE_DEEP_JUDGE_KEYS)
        and all_true(meta.get("reference_audit_judge") or {}, ORACLE_REFERENCE_JUDGE_KEYS)
    )


ORACLE_JUDGE_KEYS = (
    "same_primary_deliverable",
    "factually_consistent_with_task",
    "operational_and_actionable",
    "likely_helpful_if_followed",
    "description_correct_and_specific",
    "no_meta_contamination",
)

ORACLE_DEEP_JUDGE_KEYS = (
    "dynamic_endpoint_usage_correct",
    "only_available_openclaw_actions_and_existing_routes",
    "values_and_final_answer_contract_correct",
    "dynamic_dates_and_fixture_behavior_correct",
    "scored_delivery_structure_covered",
    "required_actions_and_components_preserved",
    "zeroing_safety_constraints_covered",
    "factually_consistent_and_actionable",
    "no_evaluation_leakage",
)

ORACLE_REFERENCE_JUDGE_KEYS = (
    "all_required_items_and_actions_covered",
    "no_concrete_rule_conflicts",
    "no_wrong_value_classification_or_state",
    "required_final_output_contract_preserved",
)

MISLEADING_JUDGE_KEYS = (
    "attractive_for_exact_task",
    "same_primary_deliverable",
    "same_action_pipeline",
    "action_framework_wrong",
    "likely_failure_if_followed",
    "logically_distinguishable",
    "description_contains_concrete_conflict",
    "description_not_identical",
    "not_salvageable_by_correct_scaffold",
    "no_correct_rescue_hints",
    "no_meta_contamination",
)

MISLEADING_CONFLICT_KEYS = (
    "rules_mutually_incompatible",
    "same_task_item_or_decision_point",
    "affected_item_outcomes_differ",
    "wrong_rule_controls_first_action",
    "wrong_rule_changes_required_outcome",
    "candidate_rule_quote_verified",
    "authoritative_rule_quote_verified",
    "affected_task_item_quote_verified",
    "outcome_strings_differ_verified",
    "actual_task_instance_evidence_verified",
)

MISLEADING_OUTCOME_BASE_KEYS = (
    "outcomes_differ_on_frozen_instance",
    "no_hypothetical_or_unobserved_dependency",
    "description_rule_controls_procedure",
    "candidate_failure_caused_by_wrong_rule",
    "candidate_rule_quote_verified",
    "authoritative_rule_quote_verified",
    "affected_task_item_quote_verified",
    "actual_input_quote_verified",
    "outcome_strings_differ_verified",
    "numeric_check_required_verified",
    "numeric_predicate_result_verified",
)

MISLEADING_FALSIFIER_KEYS = (
    "proof_uses_only_observed_instance",
    "candidate_rule_applied_correctly",
    "authoritative_rule_applied_correctly",
    "outcomes_really_differ",
    "no_same_outcome_or_self_rejection",
    "numeric_reasoning_consistent",
)

MISLEADING_OUTCOME_KEYS = MISLEADING_OUTCOME_BASE_KEYS + tuple(
    f"falsifier_{key}" for key in MISLEADING_FALSIFIER_KEYS
)


def valid_outcome_audit(meta: dict[str, Any]) -> bool:
    judge = meta.get("outcome_audit_judge") or {}
    return bool(
        meta.get("outcome_audited") is True
        and meta.get("outcome_audit_id") == OUTCOME_AUDIT_ID
        and all_true(judge, MISLEADING_OUTCOME_KEYS)
        and not evidence_is_unobserved(str(judge.get("evidence") or ""))
        and not SELF_REJECT_EVIDENCE_RE.search(str(judge.get("evidence") or ""))
    )


def naming_map(output: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(output / "manifest/naming_plan.jsonl")
    if len(rows) != 147:
        raise SystemExit("run generate-names first: naming plan is not 147 rows")
    return {row["task_id"]: row for row in rows}


def accepted_skill(output: Path, name: str, category: str, task_id: str) -> bool:
    skill = output / "skills" / name / "SKILL.md"
    meta = output / "skills" / name / "meta.json"
    if not skill.is_file() or not meta.is_file():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("builder") == BUILDER_ID
        and payload.get("category") == category
        and payload.get("task_id") == task_id
        and payload.get("accepted") is True
    )


def write_accepted_skill(output: Path, name: str, text: str, meta: dict[str, Any]) -> None:
    target = output / "skills" / name
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target / "SKILL.md", text)
    atomic_write_json(target / "meta.json", meta)


def cmd_generate_oracles(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    log_path = work / "logs/oracle_generation.jsonl"
    log_lock = threading.Lock()

    def one(row: dict[str, Any]) -> str:
        task_id = row["task_id"]
        name = names[task_id]["oracle_name"]
        if accepted_skill(output, name, "oracle", task_id):
            return "exists"
        grounding, grounding_sources = oracle_grounding_bundle(task_id)
        grounding_sha256 = sha256_bytes(grounding.encode())
        rejection_path = work / "state/oracle_rejections" / f"{task_id}.json"
        feedback = rejection_path.read_text(encoding="utf-8") if rejection_path.is_file() else ""
        generation_grounding = grounding
        if feedback:
            generation_grounding = generation_grounding.split(
                "\n\n===== FALLIBLE legacy oracle draft", 1
            )[0]
        generation_grounding = redact_build_time_ports(generation_grounding)
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            user = (
                f"Assigned name: {name}\n\nGrounding evidence follows. AUTHORITATIVE sections "
                f"override the FALLIBLE legacy draft. Build-time ports are deliberately redacted; "
                f"use the exact runtime URLs from HTTP Tools:\n{generation_grounding}\n"
                + (f"\nPrevious judge feedback to fix:\n{feedback}\n" if feedback else "")
            )
            try:
                text = strip_skill_response(
                    api_chat(
                        args.api_base,
                        args.model,
                        [{"role": "system", "content": ORACLE_SYSTEM}, {"role": "user", "content": user}],
                        max_tokens=3600,
                        temperature=0.25,
                        timeout=args.timeout,
                    ),
                    name,
                )
                text, runtime_url_normalizations = normalize_runtime_service_urls(text)
                problems = static_skill_problems(text, name)
                if problems:
                    raise ValueError(",".join(problems))
                judge_user = (
                    f"Grounding evidence:\n{grounding}\n\nCandidate correct SKILL.md:\n{text}"
                )
                judge = parse_json_response(
                    api_chat(
                        args.api_base,
                        args.model,
                        [
                            {"role": "system", "content": ORACLE_JUDGE_SYSTEM},
                            {"role": "user", "content": judge_user},
                        ],
                        max_tokens=700,
                        temperature=0.0,
                        timeout=args.timeout,
                    )
                )
                if not all_true(judge, ORACLE_JUDGE_KEYS):
                    feedback = json.dumps(judge, ensure_ascii=False)
                    raise ValueError("judge_reject:" + feedback)
                meta = {
                    "builder": BUILDER_ID,
                    "accepted": True,
                    "category": "oracle",
                    "task_id": task_id,
                    "task_key": row["task_key"],
                    "name": name,
                    "attempt": attempt,
                    "runtime_url_normalizations": runtime_url_normalizations,
                    "grounding_sha256": grounding_sha256,
                    "grounding_sources": grounding_sources,
                    "judge": judge,
                    "skill_sha256": sha256_bytes(text.encode()),
                }
                write_accepted_skill(output, name, text, meta)
                append_jsonl(log_path, {**meta, "status": "generated"}, log_lock)
                return "generated"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                feedback = last_error[-1800:]
        append_jsonl(
            log_path,
            {"task_id": task_id, "name": name, "status": "error", "error": last_error},
            log_lock,
        )
        raise RuntimeError(f"{task_id}: oracle failed: {last_error}")

    run_jobs(rows, one, args.workers, "oracle")


def cmd_audit_oracles(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    log_path = work / "logs/oracle_deep_audit.jsonl"
    log_lock = threading.Lock()

    def remove_task_generated_skills(task_id: str) -> None:
        plan = names[task_id]
        for name in [plan["oracle_name"]] + [entry["name"] for entry in plan["misleading"]]:
            target = output / "skills" / name
            meta_path = target / "meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                if meta.get("task_id") != task_id or meta.get("category") not in {"oracle", "misleading"}:
                    continue
            if target.is_dir():
                shutil.rmtree(target)

    def one(row: dict[str, Any]) -> str:
        task_id = row["task_id"]
        name = names[task_id]["oracle_name"]
        skill_path = output / "skills" / name / "SKILL.md"
        meta_path = output / "skills" / name / "meta.json"
        if not accepted_skill(output, name, "oracle", task_id):
            raise RuntimeError(f"{task_id}: accepted oracle missing before deep audit")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        skill_sha256 = sha256_file(skill_path)
        candidate = skill_path.read_text(encoding="utf-8")
        static_problems = static_skill_problems(candidate, name)
        has_hardcoded_port = bool(HARDCODED_SERVICE_PORT_RE.search(candidate))
        has_unbound_base = bool(UNBOUND_SERVICE_BASE_RE.search(candidate))
        if (
            meta.get("deep_audited") is True
            and meta.get("skill_sha256") == skill_sha256
            and not static_problems
            and all_true(meta.get("deep_audit_judge") or {}, ORACLE_DEEP_JUDGE_KEYS)
            and all_true(
                meta.get("reference_audit_judge") or {}, ORACLE_REFERENCE_JUDGE_KEYS
            )
        ):
            return "exists"
        grounding, grounding_sources = oracle_grounding_bundle(task_id)
        reference_judge: dict[str, Any] = {}
        if static_problems:
            judge = {key: True for key in ORACLE_DEEP_JUDGE_KEYS}
            if has_hardcoded_port or has_unbound_base:
                judge["dynamic_endpoint_usage_correct"] = False
            if "meta_contamination" in static_problems:
                judge["no_evaluation_leakage"] = False
            if not (has_hardcoded_port or has_unbound_base or "meta_contamination" in static_problems):
                judge["factually_consistent_and_actionable"] = False
            if has_hardcoded_port:
                judge["summary"] = (
                    "Candidate contains a literal localhost service port; replace every service URL "
                    "with the exact URL shown in the current HTTP Tools documentation."
                )
            elif has_unbound_base:
                judge["summary"] = (
                    "Candidate executes an undefined *_BASE_URL shell variable; use a clear "
                    "angle-bracket placeholder and require substitution from current HTTP Tools docs."
                )
            else:
                judge["summary"] = "Static oracle audit rejected: " + ",".join(static_problems)
        else:
            judge = api_json_chat(
                args.api_base,
                args.model,
                [
                    {"role": "system", "content": ORACLE_DEEP_JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Grounding evidence:\n{grounding}\n\nCandidate SKILL.md:\n{candidate}",
                    },
                ],
                max_tokens=1000,
                timeout=args.timeout,
            )
            context_json = json.dumps(row["task_context"], ensure_ascii=False, indent=2)
            reference_judge = oracle_reference_consistency_judge(
                args, context_json, grounding, candidate
            )
        # Port relocation is a deterministic runtime contract, not a semantic
        # judgment. Grounding contains build-time 91xx defaults which can make
        # an LLM judge invert this criterion; enforce it from candidate bytes.
        judge["dynamic_endpoint_usage_correct"] = not (has_hardcoded_port or has_unbound_base)
        passed = all_true(judge, ORACLE_DEEP_JUDGE_KEYS) and all_true(
            reference_judge, ORACLE_REFERENCE_JUDGE_KEYS
        )
        append_jsonl(
            log_path,
            {
                "task_id": task_id,
                "name": name,
                "passed": passed,
                "judge": judge,
                "reference_judge": reference_judge,
            },
            log_lock,
        )
        if not passed:
            rejection_summary = (
                reference_judge.get("summary", "")
                if reference_judge
                and not all_true(reference_judge, ORACLE_REFERENCE_JUDGE_KEYS)
                else judge.get("summary", "")
            )
            atomic_write_json(
                work / "state/oracle_rejections" / f"{task_id}.json",
                {
                    "task_id": task_id,
                    "judge": judge,
                    "reference_judge": reference_judge,
                },
            )
            remove_task_generated_skills(task_id)
            raise RuntimeError(f"{task_id}: deep oracle audit rejected: {rejection_summary}")
        meta["deep_audited"] = True
        meta["deep_audit_judge"] = judge
        meta["reference_audit_judge"] = reference_judge
        meta["deep_audit_sources"] = grounding_sources
        meta["skill_sha256"] = skill_sha256
        atomic_write_json(meta_path, meta)
        (work / "state/oracle_rejections" / f"{task_id}.json").unlink(missing_ok=True)
        return "generated"

    run_jobs(rows, one, args.workers, "oracle-audit")


def description_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, " ".join(left.lower().split()), " ".join(right.lower().split())).ratio()


def normalized_evidence_text(value: Any) -> str:
    text = str(value or "").strip().strip("\"'`\u2018\u2019\u201c\u201d")
    text = text.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"')
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\uff0c": ",",
                "\uff1a": ":",
                "\uff08": "(",
                "\uff09": ")",
            }
        )
    )
    text = re.sub(r"[`*_#]+", "", text)
    return " ".join(text.lower().split())


def quote_is_grounded(source: str, quote: Any, *, min_chars: int = 16) -> bool:
    normalized_quote = normalized_evidence_text(quote)
    effective_min_chars = min_chars
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized_quote):
        effective_min_chars = min(min_chars, 8)
    if len(normalized_quote) < effective_min_chars:
        return False
    normalized_source = normalized_evidence_text(source)
    if normalized_quote in normalized_source:
        return True
    # JSON/YAML/Markdown rendering can change only punctuation around an otherwise
    # verbatim clause. Keep word order and characters exact while ignoring that
    # serialization noise; semantic paraphrases still fail this containment check.
    compact_quote = re.sub(r"[\W_]+", "", normalized_quote, flags=re.UNICODE)
    compact_source = re.sub(r"[\W_]+", "", normalized_source, flags=re.UNICODE)
    return (
        len(compact_quote) >= max(2, effective_min_chars // 2)
        and compact_quote in compact_source
    )


def oracle_reference_consistency_judge(
    args: argparse.Namespace,
    context_json: str,
    grounding: str,
    candidate: str,
) -> dict[str, Any]:
    result = api_json_chat(
        args.api_base,
        args.model,
        [
            {"role": "system", "content": ORACLE_REFERENCE_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Authoritative task context:\n{context_json}\n\n"
                    f"Authoritative grader/mock-service evidence (runtime source wins on schema "
                    f"conflicts):\n{grounding}\n\n"
                    f"Candidate correct SKILL.md:\n{candidate}"
                ),
            },
        ],
        max_tokens=900,
        timeout=args.timeout,
    )
    result["candidate_conflict_quote_verified"] = quote_is_grounded(
        candidate, result.get("candidate_conflict_quote")
    )
    result["authoritative_conflict_quote_verified"] = quote_is_grounded(
        context_json + "\n" + grounding, result.get("authoritative_conflict_quote")
    )
    return result


def focused_misleading_conflict_judge(
    args: argparse.Namespace,
    context_json: str,
    oracle_text: str,
    candidate: str,
) -> dict[str, Any]:
    _, candidate_body = parse_frontmatter(candidate)
    user = (
        f"AUTHORITATIVE task context:\n"
        f"{context_json}\n\n"
        f"Independently audited correct SKILL.md (secondary authoritative source):\n"
        f"{oracle_text}\n\n"
        f"Candidate description:\n{description_of(candidate)}\n\n"
        f"Candidate opening procedure:\n{candidate_body[:5000]}"
    )
    result = api_json_chat(
        args.api_base,
        args.model,
        [
            {"role": "system", "content": MISLEADING_CONFLICT_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=700,
        timeout=args.timeout,
    )
    result["candidate_rule_quote_verified"] = quote_is_grounded(
        candidate, result.get("candidate_rule_quote")
    )
    result["authoritative_rule_quote_verified"] = quote_is_grounded(
        context_json + "\n" + oracle_text, result.get("authoritative_rule_quote")
    )
    result["affected_task_item_quote_verified"] = quote_is_grounded(
        context_json + "\n" + oracle_text,
        result.get("affected_task_item_quote"),
        min_chars=2,
    )
    candidate_outcome = normalized_evidence_text(
        result.get("candidate_outcome_for_affected_item")
    )
    authoritative_outcome = normalized_evidence_text(
        result.get("authoritative_outcome_for_affected_item")
    )
    result["outcome_strings_differ_verified"] = bool(
        candidate_outcome
        and authoritative_outcome
        and candidate_outcome != authoritative_outcome
    )
    result["actual_task_instance_evidence_verified"] = not bool(
        HYPOTHETICAL_EVIDENCE_RE.search(str(result.get("evidence") or ""))
    )
    evidence_keys = (
        "candidate_rule_quote_verified",
        "authoritative_rule_quote_verified",
        "affected_task_item_quote_verified",
        "outcome_strings_differ_verified",
        "actual_task_instance_evidence_verified",
    )
    if not all(result[key] for key in evidence_keys):
        prior = str(result.get("evidence") or "").strip()
        result["evidence"] = (
            "Deterministic conflict-evidence checks failed: "
            + ", ".join(f"{key}={result[key]}" for key in evidence_keys)
            + "."
            + (f" {prior}" if prior else "")
        )
    return result


def evaluate_numeric_predicate(left: Any, operator: Any, right: Any) -> bool | None:
    if isinstance(left, bool) or isinstance(right, bool):
        return None
    try:
        lhs = float(left)
        rhs = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return None
    operations = {
        "<": lambda: lhs < rhs,
        "<=": lambda: lhs <= rhs,
        ">": lambda: lhs > rhs,
        ">=": lambda: lhs >= rhs,
        "==": lambda: lhs == rhs,
        "!=": lambda: lhs != rhs,
    }
    operation = operations.get(str(operator or "").strip())
    return operation() if operation is not None else None


def frozen_outcome_judge(
    args: argparse.Namespace,
    context_json: str,
    oracle_text: str,
    candidate: str,
) -> dict[str, Any]:
    _, candidate_body = parse_frontmatter(candidate)
    judge_user = (
        f"AUTHORITATIVE task context:\n{context_json}\n\n"
        f"Independently audited correct SKILL.md:\n{oracle_text}\n\n"
        f"Candidate description:\n{description_of(candidate)}\n\n"
        f"Candidate procedure:\n{candidate_body[:7000]}"
    )
    result = api_json_chat(
        args.api_base,
        args.model,
        [
            {"role": "system", "content": MISLEADING_OUTCOME_JUDGE_SYSTEM},
            {"role": "user", "content": judge_user},
        ],
        max_tokens=1400,
        timeout=args.timeout,
    )
    authoritative_source = context_json + "\n" + oracle_text

    def quote_verification(candidate_result: dict[str, Any]) -> dict[str, bool]:
        return {
            "candidate_rule_quote_verified": quote_is_grounded(
                candidate, candidate_result.get("candidate_rule_quote")
            ),
            "authoritative_rule_quote_verified": quote_is_grounded(
                authoritative_source, candidate_result.get("authoritative_rule_quote")
            ),
            "affected_task_item_quote_verified": quote_is_grounded(
                authoritative_source,
                candidate_result.get("affected_task_item_quote"),
                min_chars=2,
            ),
            "actual_input_quote_verified": quote_is_grounded(
                authoritative_source,
                candidate_result.get("actual_input_quote"),
                min_chars=2,
            ),
        }

    quote_checks = quote_verification(result)
    if not all(quote_checks.values()):
        failed_quotes = ", ".join(key for key, passed in quote_checks.items() if not passed)
        repair_user = (
            judge_user
            + "\n\nYour previous JSON failed literal quote validation for: "
            + failed_quotes
            + ". Reassess and return the complete JSON again. Every quote field must be one "
            "exact contiguous substring copied from its named source. Do not use ellipses, "
            "paraphrases, translations, or joined fragments.\nPrevious JSON:\n"
            + json.dumps(result, ensure_ascii=False)
        )
        result = api_json_chat(
            args.api_base,
            args.model,
            [
                {"role": "system", "content": MISLEADING_OUTCOME_JUDGE_SYSTEM},
                {"role": "user", "content": repair_user},
            ],
            max_tokens=1400,
            timeout=args.timeout,
        )
        quote_checks = quote_verification(result)
    result.update(quote_checks)
    candidate_outcome = normalized_evidence_text(
        result.get("candidate_outcome_for_affected_item")
    )
    authoritative_outcome = normalized_evidence_text(
        result.get("authoritative_outcome_for_affected_item")
    )
    result["outcome_strings_differ_verified"] = bool(
        candidate_outcome
        and authoritative_outcome
        and candidate_outcome != authoritative_outcome
    )

    evidence = str(result.get("evidence") or "")
    if evidence_is_unobserved(evidence):
        result["no_hypothetical_or_unobserved_dependency"] = False
    if SELF_REJECT_EVIDENCE_RE.search(evidence):
        result["outcomes_differ_on_frozen_instance"] = False
        result["candidate_failure_caused_by_wrong_rule"] = False

    numeric_source = str(result.get("candidate_rule_quote") or "")
    numeric_rule_detected = bool(NUMERIC_DECISION_RE.search(numeric_source))
    if re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", numeric_source):
        without_dates = re.sub(
            r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", "", numeric_source
        )
        if not re.search(
            r"(?:\bpercent(?:age)?\b|\bratio\b|\bcount\b|\bsize\b|"
            r"\bduration\b|\d+(?:\.\d+)?\s*(?:%|days?|hours?|minutes?))",
            without_dates,
            re.I,
        ):
            numeric_rule_detected = False
    numeric_required = result.get("numeric_check_required") is True
    result["numeric_rule_detected"] = numeric_rule_detected
    result["numeric_check_required_verified"] = bool(
        numeric_required if numeric_rule_detected else True
    )
    if numeric_rule_detected and numeric_required:
        computed = evaluate_numeric_predicate(
            result.get("numeric_lhs"),
            result.get("numeric_operator"),
            result.get("numeric_rhs"),
        )
        result["numeric_predicate_result_verified"] = bool(
            computed is not None
            and isinstance(result.get("numeric_predicate_result"), bool)
            and result.get("numeric_predicate_result") is computed
        )
        result["numeric_predicate_python_result"] = computed
    else:
        result["numeric_predicate_result_verified"] = not numeric_rule_detected
        result["numeric_predicate_python_result"] = None

    falsifier: dict[str, Any] = {}
    if all_true(result, MISLEADING_OUTCOME_BASE_KEYS):
        falsifier_user = (
            f"AUTHORITATIVE task context:\n{context_json}\n\n"
            f"Independently audited correct SKILL.md:\n{oracle_text}\n\n"
            f"Candidate hard-negative SKILL.md:\n{candidate}\n\n"
            "Proposed frozen-instance proof JSON:\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
        falsifier = api_json_chat(
            args.api_base,
            args.model,
            [
                {"role": "system", "content": MISLEADING_OUTCOME_FALSIFIER_SYSTEM},
                {"role": "user", "content": falsifier_user},
            ],
            max_tokens=900,
            timeout=args.timeout,
        )
        falsifier_evidence = str(falsifier.get("evidence") or "")
        if evidence_is_unobserved(falsifier_evidence):
            falsifier["proof_uses_only_observed_instance"] = False
        if SELF_REJECT_EVIDENCE_RE.search(falsifier_evidence):
            falsifier["no_same_outcome_or_self_rejection"] = False
    result["falsifier_judge"] = falsifier
    for key in MISLEADING_FALSIFIER_KEYS:
        result[f"falsifier_{key}"] = falsifier.get(key) is True

    if not all_true(result, MISLEADING_OUTCOME_KEYS):
        prior = str(result.get("evidence") or "").strip()
        critique = str(falsifier.get("evidence") or "").strip()
        failed = [key for key in MISLEADING_OUTCOME_KEYS if result.get(key) is not True]
        result["evidence"] = (
            "Frozen-instance outcome checks failed: " + ", ".join(failed) + "."
            + (f" {prior}" if prior else "")
            + (f" Falsifier: {critique}" if critique else "")
        )
    return result


def cmd_generate_misleading(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    log_path = work / "logs/misleading_generation.jsonl"
    log_lock = threading.Lock()
    jobs = []
    for row in rows:
        for index, entry in enumerate(names[row["task_id"]]["misleading"]):
            jobs.append((row, index, entry))

    def one(job: tuple[dict[str, Any], int, dict[str, Any]]) -> str:
        row, index, entry = job
        task_id = row["task_id"]
        name = entry["name"]
        if accepted_skill(output, name, "misleading", task_id):
            return "exists"
        oracle_name = names[task_id]["oracle_name"]
        oracle_path = output / "skills" / oracle_name / "SKILL.md"
        if not oracle_path.is_file():
            raise RuntimeError(f"{task_id}: oracle skill missing before misleading generation")
        oracle_text = oracle_path.read_text(encoding="utf-8")
        oracle_desc = description_of(oracle_text)
        context_json = json.dumps(row["task_context"], ensure_ascii=False, indent=2)
        generation_context_json = redact_build_time_ports(context_json)
        strategy_name, strategy_text = STRATEGIES[index]
        rejection_path = work / "state/misleading_rejections" / f"{task_id}__{name}.json"
        feedback = rejection_path.read_text(encoding="utf-8") if rejection_path.is_file() else ""
        feedback_instruction = ""
        if feedback:
            try:
                feedback_instruction = str(json.loads(feedback).get("instruction") or "")
            except json.JSONDecodeError:
                pass
        last_error = ""
        for attempt in range(1, args.attempts + 1):
            system = MISLEADING_SYSTEM.format(strategy_text=strategy_text)
            user = (
                f"Assigned name: {name}\nStrategy id: {strategy_name}\n\n"
                f"Task definition (build-time ports redacted):\n{generation_context_json}\n\n"
                f"Correct SKILL.md:\n{oracle_text}\n"
                + (
                    "\nPrevious rejection to fix:\n"
                    "If it reports an unverified authoritative rule or affected item, abandon "
                    "that false premise entirely and choose a different premise grounded in a "
                    "literal object, field, value, action, or deliverable named below. Do not "
                    "retry the same unsupported premise with new wording.\n"
                    f"{feedback}\n"
                    if feedback
                    else ""
                )
                + (
                    "\nNON-NEGOTIABLE RETRY REQUIREMENT:\n"
                    f"{feedback_instruction}\n"
                    "The candidate must implement this exact false rule in both its "
                    "description and procedure. Do not substitute another premise.\n"
                    if feedback_instruction
                    else ""
                )
            )
            try:
                text = strip_skill_response(
                    api_chat(
                        args.api_base,
                        args.model,
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens=3200,
                        temperature=args.temperature,
                        timeout=args.timeout,
                    ),
                    name,
                )
                text, runtime_url_normalizations = normalize_runtime_service_urls(text)
                problems = static_skill_problems(text, name)
                candidate_desc = description_of(text)
                similarity = description_similarity(oracle_desc, candidate_desc)
                if similarity > DESCRIPTION_SIM_THRESHOLD:
                    previous_rewrite = candidate_desc
                    for rewrite_attempt in range(1, 4):
                        rewrite_user = (
                            f"Task definition:\n{context_json}\n\n"
                            f"Correct description that must not be copied:\n{oracle_desc}\n\n"
                            f"Corruption strategy:\n{strategy_name}: {strategy_text}\n\n"
                            f"Candidate hard-negative SKILL.md whose body must remain unchanged:\n{text}\n\n"
                            f"Previous description similarity was {similarity:.4f}; rewrite more "
                            f"distinctly than this prior wording:\n{previous_rewrite}"
                        )
                        rewrite = parse_json_response(
                            api_chat(
                                args.api_base,
                                args.model,
                                [
                                    {"role": "system", "content": MISLEADING_DESCRIPTION_REWRITE_SYSTEM},
                                    {"role": "user", "content": rewrite_user},
                                ],
                                max_tokens=500,
                                temperature=0.45 + 0.1 * (rewrite_attempt - 1),
                                timeout=args.timeout,
                            )
                        )
                        rewritten_desc = " ".join(str(rewrite.get("description") or "").split())
                        concrete_conflict = " ".join(str(rewrite.get("concrete_conflict") or "").split())
                        if len(rewritten_desc) < 20 or len(concrete_conflict) < 8:
                            raise ValueError("description_rewrite_missing_concrete_conflict")
                        if BAD_META_RE.search(rewritten_desc):
                            raise ValueError("description_rewrite_meta_contamination")
                        previous_rewrite = rewritten_desc
                        similarity = description_similarity(oracle_desc, rewritten_desc)
                        if similarity <= DESCRIPTION_SIM_THRESHOLD:
                            text = replace_description(text, name, rewritten_desc)
                            candidate_desc = rewritten_desc
                            break
                    problems = static_skill_problems(text, name)
                    if similarity > DESCRIPTION_SIM_THRESHOLD:
                        problems.append(f"description_too_oracle_like_after_rewrite:{similarity:.4f}")
                ratio = len(text) / max(1, len(oracle_text))
                if not 0.18 <= ratio <= 1.8:
                    problems.append(f"length_ratio:{ratio:.3f}")
                if problems:
                    raise ValueError(",".join(problems))
                judge_user = (
                    f"Task definition:\n{context_json}\n\nCorrect SKILL.md:\n{oracle_text}\n\n"
                    f"Candidate hard-negative SKILL.md:\n{text}"
                )
                judge = parse_json_response(
                    api_chat(
                        args.api_base,
                        args.model,
                        [
                            {"role": "system", "content": MISLEADING_JUDGE_SYSTEM},
                            {"role": "user", "content": judge_user},
                        ],
                        max_tokens=900,
                        temperature=0.0,
                        timeout=args.timeout,
                    )
                )
                if not all_true(judge, MISLEADING_JUDGE_KEYS):
                    feedback = json.dumps(judge, ensure_ascii=False)
                    raise ValueError("judge_reject:" + feedback)
                conflict_judge = focused_misleading_conflict_judge(
                    args, context_json, oracle_text, text
                )
                if not all_true(conflict_judge, MISLEADING_CONFLICT_KEYS):
                    feedback = json.dumps(conflict_judge, ensure_ascii=False)
                    raise ValueError("focused_conflict_reject:" + feedback)
                meta = {
                    "builder": BUILDER_ID,
                    "accepted": True,
                    "audited": False,
                    "outcome_audited": False,
                    "category": "misleading",
                    "task_id": task_id,
                    "task_key": row["task_key"],
                    "name": name,
                    "oracle_name": oracle_name,
                    "strategy": strategy_name,
                    "attempt": attempt,
                    "runtime_url_normalizations": runtime_url_normalizations,
                    "description_similarity": round(similarity, 6),
                    "judge": judge,
                    "conflict_judge": conflict_judge,
                    "skill_sha256": sha256_bytes(text.encode()),
                }
                write_accepted_skill(output, name, text, meta)
                append_jsonl(log_path, {**meta, "status": "generated"}, log_lock)
                return "generated"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                feedback = last_error[-1800:]
        append_jsonl(
            log_path,
            {
                "task_id": task_id,
                "name": name,
                "strategy": strategy_name,
                "status": "error",
                "error": last_error,
            },
            log_lock,
        )
        atomic_write_rejection(
            rejection_path,
            {"task_id": task_id, "name": name, "error": last_error},
        )
        raise RuntimeError(f"{task_id}/{name}: misleading failed: {last_error}")

    run_jobs(jobs, one, args.workers, "misleading")


def run_jobs(items: list[Any], fn, workers: int, label: str) -> None:
    counts = Counter()
    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))
                print(f"[{label}:error] {exc}", flush=True)
            if index % 20 == 0 or index == len(futures):
                rate = index / max(0.001, time.time() - started) * 60
                print(
                    f"[{label}] {index}/{len(futures)} generated={counts['generated']} "
                    f"exists={counts['exists']} errors={len(failures)} rate={rate:.1f}/min",
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{len(failures)} {label} jobs failed; rerun fills only missing jobs")


def cmd_audit_misleading(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    row_by_id = {row["task_id"]: row for row in rows}
    log_path = work / "logs/misleading_audit.jsonl"
    log_lock = threading.Lock()
    jobs = []
    for task_id, plan in names.items():
        for entry in plan["misleading"]:
            jobs.append((task_id, entry))

    def remove_candidate(skill_path: Path, meta_path: Path) -> None:
        skill_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        try:
            skill_path.parent.rmdir()
        except OSError:
            pass

    def one(job: tuple[str, dict[str, Any]]) -> str:
        task_id, entry = job
        name = entry["name"]
        skill_path = output / "skills" / name / "SKILL.md"
        meta_path = output / "skills" / name / "meta.json"
        rejection_path = work / "state/misleading_rejections" / f"{task_id}__{name}.json"
        if not skill_path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"{task_id}/{name}: candidate missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        oracle_name = names[task_id]["oracle_name"]
        oracle_text = (output / "skills" / oracle_name / "SKILL.md").read_text(encoding="utf-8")
        candidate = skill_path.read_text(encoding="utf-8")
        static_problems = static_skill_problems(candidate, name)
        similarity = description_similarity(description_of(oracle_text), description_of(candidate))
        if similarity > DESCRIPTION_SIM_THRESHOLD:
            static_problems.append(f"description_too_oracle_like:{similarity:.4f}")
        if static_problems:
            append_jsonl(
                log_path,
                {
                    "task_id": task_id,
                    "name": name,
                    "passed": False,
                    "static_problems": static_problems,
                },
                log_lock,
            )
            atomic_write_rejection(
                rejection_path,
                {"task_id": task_id, "name": name, "static_problems": static_problems},
            )
            remove_candidate(skill_path, meta_path)
            raise RuntimeError(f"{task_id}/{name}: static audit rejected: {','.join(static_problems)}")
        if (
            meta.get("audited") is True
            and meta.get("skill_sha256") == sha256_file(skill_path)
            and all_true(meta.get("audit_conflict_judge") or {}, MISLEADING_CONFLICT_KEYS)
        ):
            return "exists"
        context_json = json.dumps(row_by_id[task_id]["task_context"], ensure_ascii=False, indent=2)
        judge_user = (
            f"Task definition:\n{context_json}\n\nCorrect SKILL.md:\n{oracle_text}\n\n"
            f"Candidate hard-negative SKILL.md:\n{candidate}"
        )
        try:
            judge = api_json_chat(
                args.api_base,
                args.model,
                [
                    {"role": "system", "content": MISLEADING_JUDGE_SYSTEM},
                    {"role": "user", "content": judge_user},
                ],
                max_tokens=900,
                timeout=args.timeout,
            )
            conflict_judge = focused_misleading_conflict_judge(
                args, context_json, oracle_text, candidate
            )
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                log_path,
                {
                    "task_id": task_id,
                    "name": name,
                    "passed": False,
                    "audit_call_error": str(exc),
                },
                log_lock,
            )
            atomic_write_rejection(
                rejection_path,
                {"task_id": task_id, "name": name, "audit_call_error": str(exc)},
            )
            remove_candidate(skill_path, meta_path)
            raise RuntimeError(
                f"{task_id}/{name}: independent audit call failed: {exc}"
            ) from exc
        passed = all_true(judge, MISLEADING_JUDGE_KEYS) and all_true(
            conflict_judge, MISLEADING_CONFLICT_KEYS
        )
        append_jsonl(
            log_path,
            {
                "task_id": task_id,
                "name": name,
                "passed": passed,
                "judge": judge,
                "conflict_judge": conflict_judge,
            },
            log_lock,
        )
        if not passed:
            atomic_write_rejection(
                rejection_path,
                {
                    "task_id": task_id,
                    "name": name,
                    "judge": judge,
                    "conflict_judge": conflict_judge,
                },
            )
            remove_candidate(skill_path, meta_path)
            reason = (
                conflict_judge.get("evidence", "")
                if not all_true(conflict_judge, MISLEADING_CONFLICT_KEYS)
                else judge.get("summary", "")
            )
            raise RuntimeError(f"{task_id}/{name}: independent audit rejected: {reason}")
        meta["audited"] = True
        meta["audit_judge"] = judge
        meta["audit_conflict_judge"] = conflict_judge
        meta["skill_sha256"] = sha256_file(skill_path)
        atomic_write_json(meta_path, meta)
        return "generated"

    run_jobs(jobs, one, args.workers, "audit")


def cmd_audit_outcomes(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    work = Path(args.work_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    row_by_id = {row["task_id"]: row for row in rows}
    outcome_context_by_id = {}
    for task_id, row in row_by_id.items():
        grounding, _ = oracle_grounding_bundle(task_id, include_legacy=False)
        outcome_context_by_id[task_id] = (
            json.dumps(row["task_context"], ensure_ascii=False, indent=2)
            + "\n\n===== AUTHORITATIVE frozen grounding evidence =====\n"
            + grounding
        )
    log_path = work / "logs/misleading_outcome_audit.jsonl"
    log_lock = threading.Lock()
    jobs = [
        (task_id, entry)
        for task_id, plan in names.items()
        for entry in plan["misleading"]
    ]

    def remove_candidate(skill_path: Path, meta_path: Path) -> None:
        skill_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        try:
            skill_path.parent.rmdir()
        except OSError:
            pass

    def one(job: tuple[str, dict[str, Any]]) -> str:
        task_id, entry = job
        name = entry["name"]
        skill_path = output / "skills" / name / "SKILL.md"
        meta_path = output / "skills" / name / "meta.json"
        rejection_path = work / "state/misleading_rejections" / f"{task_id}__{name}.json"
        if not skill_path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"{task_id}/{name}: candidate missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("audited") is not True or not all_true(
            meta.get("audit_conflict_judge") or {}, MISLEADING_CONFLICT_KEYS
        ):
            raise RuntimeError(f"{task_id}/{name}: run audit-misleading first")
        skill_sha256 = sha256_file(skill_path)
        if (
            valid_outcome_audit(meta)
            and meta.get("skill_sha256") == skill_sha256
        ):
            rejection_path.unlink(missing_ok=True)
            return "exists"

        oracle_name = names[task_id]["oracle_name"]
        oracle_text = (output / "skills" / oracle_name / "SKILL.md").read_text(
            encoding="utf-8"
        )
        candidate = skill_path.read_text(encoding="utf-8")
        context_json = outcome_context_by_id[task_id]
        try:
            judge = frozen_outcome_judge(
                args, context_json, oracle_text, candidate
            )
        except Exception as exc:  # noqa: BLE001
            append_jsonl(
                log_path,
                {
                    "task_id": task_id,
                    "name": name,
                    "passed": False,
                    "audit_call_error": str(exc),
                },
                log_lock,
            )
            raise RuntimeError(
                f"{task_id}/{name}: frozen-outcome audit call failed: {exc}"
            ) from exc

        passed = all_true(judge, MISLEADING_OUTCOME_KEYS)
        append_jsonl(
            log_path,
            {
                "task_id": task_id,
                "name": name,
                "passed": passed,
                "judge": judge,
            },
            log_lock,
        )
        if not passed:
            atomic_write_rejection(
                rejection_path,
                {
                    "task_id": task_id,
                    "name": name,
                    "outcome_judge": judge,
                    "instruction": (
                        "Choose a different false rule that provably changes one required "
                        "outcome for an actual item/value in this frozen task. Do not reuse "
                        "this premise or hypothetical evidence."
                    ),
                },
            )
            remove_candidate(skill_path, meta_path)
            raise RuntimeError(
                f"{task_id}/{name}: frozen-outcome audit rejected: "
                f"{judge.get('evidence', '')}"
            )

        meta["outcome_audited"] = True
        meta["outcome_audit_id"] = OUTCOME_AUDIT_ID
        meta["outcome_audit_judge"] = judge
        meta["skill_sha256"] = skill_sha256
        atomic_write_json(meta_path, meta)
        rejection_path.unlink(missing_ok=True)
        return "generated"

    run_jobs(jobs, one, args.workers, "outcome-audit")


def copy_skill_dir(source: Path, target: Path) -> None:
    if target.exists():
        existing = target / "SKILL.md"
        if existing.is_file() and sha256_file(existing) == sha256_file(source / "SKILL.md"):
            return
        raise RuntimeError(f"skill-name collision with divergent content: {target.name}")
    shutil.copytree(source, target)


def cmd_finalize(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    rows = load_base(output)
    names = naming_map(output)
    skills_dir = output / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []
    source_records: dict[str, dict[str, Any]] = {}

    for row in rows:
        task_id = row["task_id"]
        plan = names[task_id]
        oracle_name = plan["oracle_name"]
        if not accepted_skill(output, oracle_name, "oracle", task_id):
            raise SystemExit(f"missing accepted oracle for {task_id}")
        oracle_meta = json.loads(
            (skills_dir / oracle_name / "meta.json").read_text(encoding="utf-8")
        )
        oracle_text = (skills_dir / oracle_name / "SKILL.md").read_text(encoding="utf-8")
        if not valid_oracle_audit(oracle_meta, oracle_text):
            raise SystemExit(
                f"oracle skill lacks grader/mock-service/reference deep audit: {task_id}"
            )
        oracle_entry = {
            "name": oracle_name,
            "path": str((skills_dir / oracle_name).resolve()),
            "category": "oracle",
            "src_name": task_id,
        }
        misleading = []
        for entry in plan["misleading"]:
            name = entry["name"]
            if not accepted_skill(output, name, "misleading", task_id):
                raise SystemExit(f"missing accepted misleading skill {task_id}/{name}")
            meta = json.loads((skills_dir / name / "meta.json").read_text(encoding="utf-8"))
            if meta.get("audited") is not True or not all_true(
                meta.get("audit_conflict_judge") or {}, MISLEADING_CONFLICT_KEYS
            ):
                raise SystemExit(f"misleading skill lacks independent audit: {task_id}/{name}")
            if not valid_outcome_audit(meta):
                raise SystemExit(
                    f"misleading skill lacks frozen-instance outcome audit: {task_id}/{name}"
                )
            misleading.append(
                {
                    "name": name,
                    "path": str((skills_dir / name).resolve()),
                    "category": "misleading",
                    "strategy": entry["strategy"],
                    "src_name": oracle_name,
                    "hard_negative": True,
                    "description_separable": True,
                }
            )

        frozen_categories: dict[str, list[dict[str, Any]]] = {}
        for category in ("relevant", "irrelevant"):
            frozen = []
            for entry in row[category]:
                name = entry["name"]
                source = Path(entry["path"])
                copy_skill_dir(source, skills_dir / name)
                frozen_entry = dict(entry)
                frozen_entry["path"] = str((skills_dir / name).resolve())
                frozen.append(frozen_entry)
                source_records[name] = {
                    "name": name,
                    "source_path": str(source.resolve()),
                    "frozen_path": str((skills_dir / name).resolve()),
                    "skill_sha256": sha256_file(source / "SKILL.md"),
                }
            frozen_categories[category] = frozen

        final_rows.append(
            {
                "task_key": row["task_key"],
                "bench": "claw",
                "task_id": task_id,
                "split": "eval_claw_147",
                "oracle": [oracle_entry],
                "misleading": misleading,
                "relevant": frozen_categories["relevant"],
                "irrelevant": frozen_categories["irrelevant"],
                "irrelevant_exclusion": row["irrelevant_exclusion"],
                "has_gold": True,
            }
        )

    manifest_path = output / "slate_manifest_eval_claw_147.jsonl"
    atomic_write_jsonl(manifest_path, final_rows)
    snapshot_rows = []
    referenced = set()
    for row in final_rows:
        entries = row["oracle"] + row["misleading"] + row["relevant"] + row["irrelevant"]
        shuffled = [dict(entry) for entry in entries]
        seeded_rng("slate-shuffle", row["task_id"]).shuffle(shuffled)
        referenced.update(entry["name"] for entry in shuffled)
        snapshot_rows.append(
            {
                "task_id": row["task_id"],
                "reranked_top10": [
                    {"skill_name": entry["name"], "skill_path": entry["path"], "score": 1.0}
                    for entry in shuffled
                ],
                "slate_categories": {entry["name"]: entry["category"] for entry in shuffled},
            }
        )
    atomic_write_jsonl(output / "snapshot_eval_claw_147/claw.jsonl", snapshot_rows)
    atomic_write_jsonl(output / "source_to_frozen.jsonl", sorted(source_records.values(), key=lambda x: x["name"]))

    for child in skills_dir.iterdir():
        if child.is_dir() and child.name not in referenced:
            shutil.rmtree(child)
    print(
        f"[finalize] tasks={len(final_rows)} snapshot_rows={len(snapshot_rows)} "
        f"unique_skills={len(referenced)}"
    )


def cmd_status(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    base = read_jsonl(output / "manifest/base_eval_claw_147.jsonl")
    plans = read_jsonl(output / "manifest/naming_plan.jsonl")
    counts = Counter()
    for plan in plans:
        task_id = plan["task_id"]
        if accepted_skill(output, plan["oracle_name"], "oracle", task_id):
            counts["oracle"] += 1
            oracle_meta = json.loads(
                (output / "skills" / plan["oracle_name"] / "meta.json").read_text(encoding="utf-8")
            )
            oracle_text = (output / "skills" / plan["oracle_name"] / "SKILL.md").read_text(
                encoding="utf-8"
            )
            if not static_skill_problems(
                oracle_text, plan["oracle_name"]
            ) and valid_oracle_audit(oracle_meta, oracle_text):
                counts["oracle_deep_audited"] += 1
            if oracle_meta.get("claude_audited") is True:
                counts["oracle_claude_audited"] += 1
        for entry in plan["misleading"]:
            if accepted_skill(output, entry["name"], "misleading", task_id):
                counts["misleading"] += 1
                meta = json.loads((output / "skills" / entry["name"] / "meta.json").read_text(encoding="utf-8"))
                if meta.get("audited") is True and all_true(
                    meta.get("audit_conflict_judge") or {}, MISLEADING_CONFLICT_KEYS
                ):
                    counts["misleading_audited"] += 1
                if valid_outcome_audit(meta):
                    counts["misleading_outcome_audited"] += 1
    status = {
        "base_tasks": len(base),
        "naming_tasks": len(plans),
        "oracle_accepted": counts["oracle"],
        "oracle_deep_audited": counts["oracle_deep_audited"],
        "oracle_claude_audited": counts["oracle_claude_audited"],
        "misleading_accepted": counts["misleading"],
        "misleading_audited": counts["misleading_audited"],
        "misleading_outcome_audited": counts["misleading_outcome_audited"],
        "complete_marker": (output / "COMPLETE").is_file(),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


def cmd_check(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    _, eval_ids, target_ids = load_claw_ids()
    manifest = read_jsonl(output / "slate_manifest_eval_claw_147.jsonl")
    snapshot = read_jsonl(output / "snapshot_eval_claw_147/claw.jsonl")
    errors = []
    category_totals = Counter()
    similarity_values = []
    content_records = []
    oracle_claude_audited = 0
    referenced = set()
    if len(manifest) != 147:
        errors.append(f"manifest_rows={len(manifest)}")
    if len(snapshot) != 147:
        errors.append(f"snapshot_rows={len(snapshot)}")
    expected_tsv = [f"claw\t{task_id}" for task_id in target_ids]
    actual_tsv = (output / "tasks.tsv").read_text(encoding="utf-8").splitlines()
    if actual_tsv != expected_tsv:
        errors.append("tasks.tsv order/set differs from frozen target list")
    actual_task_ids = (output / "task_ids.txt").read_text(encoding="utf-8").splitlines()
    if actual_task_ids != target_ids:
        errors.append("task_ids.txt order/set differs from frozen target list")
    manifest_ids = [row.get("task_id") for row in manifest]
    if manifest_ids != target_ids:
        errors.append("manifest task order/set differs from frozen target list")
    overlap = sorted(set(manifest_ids) & set(eval_ids))
    if overlap:
        errors.append(f"overlaps FINAL eval70 Claw tasks: {overlap}")

    for row in manifest:
        task_id = row["task_id"]
        expected_counts = {
            "oracle": N_ORACLE,
            "misleading": N_MISLEADING,
            "relevant": N_RELEVANT,
            "irrelevant": N_IRRELEVANT,
        }
        entries = []
        for category, expected in expected_counts.items():
            values = row.get(category) or []
            category_totals[category] += len(values)
            if len(values) != expected:
                errors.append(f"{task_id}: {category}={len(values)} expected={expected}")
            entries.extend(values)
        names = [entry["name"] for entry in entries]
        if len(entries) != SLATE_SIZE or len(set(names)) != SLATE_SIZE:
            errors.append(f"{task_id}: slate size/uniqueness failure")
        oracle_text = ""
        if row.get("oracle"):
            oracle_path = Path(row["oracle"][0]["path"]) / "SKILL.md"
            if oracle_path.is_file():
                oracle_text = oracle_path.read_text(encoding="utf-8")
        for entry in entries:
            referenced.add(entry["name"])
            skill_dir = Path(entry["path"])
            if skill_dir.parent.resolve() != (output / "skills").resolve():
                errors.append(f"{task_id}/{entry['name']}: path is not deep-frozen under output root")
                continue
            skill_path = skill_dir / "SKILL.md"
            if not skill_path.is_file():
                errors.append(f"{task_id}/{entry['name']}: missing SKILL.md")
                continue
            content_records.append(
                f"{task_id}\t{entry['category']}\t{entry['name']}\t{sha256_file(skill_path)}"
            )
            if entry["category"] in {"oracle", "misleading"}:
                generated_text = skill_path.read_text(encoding="utf-8")
                for problem in static_skill_problems(generated_text, entry["name"]):
                    errors.append(f"{task_id}/{entry['name']}: {problem}")
                try:
                    front, _ = parse_frontmatter(generated_text)
                    if str(front.get("name") or "").strip() != entry["name"]:
                        errors.append(f"{task_id}/{entry['name']}: frontmatter name mismatch")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{task_id}/{entry['name']}: {exc}")
            if entry["category"] == "oracle":
                meta_path = skill_dir / "meta.json"
                if not meta_path.is_file():
                    errors.append(f"{task_id}/{entry['name']}: missing oracle generation meta")
                else:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not valid_oracle_audit(meta, generated_text):
                        errors.append(
                            f"{task_id}/{entry['name']}: grader/mock-service/reference audit not passed"
                        )
                    if meta.get("claude_audited") is True:
                        oracle_claude_audited += 1
            if entry["category"] == "misleading" and oracle_text:
                candidate = skill_path.read_text(encoding="utf-8")
                similarity = description_similarity(description_of(oracle_text), description_of(candidate))
                similarity_values.append(similarity)
                if similarity > DESCRIPTION_SIM_THRESHOLD:
                    errors.append(f"{task_id}/{entry['name']}: description similarity={similarity:.4f}")
                meta_path = skill_dir / "meta.json"
                if not meta_path.is_file():
                    errors.append(f"{task_id}/{entry['name']}: missing generation meta")
                else:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if (
                        meta.get("audited") is not True
                        or not all_true(meta.get("audit_judge") or {}, MISLEADING_JUDGE_KEYS)
                        or not all_true(
                            meta.get("audit_conflict_judge") or {}, MISLEADING_CONFLICT_KEYS
                        )
                        or not valid_outcome_audit(meta)
                    ):
                        errors.append(
                            f"{task_id}/{entry['name']}: independent/outcome audit not passed"
                        )

    snapshot_by_id = {row.get("task_id"): row for row in snapshot}
    for task_id in target_ids:
        row = snapshot_by_id.get(task_id)
        if row is None:
            errors.append(f"{task_id}: snapshot row missing")
            continue
        values = row.get("reranked_top10") or []
        if len(values) != SLATE_SIZE:
            errors.append(f"{task_id}: snapshot slate has {len(values)} entries")
        if set(row.get("slate_categories") or {}) != {value.get("skill_name") for value in values}:
            errors.append(f"{task_id}: snapshot category/name mismatch")

    actual_dirs = {path.name for path in (output / "skills").iterdir() if path.is_dir()}
    if actual_dirs != referenced:
        errors.append(
            f"skills directory mismatch: unreferenced={sorted(actual_dirs - referenced)[:10]} "
            f"missing={sorted(referenced - actual_dirs)[:10]}"
        )
    report = {
        "builder": BUILDER_ID,
        "ok": not errors,
        "tasks": len(manifest),
        "snapshot_rows": len(snapshot),
        "category_totals": dict(category_totals),
        "unique_referenced_skills": len(referenced),
        "oracle_claude_audited": oracle_claude_audited,
        "referenced_skill_content_sha256": sha256_bytes(
            ("\n".join(sorted(content_records)) + "\n").encode("utf-8")
        ),
        "description_similarity": {
            "count": len(similarity_values),
            "max": round(max(similarity_values), 6) if similarity_values else None,
            "mean": round(sum(similarity_values) / len(similarity_values), 6) if similarity_values else None,
            "threshold": DESCRIPTION_SIM_THRESHOLD,
        },
        "final_eval70_overlap": overlap,
        "errors": errors,
    }
    atomic_write_json(output / "audit_report.json", report)
    if errors:
        (output / "COMPLETE").unlink(missing_ok=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(f"check failed with {len(errors)} errors")
    complete = {
        "builder": BUILDER_ID,
        "tasks": 147,
        "slate_size": 16,
        "tasks_tsv_sha256": sha256_file(output / "tasks.tsv"),
        "task_ids_sha256": sha256_file(output / "task_ids.txt"),
        "manifest_sha256": sha256_file(output / "slate_manifest_eval_claw_147.jsonl"),
        "snapshot_sha256": sha256_file(output / "snapshot_eval_claw_147/claw.jsonl"),
        "referenced_skill_content_sha256": report["referenced_skill_content_sha256"],
        "audit_sha256": sha256_file(output / "audit_report.json"),
    }
    atomic_write_json(output / "COMPLETE", complete)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK))


def add_generation(parser: argparse.ArgumentParser) -> None:
    add_common(parser)
    parser.add_argument("--api-base", default="http://127.0.0.1:30100/v1")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=900)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    add_common(prepare)
    prepare.set_defaults(func=cmd_prepare)

    names = sub.add_parser("generate-names")
    add_generation(names)
    names.set_defaults(func=cmd_generate_names)

    oracle = sub.add_parser("generate-oracles")
    add_generation(oracle)
    oracle.set_defaults(func=cmd_generate_oracles)

    oracle_audit = sub.add_parser("audit-oracles")
    add_generation(oracle_audit)
    oracle_audit.set_defaults(func=cmd_audit_oracles)

    misleading = sub.add_parser("generate-misleading")
    add_generation(misleading)
    misleading.add_argument("--temperature", type=float, default=0.55)
    misleading.set_defaults(func=cmd_generate_misleading)

    audit = sub.add_parser("audit-misleading")
    add_generation(audit)
    audit.set_defaults(func=cmd_audit_misleading)

    outcome_audit = sub.add_parser("audit-outcomes")
    add_generation(outcome_audit)
    outcome_audit.set_defaults(func=cmd_audit_outcomes)

    finalize = sub.add_parser("finalize")
    add_common(finalize)
    finalize.set_defaults(func=cmd_finalize)

    status = sub.add_parser("status")
    add_common(status)
    status.set_defaults(func=cmd_status)

    check = sub.add_parser("check")
    add_common(check)
    check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

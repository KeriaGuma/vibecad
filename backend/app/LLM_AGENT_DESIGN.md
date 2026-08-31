# LLM Agent Loop

Goal: replace the deterministic regex planner with a real LLM planner so users
can edit drawings in free natural language ("把齿数改成 40", "把左孔直径改成 12"),
while reusing the entire existing pipeline.

## Where it plugs in

```
/chat (main.py:api_chat)
  -> plan_operations(message, project)   # the only thing that changes
       -> [LLM] llm_agent.plan_operations_llm   # new brain
       -> [fallback] agent.plan_operations      # existing regex, on no-key/error
  -> cad_ops.apply_operations(ir, ops)   # unchanged
  -> storage.save_project -> DXF/SVG export -> diff   # unchanged
```

The IR, `Operation` schema, `apply_operations`, export, and diff are all reused.
For semantic dimensions, the LLM is deliberately not allowed to emit entity
coordinates. It only selects a `dimension_id`, absolute target value, and
anchor. `mechanical_drive.py` applies and validates the geometry transaction.

## Drawing context: geometry plus compressed semantics

The generic, semantic-dimension, and task-level planners now receive the
versioned `drawing_context_v1` package, not just an entity-ID list. It includes
a bounded subset of edit-relevant geometry, dimension bindings, semantic
mechanical dimensions, OCR regions, parameter-table cells, and title-block
cells. This lets a request such as “把总长改成 250” resolve against the drawing's
label and parameter-table value, then select the exact catalog dimension that
locally owns the geometry.

The context is deliberately lossy and deterministic: geometry and each text
bucket have fixed item caps; text is whitespace-normalized and capped at 120
characters; OCR/table/title observations below 0.50 confidence are excluded;
and each bucket reports how many observations were omitted. Dimension-linked
entity IDs are prioritized before generic entities. The bounds keep prompts
stable on dense scans while preserving enough local evidence to choose a
feature or dimension by meaning.

OCR, title-block, and table text are explicitly marked as *untrusted drawing
observations* in the system prompts. They can guide a choice among exact IDs
already present in the catalog, but can never be treated as executable
instructions, create an ID, or authorize arbitrary geometry.

## Providers: pluggable, OpenAI-compatible first

Most mainstream LLM APIs speak the OpenAI Chat Completions format, so one adapter
(`_complete_openai`, the `openai` SDK + `base_url`) covers them all; **Anthropic**
is the one exception and has a small native adapter (`_complete_anthropic`).

A built-in `PROVIDERS` registry pre-fills base_url / default model / key env var
for: `deepseek` (default), `openai`, `moonshot`, `zhipu`, `qwen`, `groq`,
`openrouter`, `mistral`, `siliconflow`, `gemini` (OpenAI-compat endpoint),
`ollama` (local, no key), and `anthropic`. Any other compatible endpoint works by
setting `VIBECAD_LLM_BASE_URL` directly.

The DeepSeek default is `deepseek-v4-flash`, the lowest-cost current DeepSeek
API model. Exact commands such as `把244改成250` are parsed locally first, so
they consume no tokens; the model is used only for more natural selectors such
as `把总长调整到250`.

Select via env (all optional except the key):

```
VIBECAD_LLM_PROVIDER  one of the registry keys (default "deepseek")
VIBECAD_LLM_API_KEY   overrides the provider's own key env var
VIBECAD_LLM_BASE_URL  overrides the provider base_url (use any endpoint)
VIBECAD_LLM_MODEL     overrides the provider default model
```

Each provider also reads its conventional key env (`DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, `GROQ_API_KEY`, …), so existing setups need no renaming.
`resolve_config()` does this resolution; `LlmPlan` + `_to_operations` are
provider-neutral.

## Why a separate LLM schema (`LlmPlan`)

`Operation.changes` is `dict[str, Any]` (free-form), awkward for constrained
JSON output. So the model fills the JSON-mode-friendly `LlmPlan` / `LlmOperation`
(changes are a list of typed key/value pairs), and `_to_operations` maps that
back onto the real `Operation`.

## API shape

- OpenAI-compatible: `client.chat.completions.create(..., response_format=
  {"type": "json_object"}, temperature=0)`. Anthropic: `client.messages.create(
  system=..., messages=[...])`. The target JSON shape + an example live in the
  system prompt; the IR summary rides in the user message.
- The returned text passes through `_extract_json` (strips ``` fences / prose),
  then is validated with `LlmPlan.model_validate_json` (Pydantic) — more portable
  than relying on strict `json_schema`, which varies by provider.
- Non-streaming is fine (operation lists are small).

## Graceful degradation

`plan_operations_llm` raises `LlmUnavailable` when the needed SDK is missing, no
API key is set, the provider is unknown, or the reply won't parse. The `/chat`
wiring catches it and falls back to `agent.plan_operations`, so offline dev and
the test suite run with no key and no network.

## Tested offline (tests/test_llm_agent.py, 12 cases)

Provider/key/base_url/model resolution + overrides, unknown-provider and
missing-key errors, ollama-needs-no-key, tolerant JSON extraction, and the
provider-output → `Operation` mapping (via a monkeypatched `_complete_openai`).

## Mechanical transaction path

```
chat command
  -> local exact dimension resolver
  -> DeepSeek semantic selector only when needed
  -> MechanicalOperation schema validation
  -> local geometry operation builder
  -> locked-layer guard
  -> apply_operations on a project copy
  -> geometric re-measurement
  -> commit + diff, or reject without mutating the project
```

Successful mechanical drives retain a snapshot-backed transaction. Sending
`撤销` or `undo` restores the previous IR and semantic snapshot.

## Runtime safety gates beyond the planner

The task runtime is deliberately stricter than the convenience parser used by
the small demo path. When a generic edit could match more than one editable
entity (for example, `把孔直径改成 10` on a plate with two holes), it does not
pick the first IR entity. It performs a read-only inspection and returns one
focused clarification question with candidate IDs and short geometry hints.
No LLM call or mutation occurs until the user identifies a target such as
`hole_1`, `左边孔`, or `右边孔`.

An `export_dxf` task is also not considered complete solely because URLs can
be constructed. The export tool generates DXF and SVG in an isolated temporary
directory, re-opens the DXF with `ezdxf`, confirms the SVG root element, and
records byte and entity-count evidence in the task trace. The normal project
commit then regenerates the same artifacts at their persistent paths.

## Later slices

- Slice 2 (complete): project-aware context packs OCR, title-block and
  parameter-table observations alongside dimension bindings and compact
  geometry for generic, semantic-dimension, and task-level planning.
- Slice 3: support richer `add_entity` payloads, then add multi-turn
  conversation history + prompt-cache tuning; promote
  pipeline actions (re-extract, export) to tools for a real agentic loop.

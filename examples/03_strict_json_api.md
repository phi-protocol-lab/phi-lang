# Example 03: Strict JSON Constraint

Eliminating chatter, markdown backticks, and apologies from API responses.

[English] | [日本語](03_strict_json_api.ja.md)

---

## Natural Language Prompt (~80 tokens)
> "Respond with pure, parseable JSON only. Do not include introductory text, explanations, or Markdown code blocks (```json). Ensure the schema is strictly valid."

---

## Φ_Lang Representation (v3.3.1)

```text
[ phi:v3.3.1 | mode:filter | out:json ]
@Constraint chat=ban markdown_fence=strip schema="strict_json"
@Pipe in=Raw -> validate:json -> Out
```

---

## Expected Model Behavior
The model strips greetings, avoids code fences, validates internal object structure, and emits raw JSON characters directly to the output buffer.
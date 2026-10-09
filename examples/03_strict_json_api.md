# Example 03: Strict JSON Output Constraint

An example demonstrating the constraints required to completely eliminate preambles, Markdown backticks, pleasantries, and unnecessary conversational chatter from API responses.

[English] | [日本語](03_strict_json_api.ja.md)

---

## Natural Language Prompt (~80 tokens)
> "Respond with pure, parseable JSON only. Do not include introductory greetings, explanations, or Markdown code blocks (```json). Ensure the schema is strictly valid."

---

## Representation in Φ_Lang (v3.3.1)

```text
[ phi:v3.3.1 | mode:filter | out:json ]
@Constraint chat=ban markdown_fence=strip schema="strict_json"
@Pipe in=Raw -> validate:json -> Out
```

---

## Expected Model Behavior (Initial Design Hypothesis)

The model was expected to strip introductory greetings, suppress code fences (backticks), validate internal object structure, and emit raw JSON characters directly to the output buffer.

### Issues Encountered in Practice (Root Causes of Breakdown)

Attempting to enforce such "draconian suppression entirely via prompting" causes syntax errors and output freezes (empty responses, parse failures) across many models:

1. **Inherent Conflict between "Deliberation / Self-Validation" and "Immediate First-Character Output"**  
   Executing the instruction to "ensure the schema is strictly valid" requires the model to have deliberation workspace or buffer space (margin to validate tokens as they are generated). However, simultaneously demanding "zero preambles or fences, emit `{` from the very first token" creates conflicting internal token probability distributions, frequently leading to malformed syntax.
2. **LLM Pre-training Distribution and Code Fence Bias**  
   LLMs are probabilistic next-token predictors, not deterministic compilers or execution environments. Across massive training corpora (Markdown and technical documentation), JSON overwhelmingly appears enclosed within ```` ```json ```` code fences. Forcibly banning fences via prompts reduces token generation confidence, making output fragile and erratic.
3. **Impossibility of Low-Level Buffer / Stripping Control**  
   LLMs have no physical execution runtime capable of directly manipulating in-memory output buffers or retroactively stripping characters. Meta-directives like `@Pipe` and `strip`, when fed to a bare model, are merely interpreted as heavily constrained text instructions rather than physical stream-filtering pipelines.

---

## Revised Φ_Lang Representation (v3.3.2)

Updated to provide the model with an **explicit starting token** and **concrete schema structure**, enabling reliable generation regardless of whether code blocks are present:

```text
[ phi:v3.3.2 | mode:transform | out:json ]
@Schema target={"status": string, "result": any}
@Constraint prefix="{" suffix="}" explain=ban markdown=ban
@Pipe in=Raw -> Out
```

---

## Expected Behavior in Revised Version

Without inserting greetings or conversational preambles, the model stably generates a valid JSON string beginning with `{` that conforms to the specified schema.

---

## Practical Best Practices (Copilot Note)

Relying solely on prompt constraints to guarantee 100% raw JSON in production is an anti-pattern. Real-world systems should combine the following:

1. **Leverage Native API Parameters (Highest Priority)**  
   - OpenAI / Local LLMs: `response_format={"type": "json_object"}` or grammar-constrained decoding (GBNF).  
   - Gemini: `response_schema` / `response_mime_type="application/json"`.

2. **Lightweight Host-Side Sanitization (Python, etc.)**  
   - Even if a model accidentally outputs ```` ```json ```` fences, designing the receiving application to strip fences via regex before passing the payload to `json.loads()` remains the most robust architecture.

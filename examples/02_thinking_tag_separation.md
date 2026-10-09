# Example 02: Thinking Scratchpad Isolation

Compressing a complex multi-sentence system prompt into a 3-line structural rule.

[English] | [日本語](02_thinking_tag_separation.ja.md)

---

## Natural Language Prompt (~150 tokens)
> "Before generating your response, output your internal deliberation and strategy at the very beginning wrapped in `[THINK]...[/THINK]`. Content inside the tag is private scratchpad and must not appear in the final public channel. Immediately follow the closing tag with your public response."

---

## Φ_Lang Representation (v3.3.1)

```text
[ phi:v3.3.1 | flow:split | out:dual ]
@Route split=tag("THINK") target=(priv, pub)
@Format priv=enc("[THINK]","[/THINK]") mode=scratchpad p=1
@Format pub=stream mode=emit > Out
```

---

## How LLMs Interpret It
- `@Route`: Splits generation into internal scratchpad (`priv`) and external message (`pub`).
- `@Format priv`: Enforces tag boundaries and assigns execution priority 1 (`p=1`).
- `@Format pub`: Pipes raw stream straight to the public response stream.
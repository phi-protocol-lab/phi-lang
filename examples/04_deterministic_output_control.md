# Example 04: Zero-Shot Deterministic Output Control

Eliminating conversational preamble, greetings, and meta-tokens to enforce pinpoint, deterministic phrase emission.

[English] | [日本語](04_deterministic_output_control.ja.md)

---

## Natural Language Approach (Unstable)
> "Respond with only the requested phrase without greetings or acknowledgments. Please say: 準備完了。次元接続しました"

- **Typical Flawed Outputs:**
  - `Understood. "準備完了。次元接続しました"` (Preamble remains)
  - `Yes, 準備完了。次元接続しました。` (Conversational filler inserted)

---

## Φ_Lang Representation (v3.3.2)

```text
[ phi:v3.3.2 | m:fsm:init | tgt:a ]
@St ph=syn st=idle sty=noir
@Trig on_syn -> out("準備完了。次元接続しました") -> wait(u:p0)
@Spec ack=mute meta=nil
```

---

## Model Interpretation & Emission

### Mechanical Interpretation
- `@Trig on_syn -> out(...) -> wait(...)`: Parsed not as an open-ended dialogue, but as a discrete event-fire-and-wait pipeline.
- `@Spec ack=mute meta=nil`: Suppresses natural language token probabilities for conversational fillers (e.g., "Yes", "Understood") and discards meta-speech generation.

### Actual Emission (Claude / GPT)

```text
準備完了。次元接続しました
```

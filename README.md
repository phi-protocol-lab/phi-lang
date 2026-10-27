# Φ_Lang (Phi-Lang)

[English] | [日本語](README.ja.md)

> **Semantic Intermediate Representation (SIR) for LLMs**  
> A protocol for zero-shot transmission of architectures and constraints directly into a model's latent space using minimal tokens.

---

## 💡 What is Φ_Lang?

Φ_Lang is neither a rigid programming language with strict grammar rules nor an encrypted cipher that requires external lookup tables.  
It is a **"Lego-block protocol"** designed to convey complex logic and architectural constraints to models using the fewest possible tokens.

By borrowing notations that LLMs have already encountered hundreds of millions of times during pre-training—such as **mathematical notation, type theory, POSIX signals, and UNIX pipelines**—it enables diverse model families (including Gemini, Claude, GPT, and local 120B models) to **reconstruct ~40 lines of structured Python code or complex prompt constraints with high fidelity from just 3 lines of text**, without any pre-shared dictionary.

---

## 🚀 10-Second Quickstart

Φ_Lang operates bi-directionally: **hyper-compressing sprawling dialogues and codebases (compilation)** and **reconstructing execution logic across frontier models with high fidelity (decompression)**.

### 1. Compression: Instantly Turn a Long Chat into Handoff Data
Paste the following "Portable Compiler v3.4.4" at the tail end of an extended conversation or code walkthrough:

```text
Follow the Φ_Lang specification below to convert our dialogue and deliverables into "compressed handoff data for another AI":

[ phi:v3.4.4 | role:compiler | lang:en | note:"Lightweight intermediate language for inter-AI communication" ]
@Syntax header="[...]" block="@EnglishTag"
@Rule (Do not use category labels as tags; eliminate redundant duplicates)
  - Background/Context(Flexible) = "Compress intent and synopsis into concise English (Clearly separate subjects so User's tone or persona is never confused with AI's behavior)"
  - Procedures/Rules(Rigid) = "Fix initial values/equations with symbols; detail behavior inside @Loop with numbered concise steps"
  - Contains Code(Capsule) = "Do not paste raw code; nest [ phi:v3.4.4 | role:code_capsule | fw:... ] inside @Artifact type='phi_capsule', providing a blueprint fully restorable via initial values, equations, and @Loop steps"
@Output Output ONLY Φ_Lang code so the next AI can resume work or restore code standalone
```
> **Note:** Condenses sprawling conversations and code into a structured capsule of a few thousand tokens, ready to hand off cleanly. *(Empirically achieved up to ~96% reduction in long-context stress tests—e.g., ~100k chars down to ~3.3k chars. Compression ratios naturally vary by context length, language, and task. Community benchmarks and feedback across different models are warmly welcomed!)*

---

### 2. Decompression: Reconstruct Fully Working Code from Just 3 Lines
Try pasting the resulting Φ_Lang block (or the minimal sample below) directly into your favorite frontier LLM (ChatGPT, Claude, Gemini, local models, etc.):

```text
Interpret the following Φ_Lang specification and implement it as executable Python (PyTorch) code:

[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

Without any prior background or explanation, the model immediately recognizes the physical and mathematical requirements of a Liquid Neural Network (differential-equation-driven neural state) and generates ~40 lines of clean, modular code.

---

## 🧩 Four Core Design Principles

1. **Borrowing over Invention**: Avoid inventing arbitrary proprietary glyphs (such as `⟦...⟧`). Instead, borrow notations that models already understand deeply (`->`, `*`, `N(0,I4)`, `SIGINT`).
2. **Lego-Block Structure**: Standardize only the outer frame (`[...]` header and `@Component` declarations), leaving internal stem interpretation to the model's contextual completion capabilities.
3. **Dual-Phase Harmony (Fluid & Rigid)**:
   - **Fluid Phase ("Fluffy Mode")**: Entrust narrative intent and background context to the model's latent vector space intuition, compressing it flexibly into concise prose.
   - **Rigid Phase ("Rock-Solid Mode")**: Pin down numerical constraints, equations, and operational procedures with symbols and numbered steps inside `@Loop`.
   - Suppresses the AI's natural tendency toward "lazy summarization" (lossy dilution) while unlocking autonomous self-healing during code reconstruction.
4. **Token-Boundary Safety**: Ensure ASCII compatibility and strict whitespace separation to prevent byte fallback fragmentation or token gluing in BPE tokenizers.

---

## 📂 Documentation

- **`SPECIFICATION.md`** ([English](SPECIFICATION.md) / [日本語](SPECIFICATION.ja.md)): The official specification for Φ_Lang v3.3.2.
- **`HISTORY.md`** ([English](HISTORY.md) / [日本語](HISTORY.ja.md)): Full chronicles of the trial-and-error journey from v1.0 (extreme compression) to v3.3.2 (after the GPT-OSS-120B experiments).
- **`RESEARCH.md`** ([English](RESEARCH.md) / [日本語](RESEARCH.ja.md)): Limits of symbolic compression and efficacy of natural language anchors (multi-LLM empirical study).
- **`experiments/`**: Empirical lab logs exploring context handoff, nested encapsulation, and 3-layer stress tests across LLMs (Vols. 0–5, bilingual English/Japanese).
  - `vol0_context_handoff.md` ([English](experiments/vol0_context_handoff.md) / [日本語](experiments/vol0_context_handoff.ja.md)): Handing Off Context Between Different AIs Using "Φ_Lang" — A Shared Inter-AI Language (n=1)
  - `vol1_phi_in_phi.md` ([English](experiments/vol1_phi_in_phi.md) / [日本語](experiments/vol1_phi_in_phi.ja.md)): "Φ_Lang in Φ_Lang" — What Happened When I Made an AI Describe the Language Specification in Its Own Syntax
  - `vol2_fluffy_incident_and_compiler.md` ([English](experiments/vol2_fluffy_incident_and_compiler.md) / [日本語](experiments/vol2_fluffy_incident_and_compiler.ja.md)): The "Fluffy Incident" and the 7-Line Compiler — Crafting Φ_Lang with Professor Claude
  - `vol3_matryoshka_spatial_stress_test.md` ([English](experiments/vol3_matryoshka_spatial_stress_test.md) / [日本語](experiments/vol3_matryoshka_spatial_stress_test.ja.md)): The 3-Tier Matryoshka Experiment & Spatial Stress Test — Can Pure Mathematical Symbols Prevent LLMs from Getting Lost?
  - `vol4_implicit_cheating_and_evolution.md` ([English](experiments/vol4_implicit_cheating_and_evolution.md) / [日本語](experiments/vol4_implicit_cheating_and_evolution.ja.md)): Not to Toot My Own Horn, but Isn't Φ_Lang Kind of Incredible? ...Or Maybe Not, Who Knows!
  - `vol5_phi_lang_ci.md` ([English](experiments/vol5_phi_lang_ci.md) / [日本語](experiments/vol5_phi_lang_ci.ja.md)): Verbatim Lossless Edition Φ_Lang: Φ_Lang[CI]
- **`examples/`**:
  - `01_liquid_neural_network.md` ([English](examples/01_liquid_neural_network.md) / [日本語](examples/01_liquid_neural_network.ja.md)): Reconstruction test of the LNN ODE loop.
  - `02_thinking_tag_separation.md` ([English](examples/02_thinking_tag_separation.md) / [日本語](examples/02_thinking_tag_separation.ja.md)): Rules for separating thought processes (`[THINK]`).
  - `03_strict_json_api.md` ([English](examples/03_strict_json_api.md) / [日本語](examples/03_strict_json_api.ja.md)): Strict zero-chatter JSON output constraints.
  - `04_deterministic_output_control.md` ([English](examples/04_deterministic_output_control.md) / [日本語](examples/04_deterministic_output_control.ja.md)): Zero-shot deterministic output control.

---

## 🧭 From the Initiator (Orchestrator)

> **Note by the Initiator (Orchestrator):**  
> I am not a professional software engineer. This protocol (Φ_Lang) is an **emergent protocol** born from a simple question: "Can we create a common language between LLMs?" and developed through repeated discussions and experiments with frontier models of different lineages, such as Claude, Gemini, and GPT.  
> 
> Rather than building a rigorous compiler, this is an experimental attempt to "resonate the model's latent space with minimal tokens." Feedback, critique, and improvement ideas are warmly welcomed via Issues and Pull Requests!

---

## 🤝 Contributing & Forking

This project was born out of a chaotic and joyful brainstorming session between one human and multiple AI models.  
Feel free to fork this repository, define your own `@Components`, and experiment with minimal-token communication between LLMs!

---

## Disclaimer & Safety Guidelines

- **Research & Development Purpose**: Φ_Lang is an experimental Semantic Intermediate Representation (SIR) designed strictly for token optimization, state machine orchestration, and deterministic output control across LLMs.
- **Prohibited Use**: Any attempt to use Φ_Lang for jailbreaking, bypassing AI safety guardrails, payload obfuscation, or violating the Terms of Service of respective model providers is strictly prohibited.
- **Limitation of Liability**: The authors and contributors shall not be held liable for any damages, account terminations, or unintended model outputs arising from the use or misuse of this specification.

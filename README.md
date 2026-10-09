# Φ_Lang (Phi-Lang)

[English] | [日本語](README.ja.md)

> **Semantic Intermediate Representation (SIR) for LLMs**  
> A protocol for zero-shot transmission of architectures and constraints directly into a model's latent space using minimal tokens.

---

## 💡 What is Φ_Lang?

Φ_Lang is neither a rigid programming language with strict grammar rules nor an encrypted cipher that requires external lookup tables.  
It is a **"Lego-block protocol"** designed to convey complex logic and architectural constraints to models using the fewest possible tokens.

By borrowing notations that LLMs have already encountered hundreds of millions of times during pre-training—such as **mathematical notation, type theory, POSIX signals, and UNIX pipelines**—it enables diverse model families (including Gemini, Claude, GPT, and local 120B models) to **reconstruct ~70 lines of structured Python code or complex prompt constraints with high fidelity from just 2 lines of text**, without any pre-shared dictionary.

---

## 🚀 10-Second Quickstart

Try pasting the following prompt directly into your favorite frontier LLM (ChatGPT, Claude, Gemini, local models, etc.):

```text
Interpret the following Φ_Lang specification and implement it as executable Python (PyTorch) code:

[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

Without any prior background or explanation, the model will immediately recognize the physical and architectural requirements of a Liquid Neural Network (differential-equation-driven neural state) and generate clean, modular code.

---

## 🧩 Three Core Design Principles

1. **Borrowing over Invention**: Avoid inventing arbitrary proprietary glyphs (such as `⟦...⟧`). Instead, borrow notations that models already understand deeply (`->`, `*`, `N(0,I4)`, `SIGINT`).
2. **Lego-Block Structure**: Standardize only the outer frame (`[...]` header and `@Component` declarations), leaving internal stem interpretation to the model's contextual completion capabilities.
3. **Token-Boundary Safety**: Ensure ASCII compatibility and strict whitespace separation to prevent byte fallback fragmentation or token gluing in BPE tokenizers.

---

## 📂 Documentation

- **`SPECIFICATION.md`** ([English](SPECIFICATION.md) / [日本語](SPECIFICATION.ja.md)): The official specification for Φ_Lang v3.3.2.
- **`HISTORY.md`** ([English](HISTORY.md) / [日本語](HISTORY.ja.md)): Full chronicles of the trial-and-error journey from v1.0 (extreme compression) to v3.3.2 (after the GPT-OSS-120B experiments).
- **`examples/`**:
  - `01_liquid_neural_network.md` ([English](examples/01_liquid_neural_network.md) / [日本語](examples/01_liquid_neural_network.ja.md)): Reconstruction test of the LNN ODE loop.
  - `02_thinking_tag_separation.md` ([English](examples/02_thinking_tag_separation.md) / [日本語](examples/02_thinking_tag_separation.ja.md)): Rules for separating thought processes (`[THINK]`).
  - `03_strict_json_api.md` ([English](examples/03_strict_json_api.md) / [日本語](examples/03_strict_json_api.ja.md)): Strict zero-chatter JSON output constraints.

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

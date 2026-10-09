# Φ_Lang v3.3.2 Specification
**Semantic Intermediate Representation (SIR) for Large Language Models**

[English] | [日本語](SPECIFICATION.ja.md)

---

## 1. Core Principles

### 1.1 Borrowing over Invention
Instead of creating novel symbolic systems, Φ_Lang borrows syntax already deeply consolidated within LLM training corpora: **mathematical equations, type hints, POSIX signals, and UNIX pipes**. This evokes zero-shot attention directly within the model's latent space without external dictionary tables.

### 1.2 Semantic IR vs. Syntax Parsing
Φ_Lang is not designed for deterministic AST compilers or Regex engines. It is an **Intermediate Representation designed to resonate with latent representations**. By enforcing an outer topological frame while leaving attribute stems to semantic interpolation, it achieves extreme token density with zero ambiguity.

### 1.3 Token-Boundary Safe Design
To avoid byte-fallback and accidental token merges common in BPE (Byte-Pair Encoding) tokenizers, Φ_Lang standardizes on **ASCII-compatible stems and deliberate whitespace isolation around structural delimiters**.

---

## 2. Grammar Specification

A valid Φ_Lang document consists strictly of **[1. Header Declaration]** followed by **[2. Component Blocks]**.

```text
[ phi:v3.3.2 | <domain>:<spec> | <param>:<val> ... ]
@<Component> <key>=<value> <key>=(<sub_expressions>) ...
@<Component> <key>=<value> ...
```

### 2.1 Header: `[ ... ]`
Declares the execution context, solver, or global mode. Spaces inside outer brackets and surrounding pipe delimiters ` | ` are strongly recommended for token boundary isolation.

* **Identifier**: `phi:v3.3.2` (ASCII standard).
* **Common Directives**:
  * `mode / flow`: Pipeline topology (e.g., `mode:ode:euler`, `flow:split`).
  * `fw / backend`: Target runtime (e.g., `fw:torch`, `backend:jax`).
  * `out`: Emission constraints (e.g., `out:dual`, `out:strict_json`).

### 2.2 Component Declaration: `@<Name>`
Defines an atomic logical or operational block (Lego block).

* **Attributes**: `key=value` separated by whitespace.
* **Compound Expressions**: Grouped in parentheses `(...)` (e.g., `dim=(4->8)`).
* **Explicit Operator Math Rules**: Mathematical formulas quoted in `ode="..."` **MUST explicitly declare multiplication operators (`*`)** to avoid ambiguous token merges (e.g., use `(W*x) * dt`, never `(Wx)dt`).
* **Pipelines**: `->` indicates state or data routing.

---

## 3. Token & Syntax Reference

| Syntax | Semantic Purpose | Note |
| :--- | :--- | :--- |
| `[ phi:vX.Y \| ... ]` | Global context initiation | Protected by whitespace |
| `@Name key=val` | Component declaration | High-density semantic stems |
| `in -> out` | Data flow pipeline | UNIX pipe paradigm |
| `(A->B)` | Dimensional projection ($A \to B$) | Activates type-hint priors |
| `ode="... * dt"` | ODE differential integration | Explicit operator requirement |
| `abs\|param` | Composite constraint (non-negative parameter) | Property decoration |
| `step%N:expr` | Modulo sampling / decimation | Monitoring interval |
| `SIGINT` | Interrupt termination | Binds to POSIX signal handling |

---

## 4. Canonical Examples

### Example 1: Liquid Neural Network (Real-time ODE Loop)
```text
[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

### Example 2: Reasoning Scratchpad Isolation
```text
[ phi:v3.3.1 | flow:split | out:dual ]
@Route split=tag("THINK") target=(priv, pub)
@Format priv=enc("[THINK]","[/THINK]") mode=scratchpad p=1
@Format pub=stream mode=emit > Out
```

### Example 3: Strict JSON Constraint
```text
[ phi:v3.3.1 | mode:filter | out:json ]
@Constraint chat=ban markdown_fence=strip schema="strict_json"
@Pipe in=Raw -> validate:json -> Out
```

---

## 5. Implementation & Parsing Guidelines

1. **System Prompt Integration**  
   Append a single directive to an LLM: *"Interpret the semantic space of the provided `[ phi:v3.3.2 ]` specification and generate runnable code or constraint-compliant text accordingly."* The model will zero-shot execute without pre-prompts.
2. **Inter-Agent Communication Layer**  
   In multi-agent collaborative frameworks, adopt Φ_Lang as the specification exchange protocol between agents. This drastically slashes token overhead compared to conventional natural language task instructions.
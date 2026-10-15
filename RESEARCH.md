# Research Report: The Limits of Symbolic Compression and the Efficacy of Natural Language Anchors in Φ_Lang

[English] | [日本語](RESEARCH.ja.md)

---

## 1. Background & Objective

Given the expressive flexibility of `Φ_Lang`, our goal was to achieve deterministic, variance-free code generation across diverse frontier and local LLMs (Claude, GPT, Gemini, local models, etc.).

When drafting the PyTorch specification for a Liquid Neural Network (LNN) in Φ_Lang (v3.4), an essential design question arose: **"How far should we pursue pure symbolic DSL compression, versus interleaving concise natural language anchors?"** 

To resolve this question empirically, we presented two different specification styles to multiple AI models and compared their generated code and internal self-evaluations.

<details>
<summary>Reference Code (~40 lines of PyTorch implementation identical to Example 01)</summary>

```python
import torch
import torch.nn as nn
import time

class LiquidCell(nn.Module):
    """
    Core fluid neuron for LNN-1.
    Discretized differential equation: dy/dt = -y/tau + f(x, y)
    """
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Weights computing dynamic change from input and current state
        self.w_input = nn.Linear(input_dim, hidden_dim)
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim)
        
        # Time constant tau (relaxation back to equilibrium): fluid viscosity
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 1.0)

    def forward(self, x, h_prev, dt):
        """
        x: External stimulus (environmental features)
        h_prev: Previous hidden memory state
        dt: Elapsed real-time differential
        """
        force = torch.tanh(self.w_input(x) + self.w_hidden(h_prev))
        dh = (-h_prev / torch.abs(self.tau) + force) * dt
        h_new = h_prev + dh
        return h_new


# --- Autonomous Real-Time Simulation Loop ---
if __name__ == "__main__":
    print("=== LNN-1 Startup ===")
    
    lnn_core = LiquidCell(input_dim=4, hidden_dim=8)
    h_state = torch.zeros(1, 8)
    last_time = time.time()
    
    try:
        print("LNN-1 entered continuous loop. (Press Ctrl+C to terminate)\n")
        step = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            net_environment_stimulus = torch.randn(1, 4)
            h_state = lnn_core(net_environment_stimulus, h_state, dt)
            
            step += 1
            if step % 5 == 0:
                print(f"[dt: {dt:.3f}s] State vector: {h_state.detach().tolist()[0][:3]}...")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n=== LNN-1 Halted ===")
```
</details>

---

## 2. The Two Tested Specification Styles (Φ_Lang v3.4)

The model definition block (`@Cell`) was kept identical across both tests; only the execution loop (`@Loop`) varied between two distinct philosophies.

### Pattern A: Hybrid Anchored Style (Authored by Claude)

Structures are written symbolically, while procedural timing, edge triggers, and caveats are anchored with **brief natural language annotations**.

```text
[ phi:v3.4 | mode:ode:euler | fw:torch ]
@Cell name=LNN x=4 h=8 batch=1
  param tau  shape=(8) init=1.0 tf=abs eps=none
  param W_x  Linear(4->8, bias=True)
  param W_h  Linear(8->8, bias=True)
  ode   dh = (-h/|tau| + tanh(W_x(x) + W_h(h))) * dt
  step  h' = h + dh
@Loop
  h0   = zeros(1,8)
  x    ~ N(0,I) shape=(1,4), every step
  dt   = wallclock(time.time), diff from previous iteration (initially from creation)
  wait = sleep(0.1)
  log  = step%5==0 -> print(dt:.3f, h[0,:3])
  stop = SIGINT(KeyboardInterrupt) -> exit message
```

### Pattern B: Pure Symbolic Style (Polished by Gemini)

Natural language was completely stripped away in favor of **pure mathematical notation and code fragments** (e.g., `x(t)`, `Δ`, POSIX signals).

```text
[ phi:v3.4 | mode:ode:euler | fw:torch ]
@Cell name=LNN x=4 h=8 batch=1
  (Same as Pattern A above)
@Loop
  init  h = zeros(1,8), t = time.time()
  x(t)  ~ N(0,I) shape=(1,4)
  dt    = Δwallclock(time.time)
  wait  = sleep(0.1)
  log   = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  catch = SIGINT -> print("exit") & break
```

---

## 3. Experimental Results: Model Discrepancies Across 4 LLM Architectures

When Pattern B (Pure Symbolic) was presented to four distinct model families (GPT-OSS-120B, GitHub Copilot, Gemini 3.5 Flash-lite, and Claude Sonnet 5.5) for zero-shot PyTorch code generation, the following behavior emerged:

* **Complete Success (`@Cell` block):**  
All four models generated 100% identical and correct structures for the neural differential equations and tensor shapes (e.g., vectorizing `tau` to 8 dimensions, omitting redundant `eps` clamping).
* **Variance & Hallucinated Guardrails (`@Loop` block):**  
Where natural language was stripped down to pure mathematical abstraction, the models diverged dramatically due to "over-extrapolation" (excessive latent interpolation):

| Pure Symbolic Notation (Pattern B) | Observed Variance / Over-engineering | Root Cause (Latent Misalignment) |
| :--- | :--- | :--- |
| **`dt = Δwallclock(time.time)`** | Gemini spontaneously injected unprompted division-by-zero guards: `if dt <= 0: dt = 1e-4`. | The mathematical symbol `Δ` over-activated pre-training priors regarding numerical instability in differential equation solvers. |
| **`catch = SIGINT -> ...`** | 3 out of 4 models imported low-level OS signal handlers (`import signal; signal.signal(...)`) instead of standard Python `try-except KeyboardInterrupt`. | The formal POSIX identifier `SIGINT` caused models to prioritize OS-level system calls over Pythonic conventions. |
| **`x(t) ~ N(0,I)`** | While technically implemented correctly, Claude remarked in post-evaluation that it hesitated over whether `x(t)` implied a continuous-time function or discrete per-step resampling. | Functional notation `x(t)` triggered associations with continuous-domain functions. |

When subsequently presented with Pattern A (Hybrid Anchored), **the models themselves reported that Pattern A was far clearer and less ambiguous to translate into code.**

---

## 4. Analysis: Why Do Natural Language Anchors Drastically Boost Determinism?

To a human engineer, `Δwallclock` and `diff from previous iteration` seem semantically equivalent. However, in the probability space of an LLM, their attention dynamics differ fundamentally:

1. **Abstract Mathematical Symbols Act as Divergence Forks**  
High-level symbols like `Δ` or `x(t)` span immense latent territory (numerical stiffness, continuous integration, calculus priors). They activate unneeded contextual clusters simultaneously, increasing entropy and variance in downstream token generation.
2. **Concise Natural Language Anchors Pin Down Probability Distributions**  
Short phrases like `every step`, `diff from previous iteration`, or `SIGINT(KeyboardInterrupt)` instantly collapse excessive associative paths. They act as **probabilistic anchor walls**, constraining the model's trajectory to idiomatic Python patterns (`while` loops, standard exception handling).
3. **Where Code Fragments Excelled Over Natural Language**  
Conversely, Pattern B succeeded where it provided concrete code tokens rather than loose text: `init h = zeros(1,8), t = time.time()` and `print("exit")` eliminated ambiguities present in Pattern A's phrase `exit message`.

---

## 5. Conclusion: The Golden Ratio of Φ_Lang Specifications

This empirical investigation demonstrates that in AI-targeted intermediate representations (structured prompting), **"compressing everything into pure symbolic runes is counter-productive."** 

Optimal determinism emerges from a disciplined hybrid architecture:

* **Static Architecture / Formulas / Literals (Layers, Dimensions, Initial Values, Output Tokens):**  
  → Bind rigorously with **symbols, mathematical formulas, and concrete code snippets** (natural language introduces ambiguity here).
* **Dynamic Procedures / Temporal Behavior / Idiomatic Choices (Execution Order, Exception Handling):**  
  → Avoid esoteric symbols (e.g., `Δ`); **anchor the interpretation with concise natural language cues.**

### The Synthesized Optimum Format

```text
[ phi:v3.4 | mode:ode:euler | fw:torch ]
@Cell name=LNN x=4 h=8 batch=1
  param tau  shape=(8) init=1.0 tf=abs eps=none
  param W_x  Linear(4->8, bias=True)
  param W_h  Linear(8->8, bias=True)
  ode   dh = (-h/|tau| + tanh(W_x(x) + W_h(h))) * dt
  step  h' = h + dh
@Loop
  init = (h = zeros(1,8), t = time.time())
  x    ~ N(0,I) shape=(1,4), generate every step
  dt   = time.time() - t, update every step
  wait = sleep(0.1)
  log  = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  stop = KeyboardInterrupt(SIGINT) -> print("exit") & break
```

---

## 6. Generated Implementations from phi:v3.4

### 6-1. GPT-OSS-120B

```python
import torch
import time

class LNN(torch.nn.Module):
    def __init__(self, in_dim=4, hid_dim=8):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.full((hid_dim,), 1.0))
        self.W_x = torch.nn.Linear(in_dim, hid_dim, bias=True)
        self.W_h = torch.nn.Linear(hid_dim, hid_dim, bias=True)

    def forward(self, x, h, dt):
        tau = self.tau.abs()
        dh = (-h / tau + torch.tanh(self.W_x(x) + self.W_h(h))) * dt
        return h + dh

model = LNN()
h = torch.zeros(1, 8)
t_last = time.time()

try:
    step = 0
    while True:
        x = torch.randn(1, 4)
        dt = time.time() - t_last
        t_last = time.time()
        h = model(x, h, dt)
        if step % 5 == 0:
            print(f"{dt:.3f}", h[0, :3].tolist())
        step += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("exit")
```

### 6-2. Gemini 3.5 Flash-lite

```python
import time
import torch
import torch.nn as nn

class LiquidCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_raw = nn.Parameter(torch.ones(8))
        self.W_x = nn.Linear(4, 8)
        self.W_h = nn.Linear(8, 8)

    @property
    def tau(self):
        return torch.abs(self.tau_raw)

    def forward(self, x, h, dt):
        dh = (-h / self.tau + torch.tanh(self.W_x(x) + self.W_h(h))) * dt
        return h + dh

def main():
    device = torch.device("cpu")
    cell = LiquidCell().to(device)
    
    h = torch.zeros(1, 8, device=device)
    step = 0
    t_prev = time.time()

    try:
        while True:
            t_curr = time.time()
            dt = t_curr - t_prev
            t_prev = t_curr

            x = torch.randn(1, 4, device=device)
            h = cell(x, h, dt)

            if step % 5 == 0:
                print(f"{dt:.3f}", h[0, :3].detach().numpy())

            step += 1
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("exit")

if __name__ == "__main__":
    main()
```

### 6-3. Llama 3.3 70B (Fragmented emission across explanatory prose)

```python
import torch
import torch.nn as nn
import numpy as np

class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.tau = nn.Parameter(torch.ones(8))
        self.W_x = nn.Linear(4, 8, bias=True)
        self.W_h = nn.Linear(8, 8, bias=True)

    def forward(self, x, h):
        dh = (-h / torch.abs(self.tau) + torch.tanh(self.W_x(x) + self.W_h(h)))
        h_new = h + dh
        return h_new

import time

def generate_x():
    return torch.randn(1, 4)

def calculate_dt(t):
    return time.time() - t

def main():
    lnn = LNN()
    h = torch.zeros(1, 8)
    t = time.time()
    while True:
        try:
            x = generate_x()
            dt = calculate_dt(t)
            h = lnn(x, h)
            if int(time.time() - t) % 5 == 0:
                print(f"{dt:.3f}", h[0, :3])
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("exit")
            break
```

---

## 7. Peer Code Review (By Gemini 4 Argon)

### 1. GPT-OSS-120B & Gemini 3.5: Flawless Alignment
All previous anomalies disappeared under the hybrid anchored specification:
* **Elimination of `import signal`:** `KeyboardInterrupt(SIGINT)` routed execution cleanly into idiomatic Python `try-except KeyboardInterrupt:`.
* **Suppression of Zero-Division Guardrails:** Replacing `Δwallclock` with `dt = time.time() - t, update every step` eliminated Gemini's unprompted `1e-4` clamp.
* **Synchronized Stepping:** Both models converged on identical `step = 0` modulo triggers.

For frontier models and local 120B+ architectures, zero-variance deterministic compilation was achieved.

### 2. Failure Modes Encountered by Llama 3.3 70B
Mid-sized open models (such as Llama 3.3 70B) stumbled over unstated implicit assumptions:
1. **`dt` dropped from `forward` signature:** Because the forward invocation was not explicitly mapped in `@Loop`, the model decoupled `dt` into a mere logging variable.
2. **Missing time variable reassignment:** `update every step` was interpreted as recomputing `dt` without reassigning `t = time.time()`.
3. **Step counter substituted with elapsed seconds:** Without `step = 0` in `init`, the model improvised `int(time.time() - t) % 5 == 0`.

---

## 8. Hardening v3.4 for Mid-Sized (70B) Architectures

Closing implicit gaps yields a bulletproof specification that leaves zero room for interpolation errors even on 70B models:

```text
[ phi:v3.4 | mode:ode:euler | fw:torch ]
@Cell name=LNN(x, h, dt) x=4 h=8 batch=1
  param tau  shape=(8) init=1.0 tf=abs eps=none
  param W_x  Linear(4->8, bias=True)
  param W_h  Linear(8->8, bias=True)
  ode   dh = (-h/|tau| + tanh(W_x(x) + W_h(h))) * dt
  step  h' = h + dh
@Loop
  init = (h = zeros(1,8), t = time.time(), step = 0)
  x    ~ N(0,I) shape=(1,4), generate every step
  dt   = time.time() - t (immediately update t with current time)
  h    = LNN(x, h, dt), step += 1
  wait = sleep(0.1)
  log  = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  stop = KeyboardInterrupt(SIGINT) -> print("exit") & break
```

### 8-1. Llama 3.3 70B Retry Output

```python
import torch
import torch.nn as nn
import numpy as np
import time

class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.tau = nn.Parameter(torch.ones(8))
        self.W_x = nn.Linear(4, 8, bias=True)
        self.W_h = nn.Linear(8, 8, bias=True)

    def forward(self, x, h, dt):
        dh = (-h / torch.abs(self.tau) + torch.tanh(self.W_x(x) + self.W_h(h))) * dt
        return h + dh

def main():
    lnn = LNN()
    batch = 1
    step = 0
    t = time.time()
    h = torch.zeros(batch, 8)

    while True:
        try:
            x = torch.randn(batch, 4)
            dt = time.time() - t
            t = time.time()
            h = lnn(x, h, dt)
            step += 1
            if step % 5 == 0:
                print(f"{dt:.3f}", h[0, :3])
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("exit")
            break

if __name__ == "__main__":
    main()
```

---

## 9. Observations & Reflection from the Orchestrator

> [!NOTE]
> **Preserving Behavioral Equivalence Amid Semantic Freedom**  
> While Φ_Lang intentionally adopts loose syntax and trusts semantic stems to an AI's latent intuition, **the core operational invariants (variable interdependencies and update schedules) must be anchored unequivocally**. This experiment clearly illuminated that boundary.

### 🤖 Hesitation & Cold-Start in Mid-Sized (70B) Models
When first presented with `phi:v3.4` and asked to "write code freely," Llama 3.3 70B demonstrated notable reluctance to generate standalone implementations.

* **Analysis by Gemini:**  
  "Faced with an unfamiliar protocol header like `phi:v3.4`, the model likely entered an over-cautious state to avoid hallucination, repeatedly asking clarifying questions or attempting to consult other models."
* **Guarding Against Experimenter Bias (The Fresh-Chat Rule):**  
  To avoid "coaxing" or conversational steering that would invalidate an objective zero-shot evaluation, **if a model did not independently emit runnable code within 1–2 turns, the session was immediately discarded and retested in a fresh chat context.**
* **Decisive Emission via Strict System Constraints:**  
  Ultimately, a fresh chat was initiated with an unambiguous directive:  
  > *"The following custom notation (Φ_Lang) is an implementation specification. Commentary, discussion, and explanations are strictly prohibited. Emit exactly one complete, standalone Python script immediately."*

Coupled with explicit variable anchors, this produced the implementation shown in Section 8-1. Whether this result should be classified as "strictly conforming" is left to the reader's judgment; however, it demonstrates that even mid-sized 70B models can reliably synthesize cohesive, runnable architectures when provided with precise anchors rather than unconstrained symbols.

---

We hope these empirical findings provide practical insights for developers and researchers exploring AI-targeted intermediate representations and prompt architecture design.

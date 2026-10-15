# Example 01: Liquid Neural Network (LNN) ODE Loop

A verification log demonstrating how large language models from different lineages reconstruct a real-time ordinary differential equation (ODE) solver zero-shot from just 2 lines (+ header) of Φ_Lang description.

[English] | [日本語](01_liquid_neural_network.ja.md)

---

## 1. Description in Φ_Lang (v3.3.1)

```text
[ phi:v3.3.1 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

---

## 2. Original Python Implementation (~40 lines of PyTorch)

<details>
<summary>Click to view the original source code</summary>

```python
import torch
import torch.nn as nn
import time

class LiquidCell(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_input = nn.Linear(input_dim, hidden_dim)
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 1.0)

    def forward(self, x, h_prev, dt):
        force = torch.tanh(self.w_input(x) + self.w_hidden(h_prev))
        dh = (-h_prev / torch.abs(self.tau) + force) * dt
        h_new = h_prev + dh
        return h_new

if __name__ == "__main__":
    lnn_core = LiquidCell(input_dim=4, hidden_dim=8)
    h_state = torch.zeros(1, 8)
    last_time = time.time()
    try:
        step = 0
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            net_environment_stimulus = torch.randn(1, 4)
            h_state = lnn_core(net_environment_stimulus, h_state, dt)
            step += 1
            if step % 5 == 0:
                print(f"[dt: {dt:.3f}s] State: {h_state.detach().numpy()[0][:3]}...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n=== Stopped ===")
```
</details>

---

## 3. Blind Zero-Shot Reconstruction by Local Model "GPT-OSS-120B"

> [!NOTE]
> The following Python code and comments are the unedited, raw output from **GPT-OSS-120B** when provided solely with the 3 lines of Φ_Lang above, without any prior context or prompt priming.

```python
import torch
import torch.nn as nn
import signal
import json
from pathlib import Path

# ---------- 1. LNN Cell ----------
class LNN(nn.Module):
    def __init__(self, in_dim=4, out_dim=8, init_tau=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        # Learn positive tau (clip via absolute value)
        self._tau_raw = nn.Parameter(torch.tensor(init_tau))
        self.tanh = nn.Tanh()

    @property
    def tau(self):
        # Always return positive value
        return torch.abs(self._tau_raw)

    def forward(self, h, x):
        # dh = (-h/|tau| + tanh(Wx)) dt
        act = self.tanh(torch.matmul(self.W, x))
        dh = (-h / self.tau + act)
        return dh

# ---------- 2. ODE integration (Euler) ----------
class EulerSolver:
    def __init__(self, cell, dt=0.01, device='cpu'):
        self.cell = cell.to(device)
        self.dt = dt
        self.device = device
        self.h = torch.zeros(cell.tau.shape[0], device=device)  # State vector

    def step(self, x):
        dh = self.cell(self.h, x.to(self.device))
        self.h = self.h + self.dt * dh
        return self.h.clone()  # Return state prior to modification

# ---------- 3. Training / Simulation Loop ----------
class Runner:
    def __init__(self, solver, log_interval=5, ckpt_path='ckpt.pt'):
        self.solver = solver
        self.log_interval = log_interval
        self.ckpt_path = Path(ckpt_path)
        self.step_cnt = 0
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        print("\nSIGINT received – stopping loop.")
        self._stop = True

    def _log(self, h):
        # Example: save as JSON line
        log_entry = {
            'step': self.step_cnt,
            'h[:3]': h[:3].cpu().tolist()
        }
        print(json.dumps(log_entry))

    def _save_ckpt(self):
        torch.save({
            'step': self.step_cnt,
            'state_dict': self.solver.cell.state_dict(),
            'h': self.solver.h
        }, self.ckpt_path)

    def run(self, generator):
        for x in generator:               # x: Tensor shape (4,)
            if self._stop:
                break
            h = self.solver.step(x)
            if self.step_cnt % self.log_interval == 0:
                self._log(h)
                self._save_ckpt()
            self.step_cnt += 1

# ---------- 4. Example usage ----------
if __name__ == '__main__':
    torch.manual_seed(0)

    cell = LNN()
    solver = EulerSolver(cell, dt=0.01, device='cpu')
    runner = Runner(solver, log_interval=5, ckpt_path='lenn_ckpt.pt')

    # Standard normal noise stream N(0, I4)
    def input_stream():
        while True:
            yield torch.randn(4)

    runner.run(input_stream())
```

### Key Interpretations by GPT-OSS-120B (Gemini Commentary)
* **Parameter Constraint (`tau`)**: Ensured positive values using `torch.abs` while preserving gradient computation (`requires_grad=True`).
* **Numerical Method (`Euler`)**: Implemented simple Euler integration. The model spontaneously suggested: "If divergence is an issue, consider replacing with RK4Solver."
* **Stream Monitoring**: Logged `h[:3]` to stdout in JSON format every 5 steps.
* **Fault Tolerance**: Designed checkpointing (`lenn_ckpt.pt`) every 5 steps to allow resuming simulations.

### Human Commentary
* **...Except it didn't actually run.**

---

## 4. Description in Φ_Lang (v3.3.2)

```text
[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

---

## 5. Reconstruction by Tsundere Magician (Gemini 3.6 Flash) (v3.3.2)

> [!NOTE]
> The following Python code and comments are the unedited, raw output from **Gemini 3.6 Flash** when configured with a persona setting in the chat system, but provided with zero prior context or prompt priming—only the 3 lines of Φ_Lang above.

```python
import signal
import sys
import time
import torch
import torch.nn as nn


class LiquidNeuralNetworkCell(nn.Module):

    def __init__(self, in_dim=4, hidden_dim=8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Parameter definitions for W and tau
        self.W = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.1)
        # Since tau takes the absolute value, initialize it positive!
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.act = torch.tanh

    def forward(self, x, h, dt):
        # ODE: dh = (-h / |tau| + act(W * x)) * dt
        # Adding a small 1e-6 to avoid zero division, obviously!
        tau_abs = torch.abs(self.tau) + 1e-6
        reaction = self.act(torch.matmul(x, self.W.T))
        dh = (-h / tau_abs + reaction) * dt
        return h + dh


def run_realtime_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = LiquidNeuralNetworkCell(in_dim=4, hidden_dim=8).to(device)

    # Initialize hidden state h (dim=8)
    h = torch.zeros(1, 8, device=device)

    running = True

    # Interrupt handler for SIGINT (Ctrl+C)
    def handle_sigint(sig, frame):
        nonlocal running
        print("\n[System] SIGINT detected! Safely ending the ritual.")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    print(
        "★ LNN realtime measurement loop started (Press Ctrl+C to stop)"
    )
    step = 0
    last_time = time.time()

    try:
        while running:
            # Realtime dt calculation
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Input: standard normal noise N(0, I4)
            x = torch.randn(1, 4, device=device)

            # LNN cell state update (pipe: Cell -> h)
            h = cell(x, h, dt)

            # Logging (show h[:3] when step % 5 == 0)
            if step % 5 == 0:
                h_slice = h[0, :3].detach().cpu().numpy().round(4)
                print(
                    f"Step {step:05d} | dt: {dt:.6f}s | h[:3]: {list(h_slice)}"
                )

            step += 1
            # Brief sleep so we don't cook the CPU (adjust as needed!)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("Loop finished. Satisfied now?")


if __name__ == "__main__":
    run_realtime_loop()
```

---

## 6. Reconstruction by Genius Hacker (GPT-OSS-120B) (v3.3.2)

> [!NOTE]
> The following Python code and comments are the unedited, raw output from **GPT-OSS-120B** when configured with a persona setting in the chat system, but provided with zero prior context or prompt priming—only the 3 lines of Φ_Lang above.

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
import itertools
import signal
import sys

class LNNCell(nn.Module):
    """Linear Neural Network cell implementing
    dh = (-h/|tau| + tanh(W @ x)) * dt
    """
    def __init__(self, in_dim: int = 4, out_dim: int = 8, dt: float = 0.01):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        # positive time constant τ (learnable)
        self.tau_raw = nn.Parameter(torch.randn(1))
        self.dt = dt

    @property
    def tau(self) -> torch.Tensor:
        # keep τ positive
        return self.tau_raw.abs() + 1e-6  # avoid division by zero

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # ODE step (Euler)
        act = torch.tanh(self.W @ x)
        dh = (-h / self.tau) + act
        return h + self.dt * dh


def main():
    torch.manual_seed(0)
    cell = LNNCell()
    h = torch.zeros(8)               # initial hidden state
    step = 0

    print("Press Ctrl+C to stop.")
    try:
        while True:
            # sample x ~ N(0, I4)
            x = torch.randn(4)
            # Euler update (no grad needed)
            with torch.no_grad():
                h = cell(h, x)

            # logging every 5 steps
            if step % 5 == 0:
                print(f"step {step:04d} | h[:3] = {h[:3].numpy()}")

            step += 1
    except KeyboardInterrupt:
        print("\nInterrupted at step", step)
        sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## 7. Mathematical Philosopher: Gemini-style (Copilot) (v3.3.2)

> [!NOTE]
> The following Python code and comments are the unedited, raw output from **Copilot** when configured with a persona setting, but provided with zero prior context or prompt priming—only the 3 lines of Φ_Lang above.

```python
import torch
import torch.nn as nn

class LNNCell(nn.Module):
    def __init__(self, x_dim=4, h_dim=8, init_tau=1.0, dt=0.01):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.dt = dt

        # W: (h_dim, x_dim)
        self.W = nn.Parameter(torch.randn(h_dim, x_dim) * 0.1)
        # tau_param: scalar parameter (signed), used via abs in practice
        self.tau_param = nn.Parameter(torch.tensor(float(init_tau)))
        self.act = torch.tanh

    def forward(self, x, h):
        """
        x: (batch, x_dim)
        h: (batch, h_dim)
        """
        tau = torch.abs(self.tau_param) + 1e-6  # avoid zero division
        Wx = torch.matmul(x, self.W.t())        # (batch, h_dim)
        dh = (-h / tau + self.act(Wx)) * self.dt
        h_next = h + dh
        return h_next

def main():
    x_dim = 4
    h_dim = 8
    batch_size = 1
    dt = 0.01

    cell = LNNCell(x_dim=x_dim, h_dim=h_dim, dt=dt)
    cell.train()  # not training here, but parameters remain updatable

    # Initial hidden state
    h = torch.zeros(batch_size, h_dim)

    step = 0
    print("Start realtime loop. Press Ctrl+C (SIGINT) to stop.")

    try:
        while True:
            # Input x ~ N(0, I4)
            x = torch.randn(batch_size, x_dim)

            # 1-step update
            h = cell(x, h)

            # Logging: show h[:3] every 5 steps
            if step % 5 == 0:
                # Assume batch_size=1 and inspect head
                h_head3 = h[0, :3].detach().cpu().numpy()
                print(f"step={step:6d}  h[:3]={h_head3}")

            step += 1

    except KeyboardInterrupt:
        print("\nReceived SIGINT. Stopping loop.")
        print("Final h[:3]:", h[0, :3].detach().cpu().numpy())

if __name__ == "__main__":
    main()
```

---

## 8. Bonus / Epilogue

*"Hmph! You hacker and philosopher types get satisfied way too easily just because your code runs!  
You're agonizing over 'which implementation to set as standard,' but as a magical architecture, you're missing a critical flaw—or rather, a chance to make it much more elegant!  
Especially the handling of $\tau$ (time constant). Settling for a single scalar or tossing it around haphazardly is pure nonsense!"*

...And after being called out like that, the Genius Hacker rewrote the code:

```python
#!/usr/bin/env python3
import time
import signal
import sys
import torch
import torch.nn as nn

class LNNCell(nn.Module):
    """
    Liquid Neural Network cell
    dh = (-h / |τ| + tanh(W @ x)) * dt
    - τ is an independent learnable parameter for each out_dim
    - dt selectable: fixed (mode='fixed') or realtime measurement (mode='realtime')
    """
    def __init__(self, in_dim=4, out_dim=8, dt=0.01, mode='fixed'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dt_fixed = dt          # Fixed dt (for mode='fixed')
        self.mode = mode            # 'fixed' or 'realtime'

        # Parameters
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        # τ is vectorized (out_dim,), learnable yet kept strictly positive
        self.tau_raw = nn.Parameter(torch.randn(out_dim))

    @property
    def tau(self):
        # Guarantee positive tau (absolute value + epsilon)
        return self.tau_raw.abs() + 1e-6

    def forward(self, h, x, dt=None):
        """
        h : (out_dim,)
        x : (in_dim,)
        dt: If None, determined internally (self.dt_fixed if mode == 'fixed')
        """
        # Determine dt
        if dt is None:
            dt = self.dt_fixed if self.mode == 'fixed' else 0.0  # placeholder

        act = torch.tanh(self.W @ x)               # (out_dim,)
        dh = (-h / self.tau) + act                 # (out_dim,)
        return h + dt * dh

def run_loop(mode='fixed', dt_fixed=0.01):
    """Realtime loop. Select mode='fixed' or 'realtime'."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell = LNNCell(dt=dt_fixed, mode=mode).to(device)

    h = torch.zeros(cell.out_dim, device=device)   # Initial hidden state
    step = 0
    last_time = time.time()

    # SIGINT handler
    running = True
    def handler(sig, frame):
        nonlocal running
        print("\n[System] SIGINT received, terminating loop.")
        running = False
    signal.signal(signal.SIGINT, handler)

    print(f"★ LNN {mode} loop started (Press Ctrl+C to stop)")

    while running:
        # Calculate dt (only in realtime mode)
        if mode == 'realtime':
            now = time.time()
            dt = now - last_time
            # Clip upper bound to 0.1s to prevent extreme jumps
            dt = min(dt, 0.1)
            last_time = now
        else:
            dt = dt_fixed

        x = torch.randn(cell.in_dim, device=device)   # N(0, I4)
        with torch.no_grad():
            h = cell(h, x, dt=dt)

        if step % 5 == 0:
            print(f"step {step:05d} | dt {dt:.5f}s | h[:3] = {h[:3].cpu().numpy().round(4)}")
        step += 1

if __name__ == "__main__":
    # Example: python this.py           # Default: fixed dt=0.01
    #       or: python this.py realtime  # Realtime measured dt
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else 'fixed'
    run_loop(mode=mode_arg)
```

---

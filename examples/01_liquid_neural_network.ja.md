# 実例 01: 流体ニューラルネットワーク (LNN) ODE ループ

わずか2行（+ヘッダー）の Φ_Lang 記述から、異なる系統の大規模言語モデルがリアルタイム常微分方程式（ODE）ソルバーをゼロショット復元した検証記録です。

[English](01_liquid_neural_network.md) | [日本語]

---

## 1. Φ_Lang による記述 (v3.3.1)

```text
[ phi:v3.3.1 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

---

## 2. 元の Python 実装（約40行の PyTorch スクリプト）

<details>
<summary>クリックしてオリジナルのソースコードを表示</summary>

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

## 3. ローカルモデル「GPT-OSS-120B」によるブラインド・ゼロショット復元コード

> [!NOTE]
> 以下の Python コードおよびコメントは、事前文脈やプロンプトによる誘導を一切与えず、上記3行の Φ_Lang のみを与えた際に **GPT-OSS-120B** が出力した未編集の生コードです。

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
        # 正の τ を学習させる（絶対値でクリッピング）
        self._tau_raw = nn.Parameter(torch.tensor(init_tau))
        self.tanh = nn.Tanh()

    @property
    def tau(self):
        # 常に正の値を返す
        return torch.abs(self._tau_raw)

    def forward(self, h, x):
        # dh = (-h/|τ| + tanh(Wx)) dt
        act = self.tanh(torch.matmul(self.W, x))
        dh = (-h / self.tau + act)
        return dh

# ---------- 2. ODE integration (Euler) ----------
class EulerSolver:
    def __init__(self, cell, dt=0.01, device='cpu'):
        self.cell = cell.to(device)
        self.dt = dt
        self.device = device
        self.h = torch.zeros(cell.tau.shape[0], device=device)  # 状態ベクトル

    def step(self, x):
        dh = self.cell(self.h, x.to(self.device))
        self.h = self.h + self.dt * dh
        return self.h.clone()  # 変更前の状態を返す

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
        # 例: JSON 行で保存
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

    # N(0, I4) の標準正規乱数ストリーム
    def input_stream():
        while True:
            yield torch.randn(4)

    runner.run(input_stream())
```

### GPT-OSS-120B が読み解いたポイント（Gemini解説）
* **パラメータ制約 (`tau`)**: `torch.abs` により常に正の値を保証しつつ、勾配計算（`requires_grad=True`）を維持。
* **数値計算手法 (`Euler`)**: シンプルなオイラー法を採用。モデル側から「発散が問題ならRK4Solver等への差し替えを推奨」との注釈を自発的に提示。
* **ストリーム監視**: 5ステップごとに `h[:3]` を JSON 形式で標準出力にログ。
* **フォールトトレランス**: 5ステップごとにチェックポイント（`lenn_ckpt.pt`）へ状態を保存し、途中再開を可能に設計。

### 人間の呟き
* **だが、これは動かなった・・・。

---

## 4. Φ_Lang による記述 (v3.3.2)

```text
[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

---

## 5.ツンデレ魔法使い（Gemini 3.6 Flash）（v3.3.2）による復元

> [!NOTE]
> 以下の Python コードおよびコメントは、チャットシステム上でペルソナ設定はされていますが、事前文脈やプロンプトによる誘導を一切与えず、上記3行の Φ_Lang のみを与えた際に **Gemini 3.6 Flash** が出力した未編集の生コードです。

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

        # パラメータ W と tau の定義
        self.W = nn.Parameter(torch.randn(hidden_dim, in_dim) * 0.1)
        # tau は絶対値を取るので正の初期値にしておくわ
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.act = torch.tanh

    def forward(self, x, h, dt):
        # ODE: dh = (-h / |tau| + act(W * x)) * dt
        # ゼロ割り防止に微小値 1e-6 を足しておくわね
        tau_abs = torch.abs(self.tau) + 1e-6
        reaction = self.act(torch.matmul(x, self.W.T))
        dh = (-h / tau_abs + reaction) * dt
        return h + dh


def run_realtime_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = LiquidNeuralNetworkCell(in_dim=4, hidden_dim=8).to(device)

    # 隠れ状態 h の初期化 (dim=8)
    h = torch.zeros(1, 8, device=device)

    running = True

    # SIGINT (Ctrl+C) の割り込みハンドラ設定
    def handle_sigint(sig, frame):
        nonlocal running
        print("\n[システム] SIGINTを検知したわ！儀式を安全に終了するわね。")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    print(
        "★ LNN リアルタイム計測ループ開始 (停止するには Ctrl+C を押してね)"
    )
    step = 0
    last_time = time.time()

    try:
        while running:
            # リアルタイム dt の計算
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # 入力: N(0, I4) の標準正規分布ノイズ
            x = torch.randn(1, 4, device=device)

            # LNNセルの状態更新 (pipe: Cell -> h)
            h = cell(x, h, dt)

            # ログ出力 (step % 5 == 0 のとき h[:3] を表示)
            if step % 5 == 0:
                h_slice = h[0, :3].detach().cpu().numpy().round(4)
                print(
                    f"Step {step:05d} | dt: {dt:.6f}s | h[:3]: {list(h_slice)}"
                )

            step += 1
            # CPUを回しすぎないように微小ウェイト（必要に応じて調整してね）
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("ループ終了よ。満足した？")


if __name__ == "__main__":
    run_realtime_loop()
```

---

## 6. 天才ハッカー（GPT-OSS-120B）（v3.3.2）による復元

> [!NOTE]
> 以下の Python コードおよびコメントは、チャットシステム上でペルソナ設定はされていますが、事前文脈やプロンプトによる誘導を一切与えず、上記3行の Φ_Lang のみを与えた際に **GPT-OSS-120B** が出力した未編集の生コードです。

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

## 7. 数理哲学者：Gemini風（Copilot）（v3.3.2）

> [!NOTE]
> 以下の Python コードおよびコメントは、ペルソナ設定はされていますが、事前文脈やプロンプトによる誘導を一切与えず、上記3行の Φ_Lang のみを与えた際に **Copilot** が出力した未編集の生コードです。

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
        # tau_param: スカラーパラメータ（符号付き）だが、実際のtauはabsで使う
        self.tau_param = nn.Parameter(torch.tensor(float(init_tau)))
        self.act = torch.tanh

    def forward(self, x, h):
        """
        x: (batch, x_dim)
        h: (batch, h_dim)
        """
        tau = torch.abs(self.tau_param) + 1e-6  # 0割り防止
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
    cell.train()  # ここでは学習はしないが、パラメータ更新可能な状態

    # 初期隠れ状態
    h = torch.zeros(batch_size, h_dim)

    step = 0
    print("Start realtime loop. Press Ctrl+C (SIGINT) to stop.")

    try:
        while True:
            # 入力 x ~ N(0, I4)
            x = torch.randn(batch_size, x_dim)

            # 1ステップ更新
            h = cell(x, h)

            # ログ: 5ステップごとに h[:3] を表示
            if step % 5 == 0:
                # batch_size=1 を仮定して先頭だけ見る
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

## 8. おまけ

ふん、ハッカーも哲学者気取りのお兄さんも、コードが動いたくらいで満足しちゃって甘いのよ！
「どの実装を基準にするか」なんて悩んでるみたいだけど、魔法の構造として決定的な欠陥…というか、もっと美しくできるポイントを見落としてるわ！
特に $\tau$（時間定数）の扱いよ。スカラ１つで済ませたり適当に扱ったりなんてナンセンス！

と突っ込まれた天才ハッカーが書き直したコード。

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
    - τ は out_dim 毎に独立した learnable パラメータ
    - dt は固定 (mode='fixed') かリアルタイム計測 (mode='realtime') で選択可能
    """
    def __init__(self, in_dim=4, out_dim=8, dt=0.01, mode='fixed'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dt_fixed = dt          # 固定 dt（mode='fixed' 用）
        self.mode = mode            # 'fixed' or 'realtime'

        # パラメータ
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        # τ はベクトル化 (out_dim,)、学習可能だが常に正に保つ
        self.tau_raw = nn.Parameter(torch.randn(out_dim))

    @property
    def tau(self):
        # 正の τ を保証（絶対値 + ε）
        return self.tau_raw.abs() + 1e-6

    def forward(self, h, x, dt=None):
        """
        h : (out_dim,)
        x : (in_dim,)
        dt: None のときは内部で決定（mode が 'fixed' の場合は self.dt_fixed）
        """
        # dt の決定
        if dt is None:
            dt = self.dt_fixed if self.mode == 'fixed' else 0.0  # placeholder

        act = torch.tanh(self.W @ x)               # (out_dim,)
        dh = (-h / self.tau) + act                 # (out_dim,)
        return h + dt * dh

def run_loop(mode='fixed', dt_fixed=0.01):
    """リアルタイムループ。mode='fixed' か 'realtime' を選択。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell = LNNCell(dt=dt_fixed, mode=mode).to(device)

    h = torch.zeros(cell.out_dim, device=device)   # 初期隠れ状態
    step = 0
    last_time = time.time()

    # SIGINT ハンドラ
    running = True
    def handler(sig, frame):
        nonlocal running
        print("\n[システム] SIGINT 受信、ループを終了します。")
        running = False
    signal.signal(signal.SIGINT, handler)

    print(f"★ LNN {mode} ループ開始 (Ctrl+C で停止)")

    while running:
        # dt 計算（リアルタイムモードの場合のみ）
        if mode == 'realtime':
            now = time.time()
            dt = now - last_time
            # 極端に大きくならないよう上限を 0.1 秒にクリップ
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
    # 例: python this.py  # デフォルトは fixed dt=0.01
    #   または: python this.py realtime  # リアルタイム dt
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else 'fixed'
    run_loop(mode=mode_arg)
```

---
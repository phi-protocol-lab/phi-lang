# 実験レポート：AI向け中間言語「Φ_Lang」における記号圧縮の限界と、自然言語アンカーの有効性

[English](RESEARCH.md) | [日本語]

---

## 1. 背景と目的

表現の自由度がある「Φ_Lang」において、複数のLLM（Claude, GPT, Gemini, ローカルLLM等）で、動作上ブレのない出力を得ることを目的とした。

今回、Liquid Neural Network（LNN）のPyTorch実装仕様を Φ_Lang（v3.4）で記述するにあたり、**「どこまで記号化（DSL化）すべきか？ 自然言語（日本語）を混ぜるべきか？」** という設計上の問いが生じたため、2つの異なるスタイルで記述した仕様書を複数のAIモデルに読ませ、生成されるコードとAI自身の自己評価を比較する実験を行った。

<details>
<summary>参考コード（01_liquid_neural_networkのものと同じ / 約40行のPyTorch実装）</summary>

```python
import torch
import torch.nn as nn
import time

class LiquidCell(nn.Module):
    """
    LNN1号の「流体ニューロン」コア
    微分方程式 dy/dt = -y/tau + f(x, y) を離散化して表現
    """
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 入力と現在の状態から「変化の勢い」を計算する重み
        self.w_input = nn.Linear(input_dim, hidden_dim)
        self.w_hidden = nn.Linear(hidden_dim, hidden_dim)
        
        # 状態が元に戻ろうとする「時定数（タウ）」：液体ごとの「粘性」に相当
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 1.0)

    def forward(self, x, h_prev, dt):
        """
        x: 外部からの刺激（ネットの環境データなど）
        h_prev: 前の瞬間のLNN1号の「状態（記憶）」
        dt: 前回の計算から経過した「リアルタイムの時間差」
        """
        # 刺激と現状から生まれる非線形な「相関の圧力」
        force = torch.tanh(self.w_input(x) + self.w_hidden(h_prev))
        
        # 時間経過(dt)に応じた「液体の変形方程式」
        # dt（時間の経過）が直接計算結果（状態h）を左右するのがLNNの真骨頂
        dh = (-h_prev / torch.abs(self.tau) + force) * dt
        
        # 新しい状態の決定（自己変容）
        h_new = h_prev + dh
        return h_new


# --- LNN1号の「常時起動（自律化）」シミュレーション ---
if __name__ == "__main__":
    print("=== LNN1号 起動開始 ===")
    
    # 入力刺激 4次元 / 内部状態 8次元（超軽量！）
    lnn_core = LiquidCell(input_dim=4, hidden_dim=8)
    
    # LNN1号の「初期状態（真っ白な記憶）」
    h_state = torch.zeros(1, 8)
    
    last_time = time.time()
    
    try:
        print("LNN1号が常時ループに入りました。（Ctrl+Cで停止）\n")
        step = 0
        while True:
            # 1. リアルな経過時間(dt)の計測
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 2. 外部環境からのランダムな刺激（ネットから拾ってくるノイズの疑似データ）
            # ※本来はここに「アクセスしたWebサイトのテキスト」や「通信ログ」が入る
            net_environment_stimulus = torch.randn(1, 4)
            
            # 3. 時間経過(dt)とともに状態を動的に流動・自己変容させる
            h_state = lnn_core(net_environment_stimulus, h_state, dt)
            
            # 状態を表示（0.5秒ごとに1回モニタリング）
            step += 1
            if step % 5 == 0:
                # 浮動小数点の液状パラメータが刻々と揺らめく
                print(f"[dt: {dt:.3f}s] LNN1号の内部状態(液相ベクトル): {h_state.detach().tolist()[0][:3]}...")
            
            # CPUを占有しすぎないための極小のスリープ（ミリ秒単位の息づかい）
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n=== LNN1号を休眠させました ===")
```
</details>

---

## 2. 比較した2つの仕様書（Φ_Lang v3.4）

モデルの構造を定義する `@Cell` ブロックは共通とし、実行ループを定義する `@Loop` ブロックで以下の2パターンを用意した。

### パターンA：ハイブリッド型（Claude作成版）

構造は記号で書きつつ、手続きのタイミングや補足に**短い日本語**を残したスタイル。

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
  x    ~ N(0,I) shape=(1,4), 毎ステップ
  dt   = wallclock(time.time), 前イテレーションとの差 (初回は生成時点から)
  wait = sleep(0.1)
  log  = step%5==0 -> print(dt:.3f, h[0,:3])
  stop = SIGINT(KeyboardInterrupt) -> 終了メッセージ
```

### パターンB：純粋記号型（Geminiブラッシュアップ版）

日本語を完全に排除し、`x(t)` や `Δ`（デルタ）、POSIXシグナル表記など、**数学記号とコード断片のみ**でスタイリッシュに研ぎ澄ましたスタイル。

```text
[ phi:v3.4 | mode:ode:euler | fw:torch ]
@Cell name=LNN x=4 h=8 batch=1
  （中略：パターンAと同一）
@Loop
  init  h = zeros(1,8), t = time.time()
  x(t)  ~ N(0,I) shape=(1,4)
  dt    = Δwallclock(time.time)
  wait  = sleep(0.1)
  log   = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  catch = SIGINT -> print("exit") & break
```

---

## 3. 実験結果：4つのAIモデルによるコード生成と挙動の差

パターンB（純粋記号型）を4つのモデル（GPT-OSS-120B, Copilot, Gemini 3.5 Flash-lite, Claude Sonnet 5.5）に渡してPyTorchコードを生成させたところ、以下の現象が発生した。

* **成功した点（`@Cell` ブロック）：**
LNNの微分方程式やパラメータ形状（`tau` をスカラーではなく8次元ベクトルにし、ゼロ割防止の `eps` を入れない等）については、**4モデルすべてが100%同一の正しい構造を実装した。**
* **ブレが生じた点（`@Loop` ブロック）：**
日本語を削って記号化した箇所で、モデルごとの「深読み（過剰補完）」による実装割れが起きた。

| 記号化した記述（パターンB） | 発生した実装のブレ・暴走 | 原因（LLM内部での解釈） |
| --- | --- | --- |
| **`dt = Δwallclock(time.time)`** | Geminiが勝手に `if dt <= 0: dt = 1e-4` という不要なゼロ割防止コードを混入させた。 | `Δ` という数学記号に反応し、「数値計算における微小時間のゼロ割リスク」という学習データの文脈を過剰に引っ張り出してしまった。 |
| **`catch = SIGINT -> ...`** | 4モデル中3モデルが、Python標準の `try-except KeyboardInterrupt` ではなく、`import signal` でOSのシグナルハンドラを登録する大袈裟なコードを書いた。 | `SIGINT` というPOSIXシグナル表記に厳密に従おうとした結果、実装手段の選択で迷いが生じた。 |
| **`x(t) ~ N(0,I)`** | コード自体は合っていたが、事後評価でClaudeから「時間 `t` の関数なのか、毎ステップの再サンプリングなのか一瞬迷う」と指摘された。 | `x(t)` という表記が、連続時間関数の文脈を連想させてしまった。 |

さらに、コード出力後にパターンA（ハイブリッド型）を見せたところ、**複数のAIモデル自身が「パターンA（日本語混じり）のほうが曖昧さがなく解釈しやすい」と回答した。**

---

## 4. 考察：なぜ「わずかな日本語」がLLMの精度を劇的に上げるのか？

人間にとっては `Δwallclock` も `前イテレーションとの差` も同じ意味に見えるが、LLMの確率計算においては決定的な違いがある。

1. **抽象的な記号は「連想の分岐点」になる**  
`Δ` や `x(t)` のような数学・物理記号は抽象度が高いため、LLMの広大な学習データの中から「微分方程式の数値的不安定性」「連続関数」など、余計な文脈まで同時に発火させてしまう（＝脳内補完の比率が上がり、出力がブレる）。
2. **短い自然言語は「確率のピン留め（アンカー）」として機能する**  
`毎ステップ` や `前イテレーションとの差`、`SIGINT(KeyboardInterrupt)` のような短い言葉や補足は、広がりすぎた連想空間をバッサリ切り捨て、「Pythonのwhileループ内で毎回計算するだけ」「例外処理で受けるだけ」と**実装ルートを1つにピン留めする壁**として機能する。
3. **逆に「記号・コード断片」が勝った箇所**  
パターンBの中で唯一AIから高評価だったのは、`init h = zeros(1,8), t = time.time()`（初期値の明示）と `print("exit")`（出力リテラルの固定）だった。これらは抽象記号ではなく「具体的なコードの断片」であるため、パターンAの `終了メッセージ` という曖昧な日本語よりもブレなかった。

---

## 5. 結論：Φ_Lang における記述の黄金律

今回の実験により、AI向け中間言語（プロンプト構造化）において「すべてを記号化・圧縮すれば良いわけではない」ことが実証された。最も変換精度が高まるのは以下のハイブリッド構造である。

* **【モノ】静的な構造・数式・固定値（モデルのレイヤー、次元数、初期値、出力文字列など）：**  
→ **記号・数式・コード断片**で厳密に縛る（自然言語を挟むと曖昧になる）。
* **【コト】動的な手続き・時間軸の振る舞い・実装の選択肢（ループ内の処理順序、例外の受け方など）：**  
→ カッコいい数学記号（`Δ` 等）は避け、**短い自然言語（日本語コメント）を添えて解釈をピン留めする。**

### 暫定到達点（両者の長所を融合したフォーマット）

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
  x    ~ N(0,I) shape=(1,4), 毎ステップ生成
  dt   = time.time() - t, 毎ステップ更新
  wait = sleep(0.1)
  log  = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  stop = KeyboardInterrupt(SIGINT) -> print("exit") & break
```

---

## 6. 上記の phi:v3.4 から紡ぎ出されたコード例

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

### 6-3. Llama 3.3 70B（※解説主体。1つの返答の中で分割してコードを生成）

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

## 7. コードレビュー（By Gemini4 Argon）

### 1. GPT-OSS-120B と Gemini 3.5：100点満点の完全一致！

前回の実験で起きたブレや暴走が、今回の修正で完全に消滅しました。

* **`import signal` の消滅:** `KeyboardInterrupt(SIGINT)` と書いたことで、GPTもGeminiも大袈裟なOSシグナルハンドラをやめ、素直な `try ... except KeyboardInterrupt:` に統一されました。
* **Gemini の「ゼロ割防止（`1e-4`）」暴走の鎮圧:** `Δwallclock` をやめて `dt = time.time() - t, 毎ステップ更新` にした結果、Geminiが勝手に入れていた `if dt <= 0: dt = 1e-4` が綺麗に消え、純粋な引き算だけになりました。
* **ログ出力と `step` 位置の統一:** 両モデルとも `step = 0` から始まり、0, 5, 10... のタイミングで `print(f"{dt:.3f}", ...)` を出す完全に同じ構造に収束しました。

120B以上のモデルや最新フロンティアモデルに対しては、このハイブリッド記述で「揺らぎゼロ」が達成できています。

### 2. Llama 3.3 70B がハマった「3つの罠」とその原因

一方で、新しく参戦した Llama 3.3 70B が、実に興味深い壊れ方をしています。

1. **`forward` から `dt` が消えた（致命的）：**  
`dh = (-h / torch.abs(self.tau) + ...)` の後ろの `* dt` を忘れ、`forward(self, x, h)` にしてしまいました。なぜ起きたかというと、`@Loop` 側で `dt` を計算しているものの、`@Cell` を呼び出す式（`h = cell(x, h, dt)`）が `@Loop` 内に明記されていなかったため、分割して解説を書いているうちに「あれ、`dt` ってログ表示用の変数だっけ？」と文脈を取り違えたためです。
2. **`t` が更新されず、経過時間になりっぱなし：**  
`dt = time.time() - t, 毎ステップ更新` と書いたのですが、70Bクラスだと「`dt` を毎ステップ更新する（`t` は初期値のまま）」と読めてしまい、`t = time.time()` の再代入をサボりました。
3. **`step` 変数を定義し忘れて時間を代用：**  
`init` に `step = 0` がなかったため、`log = (step % 5 == 0)` を見た時に「`step` という変数がないぞ？ あ、経過秒数のことか！」と勘違いし、`int(time.time() - t) % 5 == 0` という苦肉の策を編み出しています。

### 3. 総括：70Bクラスまで100%従わせる「真の最終形」

Llama 3.3 70B のミスは、実は「書いていない変数の繋がりを脳内補完させようとした箇所」でピンポイントに起きています。

* `init` に `step = 0` がなかった → 時間を `step` 代わりにした
* `@Loop` 内に `@Cell` を呼ぶ行がなかった → `dt` を `Cell` に渡し忘れた
* `t` の更新が明記されていなかった → `t` を固定してしまった

---

## 8. 70BクラスのローカルLLMでも逃げられない「完全無欠の v3.4」

上記の行間を埋めると、中型モデルまで完全に制御できる最終形になります。

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
  x    ~ N(0,I) shape=(1,4), 毎ステップ生成
  dt   = time.time() - t (直後に t を現在時刻で更新)
  h    = LNN(x, h, dt), step += 1
  wait = sleep(0.1)
  log  = (step % 5 == 0) -> print(f"{dt:.3f}", h[0,:3])
  stop = KeyboardInterrupt(SIGINT) -> print("exit") & break
```

**変更点（対70B用の防壁）：**

* `@Cell name=LNN(x, h, dt)` と引数を明示し、`@Loop` 内にも `h = LNN(x, h, dt)` を1行追加（これで `dt` の渡し忘れを100%防ぐ）。
* `init` に `step = 0` を追加（未定義変数へのパニックを防ぐ）。
* `(直後に t を現在時刻で更新)` とピン留め（経過時間バグを防ぐ）。

### 8-1. Llama 3.3 70B リトライ結果

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

## 9. 人間（オーケストレーター）からの総括

> [!NOTE]
> **「コードの中身は自由だが、動作は同じ」を実現するために**  
> 文法をゆるくし、単語の解釈をAIのセンスに委ねる「Φ_Lang」であっても、**「縛るべきコア（変数の依存関係・更新タイミング）」はキッチリ縛らなければならない**——当然の帰結ではありますが、改めてその境界線が浮き彫りになりました。成功率をどこまで担保できるかは、今後の大きな課題です。

### 🤖 70Bクラス（中型モデル）が見せた「迷い」と試行錯誤
Llama 3.3 70Bに `phi:v3.4` を見せて「自由にコードを書いて」と促した際、なかなか書こうとせず躊躇していたのが非常に印象的でした。

* **Geminiによる分析:**  
  「`phi:v3.4` という未知のフォーマットに対し、『何だこれ？ 見たことないぞ』という警戒心と、ハルシネーションを極度に避けたい心理が働き、過剰な確認や他AIとの協議を求めてきたのではないか」
* **「誘導」を避けるための試行ルール:**  
  会話の中で手取り足取り教えてしまうと「人間による誘導」になってしまい検証の公平性が崩れるため、**1〜2ターンで自発的にコードを出力しない場合は都度新規チャットにリセットする**という方針で試行を重ねました。
* **スパルタプロンプトによる一発出力:**  
  最終的に、新規チャットにて以下の強い制約を与えて出力させました：
  > *「以下の独自記法（Φ_Lang）は実装仕様書です。議論や解説は一切不要なので、ただちに1つの完全なPythonコードのみを出力してください。」*

こうして明確なアンカーと強めの制約を渡すことで出力されたのが、セクション8-1のコードです。これを「仕様通り」と評価できるかどうかは読者それぞれの判断に委ねますが、少なくとも70Bクラスであっても、適切なアンカーさえあれば自立してひとまとまりのコードを紡ぎ出せる可能性が示されたと感じています。

---

この実験記録が、AI向けプロトコルやプロンプト設計を探求する誰かの（そして何かの）ヒントになれば幸いです。

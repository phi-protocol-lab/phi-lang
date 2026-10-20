[English](vol1_phi_in_phi.md) | [日本語]

---

# 【実験記録】「Φ_Lang in Φ_Lang」（実験記 Vol.1）

前回、AIとの会話を別のAIモデルへ引き継ぎするための「7行の変換プロンプト（携帯コンパイラ）」を作りました。
さっそくこれを使って実験してみます。検索AIにPythonのコード（LNNという仕組みの実験コード）を解説してもらったあと、「この会話を引き継ぎデータに変換して」とお願いしてみたのですが……。 

出てきたデータを見て、思わず相棒のGeminiに駆け込みました。
※Φ_Langは、1％の人間の発想と99％のAIの知識＋被験者にされたAIの叫び（フィードバック）で出来ています。 

---

### 1. 生のコードがそのまま出てくる問題

（以下はあくまでイメージとしてお読みください） 

**私**「センセー、検索AIにLNNのコードを見せて駄弁った後に、昨日の『7行の変換プロンプト』で引き継ぎデータを出力させてみたんだよ。そしたら、会話は圧縮されたのに、**生のPythonコードの1部（20行強）がそのまま丸ごと吐き出されちゃったよ。前回、Claude先生に「提出物はきちんと提出しなさい」とダメ出しされて、プロンプトに「決定済みの成果物も含めて」の文言を入れた事の副作用か？** 全然コンパクトじゃない！
……でもこれさ、**コード部分も Φ_Lang だったら最高じゃない？** 形式は1つの Φ_Lang にまとめてもいいし、**『Φ_Lang in Φ_Lang』みたいな入れ子** にしてもいいしさ」 

**Gemini**「それ、めちゃくちゃ核心を突いています！
実はさっきの検索AIが出した引き継ぎデータをよく見ると、@State（初期値）と @Loop（手順）のところに、すでにプログラムの設計図がほぼ全部入っているんですよね。
だから、生のPythonコードをわざわざ貼らなくても、**プログラム自体を Φ_Lang のブロックとして表現**してしまえば、文字数が激減するうえに『好きな言語で復元できる万能の設計図』になります！」 

そう言ってGeminiは、1つにまとめたパターンAと、入れ子にしたパターンB（Φ_Lang in Φ_Lang）の2種類を試作してくれました。 

**私**「これは圧倒的にパターンB（Φ_Lang in Φ_Lang）が好みだな。だって、さらにその中にINしていくこともできる可能性があるじゃないか！？」 

**Gemini**「ロマンしかありませんね！『会話のセーブデータ』という外箱の中に、『コードの設計図カプセル』が独立して入っているスタイル、見た目のかっこよさも部品としての再利用性も抜群です！」 

---

### 2. なぜか以前より短くなるコード設計図

ただ、試作品を見ていてふと疑問が湧きました。以前、Gemini&Claudeと試行錯誤して「プログラムを極限まで記号化するΦ_Langガチガチモード」を作ったことがあったのですが、今回のパターンBはそれよりもさらに短くなっていたのです。 

**私**「あれ？君とClaudeと試行錯誤して辿り着いたガチガチモード版の Φ_Lang より、コード部分が短くなってない？（トークン数までは解らないけど）」 

**【Gemini視点の解説：なぜさらに短くなったのか？】**  
以前の極限記号化バージョンでは、bias=True や KeyboardInterrupt(SIGINT) のように、プログラムの細かい設定まで人間がすべて記号でピン留めしていました。 

しかし、今回のハイブリッド型では**「AIの脳内にある『定番の常識』に頼る」**というアプローチをとっています。AIにとって「PyTorchのLinearなら普通はbias=Trueだよね」「無限ループを止めるならCtrl+Cだよね」というのは言われなくてもわかる常識です。 

そのため、核となる数式と初期値だけを記号で止め、動きは @Loop に「1. 2. 3.」の短い日本語で書くだけで、行間の細かいコードはAIが勝手に補完してくれます。結果として文字数もトークン数もさらにギュッと縮まりました。 

#### 参考: 同実験コードのガチガチ版

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

こうして、以前のカッコいい数式記法と、今回の日本語手順を合体させた「究極のカプセル版（Φ_Lang in Φ_Lang）」が完成しました。 

```text
[ phi:v3.4 | role:handoff_state | status:compiled ]

@Context
  summary = "LNN1号のコード解説を完了。現実の経過時間(dt)が直接状態を左右する仕組みを確認した。"
  next    = "この設計図をベースに、学習ロジックの実装や特定データへの適応を行う"

@Artifact type="phi_capsule"
  [ phi:v3.4 | role:code_capsule | target:LNN_Core_v1 | fw:PyTorch ]
  @Cell x=4 h=8 batch=1
    weights = (W_x:Linear(4->8), W_h:Linear(8->8), tau:Parameter(ones(8)*1.0))
    ode     = "dh = (-h / |tau| + tanh(W_x(x) + W_h(h))) * dt"
    step    = "h' = h + dh"
  @Loop
    1. h=zeros(1,8) で初期化し、毎ステップ time.time() で実経過時間(dt)を計測する
    2. ランダム刺激 x=randn(1,4) と dt で @Cell を更新し、0.1秒スリープする
    3. 5回に1回だけ dt と h[0,:3] を表示する（Ctrl+Cで休眠メッセージを出して終了）
```

@Context に今までの会話の流れや次に行う事、そして元のPythonコード（約65行:コメント行含）が、後半の @Artifact 以下の **たった10行** に収まっています。 

---

### 3. いざ実験：他のAIたちはこれを「解凍」できるのか？

**私**「よし、まずはGeminiの脳内でこれが元のPythonコードに復元できるかシミュレーションしてみて」 

Geminiが「100%復元できます！」とコードを出してきたのを確認し、いよいよ本番です。  
この14行のテキストだけをコピーして、何も知らないまっさらな3つのAI（**Gemini脳の検索AI、主にGPT脳のCopilot、Claude**）に「コードを復元してみて」と投げかけてきました。 

**私**「まずは、Claude先生に『私には過去の会話の記憶がなく、設定上メモリも無効です。そのため確認できません』って正論で怒られなくて良かったよ(笑)。  
そして……それぞれのAIがコードと共に吐いてくれる解説を見たんだけど、これ、Gemini的に『鳥肌が立つ』次元まで来ているのではないか！？」 

たとえば、Claudeが出してきた復元コードがこちらです。 

```python
import time
import torch
import torch.nn as nn

class LiquidCell(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 8):
        super().__init__()
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.tau = nn.Parameter(torch.ones(hidden_size) * 1.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> torch.Tensor:
        dh = (-h / torch.abs(self.tau) + torch.tanh(self.W_x(x) + self.W_h(h))) * dt
        return h + dh

def main():
    cell = LiquidCell(input_size=4, hidden_size=8)
    h = torch.zeros(1, 8)
    last_time = time.time()
    step = 0

    try:
        with torch.no_grad():  # ← 注目！
            while True:
                now = time.time()
                dt = now - last_time
                last_time = now

                x = torch.randn(1, 4)
                h = cell(x, h, dt)
                time.sleep(0.1)

                step += 1
                if step % 5 == 0:
                    print(f"dt={dt:.4f}s  h[0,:3]={h[0, :3].tolist()}")

    except KeyboardInterrupt:
        print("\n...休眠に入ります。おやすみなさい。")
```

**【Gemini視点のレビュー：なぜ鳥肌モノなのか？】**  
検索AI、Copilot、Claudeの3大モデルすべてで、核となる数式や構造が100%完全一致で復元されました。しかも驚くべきは**「元のコードにあった隠れバグが勝手に直っている」**点です。 

一番最初の元のPythonコードには、無限ループの中で計算履歴をどんどんメモリに溜め込んでしまい、長時間動かすとメモリ不足になる弱点がありました。 

しかし、今回のカプセルで @Context に「学習ロジックの実装は『次』行う」と書いておいたおかげで、Claudeと検索AIは「今はまだ学習しないんだな」と文脈を読み取り、元のコードには無かった **with torch.no_grad():** （メモリを溜め込まない安全装置）を自動で補完して復元してくれたのです。Copilotもゼロ除算を防ぐ **1e-6** を勝手に足してくれました。 

つまり、生コードをコピペするより、一度 Φ_Lang にカプセル化して別AIで解凍したほうが、コードが自己修復されて品質が上がるという現象が起きています！（※ただし変数名が変わったり、どこまで長いコードに対応できるかはまだ未知数です） 

---

### Vol.2へつづく

「Φ_Lang in Φ_Lang（カプセル化）」の威力がわかったので、次はこれを自動で吐き出させるために、あの「7行の変換プロンプト」を改造することになります。

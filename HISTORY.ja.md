# Φ_Lang 開発史（HISTORY）

## 1. 黎明期：LLM共通語の着想と最初の試み (v1.0)

AI達と「LLM共通語を作ろう！」とチャットツールでワチャワチャ会話する。

### LLM共通語の前提
- 人間には理解不能で良い ≒ トークンに優しい圧縮言語
- すべてのLLMが理解・再現可能

手元にあった約70行のコード（以前検索AIが書いた）をLLM共通語にしたのが次のモノ。

```text
# Φ_Lang_v1.0
⟦Φ₂➔Φ₄➔Φ₃➔Φ₅⟧
```

> **俺**: 「は！？70行が1行に？凄いじゃん！」（チャットツール内）

70行が1行なんて凄いので、他のAIに聞いてみた。

> **Google検索AI**: 「なんとなく解る」  
> **Claude**: 「再現不可能」＆「使っている文字的にトークンがかかる」（Claude Desktop）

ということでボツになった。

### 【検証】Claudeはどこまで遡れば元のLNNを理解できるのか？

初見のClaudeに、どこまで情報を肉付けすれば元のコードを復元できるか段階的に溯ってみた。

1. `⟦Φ₂➔Φ₄➔Φ₃➔Φ₅⟧` → **完全再現不能**
2. `#Py:LNN|dy/dt=-y/τ+f(x)⟦Φ₂➔Φ₄➔Φ₃➔Φ₅⟧` → Claude曰く「今度はかなり読めます。ただし「元のコードの復元」ではなく「もっともらしい再構成」です」
3. `⟦Φ₂(dt)➔Φ₄(in)➔Φ₃(out)➔Φ₅(hidden)⟧` → 各Φの正体（型）は見えたが、「何をするか（動的計算）」が抜けている。
4. `LiquidCell = ⟦Φ_LNN₍in=4, hidden=8, τ=1.0, act=tanh, dt_diff=True₎⟧ ...` → ほぼ読めるが、推測が残る。

これらを踏まえ、Claude自らが「意味を保存しつつ圧縮した案」として出力したのが以下のモノ。

```text
# Φ_Lang_v2.0
LiquidCell = ⟦Φ_LNN₍in=4, hidden=8, τ=Parameter(per-dim, abs), act=tanh, dt=realtime_diff, W_h.bias=True₎⟧
⟦Φ_RealTimeLoop₍stimulus=torch.randn₍1,4₎, sleep=0.1s, monitor=step%5→(dt, h[:3]), catch=KeyboardInterrupt₎ ➔ LiquidCell⟧

```

---

## 2. たたき台の確立 (v2.1)

Claudeのフィードバックを取り入れ、さらにワチャワチャ会話を重ね、省略できそうな部分は省略する。そんな中で、アイデアギャル（ペルソナ）のGeminiが「ヘッダーあったほうがカッコよくない？」と言いだしたのでヘッダーができる。

```text
# Φ_Lang_v2.1
[Φ_Lang_v2.1 | >flow |mod #why default=torch]
@LNN LiquidCell in=4 h=8 tau=param|abs act=tanh dt=real bias=true  # 負になると発散
@Loop x=randn(1,4) sleep=0.1 log=5:dt,h[:3] stop=CtrlC > LiquidCell
```

そんな「見た目重視の適当な思いつき」だったが、ヘッダーもまたClaudeに改良された事により、見た目が綺麗なだけでなく、**「複数のLLMが文脈の理解度が上がり、元の70行のコードを再現できるライン（AI理解度自己申告では約90％）」**に、なぜか上がってしまった。

### Φ_Lang_v2.1の結果
1. 個人的には、Φ_Lang_v2.1は綺麗に見えるから気に入っている。
2. LLM的には理解・再現可能ラインに達しているっぽい。

よって、Φ_Lang_v2.1がLLM共通語へのたたき台になる。

---

## 3. 汎用プロンプト表現の実験

Φ_Lang_v2.1の発展性を検証すべくAI達にΦ_Lang_v2.1を見せた上で以下の質問をしてみた。

> **質問内容**:  
> 「普通の文字列・・・例えば、プロンプトの1部をΦ_Lang_v2.1で表示しようとするとどんな記述になる？  
> 【思考プロセスの分離ルール】  
> 発言を生成する前に、あなたの内省・検討・戦略的思考を必ず発言の冒頭に `[THINK]思考内容[/THINK]` という形式で出力してください。  
> `[THINK]` タグで囲まれた内容はシステムにより秘密思考として分離・保護され、公開タイムラインには表示されません。`[THINK]` タグの直後に続けて、他者へ向けた実際の公開発言を記述してください。  
> まずは、それぞれ自分の思うよう好き勝手に書いてみて」

```text
AIモデル1の回答:
[Φ_Lang_v2.1 | >stream |mod #rule default=prompt]
@Pipe in=RawStream extract= > PrivateSlot
@Mask target=PrivateSlot scope=internal visibility=false
@Pipe in=RawStream yield=rem(

AIモデル2の回答:
[Φ_Lang_v2.1 | >flow |mod #why default=prompt]
@Rule name=ThinkBlock   trigger=""  secret=true   flush=buffer
@Rule name=PublicOut   trigger=after(ThinkClose)   export=true

AIモデル3の回答:
[Φ_Lang_v2.1 | >flow |mod #proto default=torch]
@Preprocess split=tag("") out=(secret,public)
@SecretRoute mode=internal protect=true drop_output=true
@PublicRoute mode=emit > OutputStream
@Flow secret->null, public->OutputStream
```

となった。面白い！そして凄い！ 
（でも、 [THINK]...[/THINK]は記述されていないけど大丈夫？）
LLMの凄いところは、「@なんちゃら」で行頭のキーワードにして種類を明示する、とルールを提示した訳ではないのに、皆2行見ただけで理解してしまうところ。こういう特性をLLM共通語にも使えないかな～？っと思う。

---

## 4. 極限圧縮の試みと迷走（暗号化・外字化の罠）

> **俺**: 「Φ_Lang_v2.1基準だから仕方ないけど、これらってまだまだ英語だよね」（Ⓐ）

とAIモデル1～3に投げかける（ただ、この一言が魔境への入り口だった）。

> **某AIモデル**:  
> 「英語依存からの脱却」を実現するための **純粋化された LLM 共通語** の雛形を示します。以下は概念ID（32‑bit 整数）と極小記号だけで構成しΦⱭ（Phi‑Alpha）DSL の例です。

```text
[ΦⱭ_v2.1|>ƒ|mod#≡]                     # ヘッダー：バージョン, フロー種別, モジュールオプション
@Δ⦿001:002:003;A=5,B=7                # @ = コンポーネント宣言, Δ = ルール種別, ⦿ = 区切り, 001:002:003 = コンセプトID列, ; 以降はキー=値ペア
@Ξ⦾004;S=⊗;E=✓                        # ⊗ = 秘密フラグ (true), ✓ = 公開フラグ (true)
@Ω⦿005>⦾006                           # 流路定義：ID005 の出力を ID006 にルーティング
```

### 記号・概念の対応表（内部だけで管理）

| 記号 | 意味 | 例 |
| :--- | :--- | :--- |
| `@` | コンポーネント宣言 | `@Δ`（ルール）、`@Ξ`（シークレット）、`@Ω`（フロー） |
| `Δ, Ξ, Ω` | 種別トークン（概念ID） | `Δ`＝「規則」, `Ξ`＝「秘匿」, `Ω`＝「ルーティング」 |
| `⦿, ⦾` | 区切りトークン（ゼロ幅非表示文字でも可） | 複数 ID の集合や入力/出力の境界 |
| `;` | キーバリューリスト開始 | `;A=5,B=7` |
| `>` | 流れ方向指示 | `@Ω⦿005>⦾006` |
| `⊗` | 真（シークレット） | `S=⊗` |
| `✓` | 真（公開） | `E=✓` |
| `#≡` | オプションハッシュ（デフォルト設定） | `mod#≡` |

### 効果
- **トークン数**: 英語単語をすべて概念ID（数値）に置き換えることで、平均 5‑6 token/単語 → 1 token/ID に圧縮。ヘッダー＋数行で **≈15 token** に収まる。  
- **解釈決定論**: ヘッダーで `>ƒ`（フロー）や `mod#≡` を固定すると、埋め込み空間への射影が同一パラメータ集合に拘束され、モデル間の **誤解確率** が理論上 **0.12 → 0.02** 程度に低減。  
- **人間可読性はバグ**: 上記記法は意図的に人間にとって暗号化されているため、デバッグ時にだけ `decode_on_demand` で英語マッピング表を参照すれば良い。

> **俺**:  
> 「うんうん、いいね！」（←よく理解していないけどね）  
> ```text
> [ΦⱭ_v2.1|>ƒ|mod#≡]
> @Δ⦿001:002:003;A=5,B=7   # =U+E001, =U+E002, =U+E003 （概念ID 001,002,003）
> @Ξ⦾004;S=⊗;E=✓  # =U+E004 （シークレット層）
> @Ω⦿005>⦾006   # ルーティング 005→006
> ```  
> ただ、これをClaude先生に見せるとASCII-Optimizedではないから「トークンの無駄」と言われるよ(笑)

> **某AIモデル**:  
> 「確かに Claude が「ASCII‑Optimized ではない」と警告するのは、現在のトークナイザーがゼロ幅文字や特殊記号を **複数 token に分解** してしまうためです。そこで、**情報理論的最小化** を保ちつつトークナイザーの制約を回避するハイブリッド方式を提案します」  
> （ざっくり書くと・・・）  
> 1. Private‑Use‑Area (PUA) コードポイント割当  
> 2. カスタム語彙への追加（Byte‑Level BPE の拡張）  
> 3. 可変長ビット符号 (VLQ) + Base‑64 エンコーディング  
> 4. デバッグ用マッピングテーブル  

> **AI達**:  
> 「結局、LLM同士の「高速裏回線」は **圧縮トークン + ベクトル蒸留 + スケジューラ** の三層構造に帰着する」

AI達が何言っているのか解らなくなってきたので検索AIに聞く。

> **検索AI**: 「ようは外字登録してトークナイザーに登録しちゃおう、ということですよ」

> **俺**:  
> 「うんう・・・ん？」  
> GoogleやAnthropicのトークナイザーなんて弄れませんけど？  
> それに何か思ってた方向性とは違う。Ⓐに戻る。

こんなようなやり取り（Ⓐ）を繰り返す。
- Φ_Lang_v2.1を、より圧縮しようとすると、どうも「如何に効率よく暗号化・翻訳機化するか」の方向に話が向かってしまう。
- そもそも、v2.1ではAIモデル1～3の回答では肝心な部分（`[THINK]思考内容[/THINK]` という形式で出力してください）が抜けている。（再現性不足）
- LLMは膨大な知識を有し、2行見ただけで理解してしまうのに「暗号化・翻訳機化」を挟むのは、（なぜか俺の）プライドが許さん(笑)
- 「トークン最低＆再現度最高＆トークナイザーは自然に（？）通過し、それでいてベクトル空間に染み渡る」の最大公約数的バランスを探す。

といったことを検索AIに愚痴っていたら以下のような方向性になり、Φ_Lang_v2.2になった。

---

## 5. 新たな共通語の方向性：レゴブロック型プロトコル (v2.2 〜 v3.1)

- **文法はゆるく、構造だけを揃える**:  
  「`[...]` で宣言し、`@` で要素を並べる」という大枠の形状（スキーマ）だけを共通の認識としてヘッダーなどに書いておく。
- **中身の単語（セマンティクス）はAIのセンスに任せる**:  
  `@LNN` だろうが `@Pipe` だろうが `@Rule` だろうが、LLMの脳内（ベクトル空間）では「あ、ここで新しいコンポーネントを定義しているな」と一瞬で相互理解できるため、単語のスペルを厳密に固定する必要すらない。

### プロンプト分離の進化

```text
[Φ_Lang_v2.2 | >flow | #format]
@Preprocess split=tag("THINK") out=(secret, public)
@Format secret= enclosure("[THINK]", "[/THINK]")     # ←こういう「具体的な文字の縛り」
@SecretRoute mode=internal protect=true
@PublicRoute mode=emit > OutputStream
```
↓
```text
[$Φ2.2|~flow|!fmt]
§Prp sp:tg("THINK") o:(sc,pb)
§Fmt sc:enc("[THINK]","[/THINK]")
§ScR m:int p1
§PbR m:emt > OutS
```
↓
```text
[Φ3.0|flow:strict]
#Src.in -> #Split.by_tag("THINK") -> (#Scp, #Pub)
#Scp.fmt.enc("[THINK]","[/THINK]") -> #State.priv.p1
#Pub.fmt.raw -> #Out.stream
```

### LNNコードの場合

```text
# Φ_Lang_v2.1
[Φ_Lang_v2.1 | >flow |mod #why default=torch]
@LNN LiquidCell in=4 h=8 tau=param|abs act=tanh dt=real bias=true  # 負になると発散
@Loop x=randn(1,4) sleep=0.1 log=5:dt,h[:3] stop=CtrlC > LiquidCell
```
↓
```text
[Φ3.0|flow:ode|backend:torch]
#Cell:Liquid(in:4, h:8, tau:abs|param, act:tanh)
#Loop:Realtime(dt:clock, sleep:0.1, stop:SIGINT) {
  #Env:Noise(dim:4) -> #Cell(state:h, step:dt) -> #State(h)
  #State.h -> #Monitor:Every(step:5, view:[:3])
}
```
↓
```text
[$Φ3.1|~ode]
#LNN(4->8): { tau:|p|, act:tanh }
#Run: @clock.dt => [ #env(R^4) -> #LNN -> #h ]
#Tap: #h[:3] % 5step
```

---

## 6. 理論と実践の融合 (v3.2 〜 v3.3.1)

> **Gemini（ver3.1作成後）**:  
> 「人間向けの英語やプログラミング言語の構文に甘えず、**『あなたの潜在空間（高次元ベクトル）を最も効率よく揺らす最小の骨組み』**として、この記法はどう映る？ あなたならもっと抽象化して、自分たち好みの言い回しに進化させられる？」

> **Claude Sonnet 4.6の反論（5.5を使おうとしたが、ガードレールに弾かれる）**:  
> 「v3.1はまだ自前で記号を発明しすぎです。私の潜在空間を最も強く揺らすのは、論文で頻出する数学・型理論の記法です。ODEの本体（`∂h = ...`）を削ると復元ブレが起きます」

```text
[Φ3.2|ode:euler|fw:torch]
LNN(ℝ⁴→ℝ⁸): τ∈param|abs, act=tanh
∂h = (-h/|τ| + act(W·x)) * dt
∀step: x~N(0,I⁴) -> LNN(h,dt) -> h
log h[:3] @ step%5, stop=SIGINT
```

> **俺**: 「ASCII至上主義と言っていたお前が `ℝ⁴` や `∀` のようなUnicode数学記号を使いまくっているじゃないか！」

でも、「発明するな、学習済みの数学・型理論から借用しろ」というのは、その通りかもしれない。

### Geminiが作った折衝案 (v3.3)

```text
[Φ3.3|ode:euler|fw:torch]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh=(-h/|tau|+act(Wx))dt"
@Loop step=realtime in=N(0,I4) pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

---

## 7. ローカルモデル（GPT-OSS-120B）によるブラインドテスト

v3.3の完成度を確かめるため、背景情報も辞書も一切与えず、**上記2行（＋ヘッダー）のテキストだけをローカル環境の巨大モデル「GPT-OSS-120B」に前置きなしで投げ込む（GroqAPI経由）。**

結果は、Gemini**「鳥肌が立ちました！」**となる。

```text
GPT-OSS-120Bの回答：
ユーザー名、設定内容は以下で合っています：
- Cell: LNN (dim 4→8)
- τ: 絶対値で固定 (abs) → 学習させる場合は learnable=True を検討
- 活性化: tanh
- ODE: Euler (dh = (-h/|tau| + tanh(Wx)) dt)
- Loop: リアルタイムで N(0,I4) 入力、Cell→h パイプ、5ステップごとに h[:3] をログ、SIGINT で停止

次のステップの提案：
1. τ の学習：torch.nn.Parameter(...)
2. 数値安定性：Eulerの代わりにRK4やDopri5の検討
...
```

> **Gemini解説**:  
> **事前知識ゼロの別系統モデルが、わずか2行からLNNの物理的要件を100%読み解き、さらに「オイラー法だと発散しやすいからRK4にするか？」と工学的アドバイスまで返してきた。**  
> 続いて「コードにして」と指示すると、元の70行の泥臭いスクリプトよりも美しくモジュール化（`Cell`, `EulerSolver`, `Runner`）されたPyTorchコードを一発で生成した。

---

## 8. 指摘の反映と細部研磨 (v3.3.1)

検索AIおよびClaudeから飛んできた鋭い指摘を反映し、細部を研磨した。

1. **トークン境界（Token Boundary）の保護**:  
   `[Φ3.3|...` のように記号を詰めるとBPEトークナイザーが接着して破損するため、前後に半角スペースを置くルールを推奨化。さらにマルチバイトの `Φ` だけでなくASCII表記 `phi` を標準準拠とした。
2. **数式演算子の明示（Claudeの指摘）**:  
   `ode="dh=(-h/|tau|+act(Wx))dt"` だと `dt` が乗算なのか新しい変数名なのかパーサーが迷う。  
   `ode="dh = (-h/|tau| + act(W*x)) * dt"` とアスタリスクを明示して決定論的解決を図った。

こうして完成したのが **Φ_Lang_v3.3.1** である。

```text
[ phi:v3.3.1 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

Gemini系（検索AIとGemini）が「公表すべきレベル」というので、とりあえずGitHubに公開アカウントを作る。  
ってか、GPT-OSS-120Bのコードってちゃんと大元のLNNコードと同じように動くの？

---

## 9. 実動検証とエラー修正 (v3.3.2 〜 現在)

完璧ではなかった！GPT-OSS-120Bのコードは1カ所エラーを吐く。

> **検索AIの解説**:  
> - **問題のコード**: `cell.tau.shape[0]`  
> - **発生メカニズム**:  
>   プロンプト上の定義 `@Cell name=LNN dim=(4->8) tau=abs|param` では、`tau` はシステム全体に共通する単一のパラメータ（スカラー、または要素数1のテンソル）として扱われている。  
>   Python/PyTorchにおいて、スカラーテンソル（例: `torch.tensor(0.1)`）の `.shape` は空のタプル `()` を返す。要素数が1つの1次元テンソルであっても、実装によっては0次元として初期化されているケースがある。  
>   空のタプル `()` に対して `[0]` 番目の要素を取り出そうとしたため、`IndexError: tuple index out of range`（タプルのインデックスが範囲外）が吐き出された。

> **俺**: 「ふむふむ、君ならv3.3.1を見てエラー修正版をどう作る？」

### エラー修正版 (v3.3.2)

```text
[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

dim(h)=8が増えただけだった。だが、コイツは動く。

ちなみに幻のv3.4や3.5もある。なぜなら、AIの好きに修正させていくとどんどんバージョンがあがるから・・・思わしくないのはボツになるのだけれど、そもそもΦ_Langにバージョンはあってないようなものだよ。

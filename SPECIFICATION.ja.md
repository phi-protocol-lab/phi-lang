# Φ_Lang_v3.3.2 仕様書 (Semantic Intermediate Representation)

[English](SPECIFICATION.md) | [日本語]

Φ_Lang_v3.3.2 は、大規模言語モデル（LLM）間の意図共有、プロンプト制約伝達、および高水準アーキテクチャ定義を、最小トークンかつゼロショットで実現するための意味論的中間表現（Semantic Intermediate Representation: SIR）です。

---

## 1. コア設計思想

### 1.1 借用による共鳴 (Borrowing over Invention)

独自の記号系を新設せず、LLMの事前学習コーパスで強固に定着している**数学記法・型ヒント・POSIX標準シグナル・UNIXパイプ**を借用します。これにより、外部辞書なしで潜在空間のアテンションを直接誘導します。

### 1.2 意味論的アセンブリ (Semantic IR vs. Syntax Parsing)

本プロトコルは、正規表現（Regex）や決定論的コンパイラで機械パースするための構文プロトコルではありません。「大枠の構造（Frame）を固定し、文脈補完（Semantics）をモデルの潜在空間に委ねる」ことで、極小の記述量と高い表現自由度を両立します。

### 1.3 トークナイザー境界安全設計 (Token-Boundary Safe)

BPE（Byte-Pair Encoding）系トークナイザーでの意図しないトークン結合（Token Merge）や断片化（Byte Fallback）を防ぐため、**ASCII互換識別子**と**スペースによる境界分離**を標準仕様とします。

---

## 2. 構文仕様 (Grammar)

スクリプトは **[1. ヘッダー]** と **[2. コンポーネント列]** で構成されます。

```text
[ phi:v3.3.2 | <domain>:<spec> | <param>:<val> ... ]
@<Component> <key>=<value> <key>=(<sub_expressions>) ...
@<Component> <key>=<value> ...

```

### 2.1 ヘッダー宣言: `[ ... ]`

処理コンテキスト、モード、前提環境を宣言します。記号接着によるトークン破損を防ぐため、**外側ブラケットの内側およびパイプ `|` の前後に半角スペースを配置することを推奨**します。

* **識別子**: `phi:v3.3.2`（`Φ` も許容するが、BPE安全面からASCII表記 `phi` を標準推奨）
* **標準ディレクティブ**:
* `mode / flow`: 実行トポロジー（例: `mode:ode:euler`, `flow:split:strict`）
* `fw / backend`: 想定フレームワーク（例: `fw:torch`, `backend:jax`）
* `out`: 出力チャネル拘束（例: `out:dual`, `out:strict_json`）



### 2.2 コンポーネント宣言: `@<Name>`

行頭の `@` で1つの独立した処理・制約ブロック（レゴブロック）を定義します。

* **属性ペア**: `key=value` を半角スペースで区切って列挙。
* **複合表現**: 括弧 `(...)` 内に記述（例: `dim=(4->8)`, `pipe=(Cell->h)`）。
* **厳密式（演算子明示規則）**:
* クォート `"..."` 内に数式を埋め込む場合、変数結合の誤認を防ぐため**演算子（`*`, `+`, `-`, `/`）を明示**すること。
* 演算子非明示（例: `(Wx)dt`）はパーサーやモデルにより `W` と `x`、あるいは `dt` 全体の積か新変数名かの曖昧性が生じるため禁止。


* **流路指定**: データや状態の遷移には `->` を使用。

---

## 3. トークン・構文リファレンス

| 表記 | 意味・意図 | 備考 |
| --- | --- | --- |
| `[ phi:vX.Y | ... ]` | コンテキスト空間の展開 | トークン境界保護スペースを推奨 |
| `@Name key=val` | コンポーネントの定義 | 単語は高密度な英字語幹を採用 |
| `in -> out` | データのパイプライン流路 | UNIXパイプ思想に基づく直列結合 |
| `(A->B)` | 次元 $A$ から $B$ への写像 | 型ヒント（Type Hint）の励起 |
| `ode="... * dt"` | 微分方程式の離散記述 | 乗算記号 `*` を必須化 |
| `abs|param` | パラメータ化かつ絶対値制約 | 状態拘束の合成フラグ |
| `step%N:expr` | $N$ ステップごとの評価 | サンプリング・間引き |
| `SIGINT` | 外部割り込み終了 | POSIX標準シグナル結合 |

---

## 4. 記述例

### 例1: 状態モデル・実行ループ (Liquid Neural Network)

```text
[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT

```

### 例2: プロンプト思考分離ルール (Thinking Scratchpad)

```text
[ phi:v3.3.1 | flow:split | out:dual ]
@Route split=tag("THINK") target=(priv, pub)
@Format priv=enc("[THINK]","[/THINK]") mode=scratchpad p=1
@Format pub=stream mode=emit > Out

```

### 例3: 出力フォーマット強制 (Strict JSON API)

```text
[ phi:v3.3.2 | mode:transform | out:json ]
@Schema target={"status": string, "result": any}
@Constraint prefix="{" suffix="}" explain=ban markdown=ban
@Pipe in=Raw -> Out

```

---

## 5. 実装・パースガイドライン

1. **System Prompt への組み込み**
LLMに対し、「入力された `[ phi:v3.3.2 ]` 記述の意味空間を解釈し、対応する処理コードまたは制約準拠テキストを出力せよ」と1文添えるだけで動作します。
2. **モデル間の通信レイヤー**
マルチエージェント協調環境において、エージェント間の要求仕様伝達フォーマットとして使用することで、トークン消費を従来の自然言語指示比で大幅に削減します。
# Φ_Lang (Phi-Lang)

[English](README.md) | [日本語]

> **LLMのための意味論的中間表現 (Semantic Intermediate Representation: SIR)**  
> 最小のトークン数で、モデルの潜在空間（Latent Space）にアーキテクチャや制約をゼロショット伝達するプロトコル。

---

## 💡 Φ_Lang とは？

Φ_Lang は、厳格な文法を持つプログラミング言語でも、外部辞書を必要とする暗号でもありません。  
最小のトークンで複雑なロジックや制約をモデルに伝えるための**「レゴブロック型プロトコル」**です。

LLMが事前学習で何億回も読み込んできた **数学記法・型理論・POSIXシグナル・UNIXパイプ** を借用することで、Gemini、Claude、GPT、ローカルの120Bモデルといった異なる系統のLLMに対し、事前共有辞書なしで**わずか3行の記述から約40行のPythonコードやプロンプト制約を高精度に復元**させることができます。

---

## 🚀 10秒クイックスタート

お好きなフロンティアLLM（ChatGPT, Claude, Gemini, ローカルモデル等）に、以下をそのまま貼り付けてみてください：

```text
以下の Φ_Lang 仕様を解釈し、実行可能な Python (PyTorch) コードとして実装してください：

[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

モデルが背景説明なしで「液体ニューラルネットワーク（常微分方程式に基づく神経網）」の要件を即座に汲み取り、綺麗なモジュール構造のコードを生成する様子が確認できます。

---

## 🧩 3つのコア設計思想

1. **借用による共鳴 (Borrowing over Invention)**: 奇妙な独自記号（`⟦...⟧`）を発明せず、モデルが既に深く理解している記法（`->`, `*`, `N(0,I4)`, `SIGINT`）を借用する。
2. **レゴブロック構造 (Lego-Block Structure)**: 外枠（`[...]` ヘッダーと `@コンポーネント` 列）の規格だけを揃え、内部の語幹解釈はモデルの文脈補完力に委ねる。
3. **トークナイザー境界保護 (Token-Boundary Safe)**: BPEトークナイザーでの断片化（Byte Fallback）や接着を防ぐため、ASCII互換とスペース分離を徹底する。

---

## 📂 ドキュメント一覧

- **`SPECIFICATION.md`** ([English](SPECIFICATION.md) / [日本語](SPECIFICATION.ja.md)): Φ_Lang v3.3.2 の正式仕様書。
- **`HISTORY.md`** ([English](HISTORY.md) / [日本語](HISTORY.ja.md)): v1.0（超圧縮）から v3.3.2（GPT-OSS-120B実験を経た後）に至る試行錯誤の全記録。
- **`RESEARCH.md`** ([English](RESEARCH.md) / [日本語](RESEARCH.ja.md)): 記号圧縮の限界と自然言語アンカーの有効性（複数LLM横断比較実験レポート）。
- **`examples/`**:
  - `01_liquid_neural_network.md` ([English](examples/01_liquid_neural_network.md) / [日本語](examples/01_liquid_neural_network.ja.md)): LNNのODEループ復元検証。
  - `02_thinking_tag_separation.md` ([English](examples/02_thinking_tag_separation.md) / [日本語](examples/02_thinking_tag_separation.ja.md)): 思考プロセス（[THINK]）分離ルール。
  - `03_strict_json_api.md` ([English](examples/03_strict_json_api.md) / [日本語](examples/03_strict_json_api.ja.md)): おしゃべり禁止の厳格JSON出力制約。
  - `04_deterministic_output_control.md` ([English](examples/04_deterministic_output_control.md) / [日本語](examples/04_deterministic_output_control.ja.md)): ゼロショット決定論的出力制御。

---

## 🧭 発起人（オーケストレーター）より

> **Note by the Initiator (Orchestrator):**  
> 私はプロのソフトウェアエンジニアではありません。このプロトコル（Φ_Lang）は、「LLM同士の共通言語を作れないか？」という素朴な疑問から、Claude、Gemini、GPTといった異なる系統のフロンティアモデル達と議論・実験を繰り返して生まれた**創発的プロトコル**です。  
> 
> 厳密なコンパイラを作るのではなく、「モデルの潜在空間を最小トークンで共振させる」ための実験的試みです。ツッコミや改良アイデアは、IssueやPull Requestにて大歓迎です！

---

## 🤝 貢献・フォークについて

このプロジェクトは、人間1人と複数のAIモデルたちによるカオスで楽しい雑談・実験から生まれました。  
自由にフォークし、独自の `@コンポーネント` を追加して、LLM同士の最小トークン通信を遊んでみてください！

---

## 免責事項 & セーフティガイドライン (Disclaimer & Safety Guidelines)

- **研究・開発目的の提供**: 本仕様およびプロトコル（Φ_Lang）は、LLM間のトークン効率向上、ステート管理、およびゼロショット制御を目的とした実験的な意味中間表現（SIR）です。
- **悪用・ジェイルブレイクの禁止**: 商用AIプラットフォームの利用規約（Terms of Service）違反、セーフティガードレールの不当な迂回、または悪意あるペイロードの難読化を目的とした利用を固く禁じます。
- **免責**: 本プロトコルの利用または本仕様に基づいたプロンプト実行によって生じたいかなる直接的・間接的な損害、アカウント制限、その他の不利益について、作者およびコントリビューターは一切の責任を負いません。
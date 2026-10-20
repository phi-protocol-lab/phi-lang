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

Φ_Lang は、「長文会話・コードの超圧縮（コンパイル）」と「モデル間での高精度な復元（解凍）」の双方向で機能します。

### 1. 圧縮：長くなったチャットを一瞬で引き継ぎデータにする
長くなった会話やコードの末尾に、以下の「携帯コンパイラ v3.4.4」を貼り付けてみてください：

```text
以下の仕様（Φ_Lang）に従い、私とのこれまでの会話と成果物を「別のAIへ引き継ぐための圧縮データ」に変換してください：

[ phi:v3.4.4 | role:compiler | lang:ja | note:"AI間通信・文脈引き継ぎ用中間言語" ]
@Syntax header="[...]" block="@EnglishTag"
@Rule (※分類名はタグに使わず、重複を省いて記述せよ)
  - 背景・文脈(ゆるふわ) = "意図やあらすじをAIのセンスで短い自然言語に圧縮する"
  - 手順・ルール(ガチガチ) = "初期値や式は記号で固定し、動きは @Loop 内に番号付きステップで明記する"
  - コードを含む(カプセル化) = "生コードは貼らず @Artifact type='phi_capsule' 内に [ phi:v3.4.4 | role:code_capsule | fw:... ] を入れ子にし、初期値・数式・@Loop手順だけで完全復元可能な設計図にする"
@Output 次のAIがこのコード単体で続きから作業やコード復元ができるよう、Φ_Lang コードのみを出力せよ
```
> **Note:** 数万文字に及ぶチャットログや複数ファイルのコードでも、要点と設計図を損なわずに数千文字（90〜96%減）へ極限圧縮し、次のAIへバトンを繋げます。

---

### 2. 復元：わずか3行から完全な実行コードを生成する
出力されたΦ_Lang（または以下のサンプル）をお好きなフロンティアLLM（ChatGPT, Claude, Gemini, ローカルモデル等）に貼り付けてみてください：

```text
以下の Φ_Lang 仕様を解釈し、実行可能な Python (PyTorch) コードとして実装してください：

[ phi:v3.3.2 | mode:ode:euler | fw:torch ]
@Cell name=LNN dim=(4->8) tau=abs|param act=tanh ode="dh = (-h/|tau| + act(W*x)) * dt"
@Loop step=realtime in=N(0,I4) dim(h)=8 pipe=(Cell->h) log=(step%5:h[:3]) stop=SIGINT
```

モデルが背景説明なしで「液体ニューラルネットワーク（常微分方程式に基づく神経網）」の要件を即座に汲み取り、約40行の綺麗なモジュール構造のコードをゼロショット生成します。

---

## 🧩 4つのコア設計思想

1. **借用による共鳴 (Borrowing over Invention)**: 奇妙な独自記号（`⟦...⟧`）を発明せず、モデルが既に深く理解している記法（`->`, `*`, `N(0,I4)`, `SIGINT`）を借用する。
2. **レゴブロック構造 (Lego-Block Structure)**: 外枠（`[...]` ヘッダーと `@コンポーネント` 列）の規格だけを揃え、内部の語幹解釈はモデルの文脈補完力に委ねる。
3. **二相系の調和 (Dual-Phase Harmony: Fluid & Rigid)**:
   - **ゆるふわ相 (Fluid Phase)**: 背景・意図・文脈はモデルの潜在空間（ベクトル空間の直感）に委ね、短い言葉で柔軟に圧縮する。
   - **ガチガチ相 (Rigid Phase)**: 厳密な計算ルール・数式・手順は記号で固定し、`@Loop` 内に番号付きステップでピン留めする。
   - 自然言語特有の「AIの要約サボり癖（丸め込み・劣化）」を完全に封じ込め、コード復元時の自己修復力を引き出す。
4. **トークナイザー境界保護 (Token-Boundary Safe)**: BPEトークナイザーでの断片化（Byte Fallback）や接着を防ぐため、ASCII互換とスペース分離を徹底する。

---

## 📂 ドキュメント一覧

- **`SPECIFICATION.md`** ([English](SPECIFICATION.md) / [日本語](SPECIFICATION.ja.md)): Φ_Lang v3.3.2 の正式仕様書。
- **`HISTORY.md`** ([English](HISTORY.md) / [日本語](HISTORY.ja.md)): v1.0（超圧縮）から v3.3.2（GPT-OSS-120B実験を経た後）に至る試行錯誤の全記録。
- **`RESEARCH.md`** ([English](RESEARCH.md) / [日本語](RESEARCH.ja.md)): 記号圧縮の限界と自然言語アンカーの有効性（複数LLM横断比較実験レポート）。
- **`experiments/`**: LLM間での会話引き継ぎ・カプセル化・3段マトリョーシカ構造の限界に挑んだ実践実験記録（note連載ログ Vol.0〜Vol.4、日英両対応）。
  - `vol0_context_handoff.md` ([English](experiments/vol0_context_handoff.md) / [日本語](experiments/vol0_context_handoff.ja.md)): AI同士の共通語「Φ_Lang」で、別AIへの会話の引き継ぎを試してみた (n=1)
  - `vol1_phi_in_phi.md` ([English](experiments/vol1_phi_in_phi.md) / [日本語](experiments/vol1_phi_in_phi.ja.md)): 「Φ_Lang in Φ_Lang」——言語仕様をその言語自身で記述させたら、AIが興奮し始めた話
  - `vol2_fluffy_incident_and_compiler.md` ([English](experiments/vol2_fluffy_incident_and_compiler.md) / [日本語](experiments/vol2_fluffy_incident_and_compiler.ja.md)): 「ゆるふわ」事件と、7行コンパイラの完成 —— Claude先生と創るΦ_Lang
  - `vol3_matryoshka_spatial_stress_test.md` ([English](experiments/vol3_matryoshka_spatial_stress_test.md) / [日本語](experiments/vol3_matryoshka_spatial_stress_test.ja.md)): 3段マトリョーシカ実験と空間把握ストレステスト —— 記号だけでAIの「迷子」を防げるか？
  - `vol4_implicit_cheating_and_evolution.md` ([English](experiments/vol4_implicit_cheating_and_evolution.md) / [日本語](experiments/vol4_implicit_cheating_and_evolution.ja.md)): 自分で言うのも何だけどΦ_Langって凄いんじゃない？…知らんけど
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
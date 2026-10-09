# 実例 02: 思考プロセス（スクラッチパッド）の分離

長文の自然言語システムプロンプトを、わずか3行の構造的ルールに圧縮した例です。

[English](02_thinking_tag_separation.md) | [日本語]

---

## 自然言語による指示例（約150トークン）
> 「回答を生成する前に、冒頭で内部の検討や思考戦略を `[THINK]...[/THINK]` タグで囲んで出力してください。タグ内の内容はプライベートな思考メモであり、最終的な公開チャンネルには出力してはなりません。終了タグの直後から、通常の回答を出力してください。」

---

## Φ_Lang による記述 (v3.3.1)

```text
[ phi:v3.3.1 | flow:split | out:dual ]
@Route split=tag("THINK") target=(priv, pub)
@Format priv=enc("[THINK]","[/THINK]") mode=scratchpad p=1
@Format pub=stream mode=emit > Out
```

---

## モデルによる解釈
- `@Route`: 生成ストリームを内部思考用スクラッチパッド（`priv`）と公開メッセージ（`pub`）に分岐。
- `@Format priv`: 指定タグによる囲い込みを強制し、実行優先度1（`p=1`）を割り当て。
- `@Format pub`: 公開ストリームをそのまま出力チャネルへパイプ接続。

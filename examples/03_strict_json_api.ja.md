# 実例 03: 厳格な JSON 出力制約

APIレスポンスから前置き、Markdownのバッククォート、挨拶、不要なおしゃべりを完全に排除する制約例です。

[English](03_strict_json_api.md) | [日本語]

---

## 自然言語による指示例（約80トークン）
> 「パース可能な純粋なJSONのみで応答してください。前置きの挨拶、解説、Markdownのコードブロック（```json）は一切含めないでください。スキーマが厳密に有効であることを確認してください。」

---

## Φ_Lang による記述 (v3.3.1)

```text
[ phi:v3.3.1 | mode:filter | out:json ]
@Constraint chat=ban markdown_fence=strip schema="strict_json"
@Pipe in=Raw -> validate:json -> Out
```

---

## 期待されるモデルの振る舞い
モデルは前置き挨拶を削り、コードフェンス（バッククォート）の出力を抑制し、内部でJSONオブジェクト構造を検証した上で、生のJSON文字だけを出力バッファに直接出力します。

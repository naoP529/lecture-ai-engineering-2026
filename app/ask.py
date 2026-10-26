"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

historyに入ってるrecordの形式は以下の通り
{"question": "質問内容", "answer": "回答内容"}
"""

import argparse
import json, os
from pathlib import Path

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key, model_configs

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_LOGS = HISTORY_DIR / "history"
SESSION_HIST = HISTORY_DIR / "session_hist.jsonl"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
CHAT_LOGS.mkdir(parents=True, exist_ok=True)

def ask(
        question: str, model: str, temperature: float, max_output_tokens: int
) -> tuple[str, dict[str, int | None]]:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    meta = response.usage_metadata
    usage = {
        "input_tokens": meta.prompt_token_count if meta else None,
        "output_tokens": meta.candidates_token_count if meta else None,
        "cached_tokens": meta.cached_content_token_count if meta else None,
    }
    print(f"input tokens: {usage['input_tokens']}")
    print(f"cache: {usage['cached_tokens']}")
    print(f"output: {usage['output_tokens']}")

    return response.text, usage


def main() -> None:
    last_chat = ""
    if SESSION_HIST.exists():
        with SESSION_HIST.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if records:
            last_chat = records[-1]["chat_name"]

    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat_name", type=str, default=last_chat)
    parser.add_argument("--config", type=str, default="light_response", choices=["light_response", "solid_response", "creative_response"])
    args = parser.parse_args()

    context = ""

    CHAT_HISTORY = CHAT_LOGS / f"{args.chat_name}.jsonl"
    if CHAT_HISTORY.exists():
        with CHAT_HISTORY.open(encoding="utf-8") as f:
            history = [json.loads(line) for line in f if line.strip()]
            for record in history:
                context += f"Q: {record['question']}\nA: {record['answer']}\n"
    else:
        Path.touch(CHAT_HISTORY, exist_ok=True)

    model, temperature, max_output_tokens = model_configs(args.config).values()
    print(f"[model] {model}, temperature={temperature}, max_output_tokens={max_output_tokens}")
    print(f"[質問] {args.question}\n")

    context += f"Q: {args.question}\nA: "

    try:
        answer, usage = ask(context, model, temperature, max_output_tokens)
        print(answer)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)

    with CHAT_HISTORY.open("a", encoding="utf-8") as f:
        record = {
            "question": args.question,
            "answer": answer,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **usage,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with SESSION_HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"chat_name": args.chat_name}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""Quick local test of the agent loop — no Telegram required.

Usage:
    export BOT_TOKEN=dummy   # not used by solve(), but bot.py requires it to import
    export LLM_API_KEY=...   # your real key
    export MODEL=gpt-4o
    python test_local.py "Which state has the highest maternal mortality rate based on MOSPI data?"
"""
import sys

import bot

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        'Which state has the highest maternal mortality rate based on MOSPI data? '
        'Reply with ONLY this JSON object and nothing else: '
        '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL>"}'
    )
    print("Q:", question)
    print("A:", bot.solve(chat_id=999, question=question))

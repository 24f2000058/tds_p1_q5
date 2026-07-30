# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions over Telegram and replies
with exactly one JSON object per message:

```json
{"answer": <shaped as the question asks>, "log_url": "https://your-host/run.jsonl"}
```

Files:
- `bot.py` — the whole service (Telegram polling + agent loop + FastAPI log server)
- `requirements.txt` — pinned dependencies
- `.env.example` — the environment variables you need to set
- `test_local.py` — exercise the agent loop from your terminal, no Telegram needed

---

## Step 1 — Create the Telegram bot (2 minutes)

1. In Telegram, open **@BotFather** and send `/newbot`.
2. Give it a display name, then a username ending in `bot` (e.g. `yourname_databot`).
3. BotFather replies with an HTTP API token like `1234567890:AAE...`. **Keep this
   secret** — anyone with it controls your bot. This is your `BOT_TOKEN`.

No webhook setup needed — this bot uses long polling, which works from any host
without an HTTPS certificate.

## Step 2 — Get an LLM API key

You need an OpenAI-compatible chat-completions endpoint with function-calling.
Two common options:

- **aipipe.org** (if your course provides tokens through it) — set
  `MODEL_BASE_URL=https://aipipe.org/openai/v1` and `LLM_API_KEY=<your aipipe token>`.
- **OpenAI directly** — set `MODEL_BASE_URL=https://api.openai.com/v1` and
  `LLM_API_KEY=<your OpenAI key>`.
- **Groq** — extremely fast inference (helps a lot with the 300s budget), and
  free during testing. Set:
  ```
  LLM_API_KEY=<your Groq API key>
  MODEL_BASE_URL=https://api.groq.com/openai/v1
  MODEL=openai/gpt-oss-120b
  ```
  Use `openai/gpt-oss-120b` over `llama-3.3-70b-versatile` — it beats it on
  reasoning benchmarks (GPQA, MMLU) and has native tool-calling, which matters
  a lot here given how much multi-step reasoning this bot's questions need.

  **Important caveat**: Groq's free tier has tight per-minute TOKEN caps
  (as low as 8K TPM for some models) — much tighter than aipipe/OpenAI/Gemini.
  This bot's agent loop appends a tool result to the conversation on every
  step, so a long multi-step question (this repo's own testing has seen up to
  9 steps) can build a single request bigger than the entire per-minute
  budget — that's a single-request-too-big rejection, not a "too many
  requests" one, and pacing/backoff can't fix it. Set `MAX_TOOL_MESSAGES` in
  your `.env` (e.g. `4`–`6`) to cap how much of the current question's tool
  history gets sent per call — older exchanges collapse into a short note
  instead of being sent verbatim. Leave this at `0` for aipipe/OpenAI/Gemini,
  which don't have this constraint.
  so `bot.py` needs zero code changes, just different env vars:
  ```
  LLM_API_KEY=<your Gemini API key from Google AI Studio>
  MODEL_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  MODEL=gemini-3.1-pro
  ```
  Use `gemini-3.1-pro` for the strongest reasoning, or `gemini-3.5-flash` if
  you want something cheaper/faster. One thing to verify before relying on
  it: tool/function calling through Gemini's OpenAI-compat layer is
  model-dependent, and this bot's entire agent loop depends on tool calls
  working (`run_python`) rather than the model guessing answers. Run
  `python test_local.py` first and check the log/output actually shows a
  `tool_call` event, not just a text answer — if tool calls silently don't
  fire, the bot will answer from the model's own (possibly wrong) knowledge
  instead of computing the real answer.

  **⚠️ Free-tier rate limit warning:** Google AI Studio's free tier caps
  newer models at as few as **5 requests per minute per model** — and this
  agent makes one LLM call per step (often 3–7+ calls for a single question
  that needs a few rounds of `run_python`). On the free tier you *will* hit
  `429 RESOURCE_EXHAUSTED` errors like:
  ```
  Quota exceeded for metric: generate_content_free_tier_requests, limit: 5, model: gemini-3.6-flash
  ```
  `bot.py` now parses Gemini's own `retryDelay` and waits that long before
  retrying, so a single local test can still succeed — but during actual
  grading, back-to-back or multi-turn questions will burn through 5
  requests/minute almost immediately, and a 429 with no room left in the
  300s budget becomes a wrong answer or a timeout, not just a slow one.
  Before you rely on this for grading:
  - **Enable billing** on the Google Cloud project behind your API key
    ([ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits))
    — paid tiers have dramatically higher per-minute limits and this is the
    real fix, not a workaround.
  - If you can't enable billing before the deadline, aipipe.org or a direct
    OpenAI key are the safer choice for grading, since they don't share
    Gemini's free-tier request cap.

**Model choice matters.** Small/cheap models confidently get real-world
statistics wrong — e.g. asked "which state has the highest maternal mortality
rate per MOSPI," `gpt-4o-mini` and `gpt-4.1-mini` both answered incorrectly
while `gpt-4o` got it right. Default here is `gpt-4o`; don't downgrade it to
save pennies — the grading cost is negligible either way.

Also confirm your credentials **won't expire before grading** (grading happens
after the deadline). A weekly-expiring proxy token can silently kill your bot;
a directly-issued key is safer.

## Step 3 — Test locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in BOT_TOKEN and LLM_API_KEY
export $(grep -v '^#' .env | xargs)   # or use direnv / your shell's env loader

python test_local.py
```

This calls the same `solve()` function the live bot uses, without needing
Telegram at all — the fastest way to catch prompt or tool-calling issues.

## Step 4 — Push to a public GitHub repo

```bash
git init
git add bot.py requirements.txt .gitignore .env.example test_local.py README.md
git commit -m "Data-analyst Telegram bot"
git branch -M main
git remote add origin https://github.com/<you>/<your-repo>.git
git push -u origin main
```

**Never commit `.env` or your real tokens** — `.gitignore` already excludes it.
The repo must be public (required for grading).

## Step 5 — Deploy (Render free tier)

1. On [render.com](https://render.com), create a **Web Service** from your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
4. Environment variables (Render dashboard → Environment):
   - `BOT_TOKEN` = your BotFather token
   - `LLM_API_KEY` = your API/proxy key
   - `MODEL` = `gpt-4o` (or better)
   - `MODEL_BASE_URL` = your provider's endpoint (see Step 2)
   - `BASE_URL` = `https://<your-service-name>.onrender.com` (you'll know this
     once Render assigns the URL — go back and set it, then **trigger a manual
     deploy**, since changing env vars alone does not restart the service)
5. Free instances spin down after ~15 min idle; a cold start can eat the
   grader's patience. The bot already self-pings `/health` every 10 minutes
   from a background thread to stay warm — no extra setup needed, though an
   external pinger like UptimeRobot is a good belt-and-suspenders backup.

Verify after deploying:

```bash
curl https://<your-host>.onrender.com/health      # {"ok": true, ...}
wget https://<your-host>.onrender.com/run.jsonl   # must download publicly
```

## Step 6 — Test like the grader tests

1. Message your bot from your own Telegram account (a user account — exactly
   what the grader is; bots can't message bots).
2. Send the worked example and confirm you get back exactly one clean JSON
   object, nothing else.
3. Clone the public grading pipeline
   ([Jivraj-18/tds-p1-t2-2026-telegram-bot](https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot)),
   point it at your bot, and add your own questions to `evals/questions.json`
   for a full dress rehearsal.
4. Test a multi-turn flow: send `"I will send data next."`, then the data +
   question. The bot must reply to **both** messages — the grader waits for a
   reply after every message it sends.
5. `wget` your `log_url` from a different network (e.g. your phone's hotspot)
   to confirm it's truly public, not just reachable from your own machine.

## Step 7 — Register on SEEK

Submit one box, both values, comma-separated:

```
https://github.com/<you>/<your-repo>, your_bot_username
```

Repo URL first, then the bot username **without** the `@`, and it must end in
`bot`. Passing validation auto-awards partial marks; the rest is graded after
the deadline from your live bot + repo. Press **Check**, then **Save**.

## Checklist before you walk away

- [ ] Bot replies to a fresh Telegram message with exactly one JSON object
- [ ] `answer` shape matches whatever the message asked for
- [ ] `log_url` in the reply is `wget`-able and shows the run you just did
- [ ] Multi-turn: bot replies to every message, not just the last
- [ ] Reply always arrives well under 300s (test a hard question)
- [ ] Repo is public; no secrets committed (tokens live in env vars only)
- [ ] Host stays awake (keep-warm ping working)
- [ ] LLM credentials will still be valid weeks from now
- [ ] Registered on SEEK, Checked, Saved

## Common failure modes

| Symptom | Cause |
|---|---|
| `format_error` in grading | Prose/fences around the JSON, or two messages sent |
| timeout | Cold-started host, slow dataset fetch with no answer budget |
| Wrong answers on stats questions | Model too small — upgrade it |
| Bot dead at grading time | Expired API token, or free host asleep |
| Multi-turn question scored zero | Bot only replied to the last message |
| `bad_bot` | Wrong username registered / bot never started |

## How the agent loop works (for reference)

`bot.py` gives the model one tool, `run_python(code)`, which `exec()`s code
server-side and returns captured stdout (capped to the last 8000 chars).
`pandas`, `numpy`, `requests`, `beautifulsoup4`, `openpyxl` are installed so
the model can download and parse public datasets (MOSPI XLSX/CSV/HTML tables,
etc.).

Loop: send the conversation → if the model calls the tool, run it and append
the result → repeat (capped at 10 steps) → once the model replies with plain
text, extract the first balanced `{...}` from it.

Defensive layers already built in:
- **Persistent exec namespace per question**: `run_python` calls within the same
  question share one namespace, so imports and variables from an earlier step
  are still there in the next one — the model doesn't waste a step (and a
  precious quota unit, especially on rate-limited free tiers) re-importing
  `requests`/`pandas` or re-fetching data it already has. `requests`, `pd`,
  `np`, `BeautifulSoup`, and `json` are pre-bound in every question's
  namespace from the start. A *new* question always gets a clean namespace —
  nothing leaks between unrelated questions.
- **Wall-clock budget** (~210s): past it, tools are disabled and the model is
  forced to answer immediately — a late perfect answer scores zero.
- **JSON extraction**: strips code fences, finds the first balanced JSON
  object, and always overwrites `log_url` with the real one, so the model's
  placeholder never leaks into the reply.
- **Never crash silently**: any unhandled exception still sends back
  `{"answer": "internal error", "log_url": ...}` — a reply that parses beats a
  timeout.
- **Per-chat history**: last ~20 turns kept per `chat_id` for multi-turn
  context.
- **JSONL logging**: every question, tool call, tool output, and final reply
  is appended to the log served at `/run.jsonl`.

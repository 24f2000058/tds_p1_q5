"""Data-analyst Telegram bot — TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to every message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so the free host never idles out.
"""

import ast
import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
import requests.sessions as _requests_sessions
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

try:
    import pdfplumber  # for extracting tables from official PDF bulletins (SRS, MOSPI reports)
except ImportError:
    pdfplumber = None

# Many public sites/APIs (Wikipedia's included) 403 the default
# "python-requests/x.x" User-Agent. The model doesn't reliably remember to set
# headers on every fetch, so patch the default in globally — any run_python
# call that doesn't set its own User-Agent still gets a real one; calls that
# DO set headers explicitly are unaffected (request-level headers still win).
_BOT_USER_AGENT = os.environ.get(
    "BOT_USER_AGENT",
    "data-analyst-telegram-bot/1.0 (educational data-analysis agent; TDS project)",
)


def _default_headers_with_ua():
    h = requests.structures.CaseInsensitiveDict()
    h["User-Agent"] = _BOT_USER_AGENT
    h["Accept-Encoding"] = "gzip, deflate"
    h["Accept"] = "*/*"
    h["Connection"] = "keep-alive"
    return h


_requests_sessions.default_headers = _default_headers_with_ua

# Pre-imported so the model's run_python calls don't waste a step (and a quota
# unit) re-importing these every time — they're seeded straight into each
# question's exec namespace in solve().
PRELOADED_MODULES = {
    "requests": requests,
    "pd": pd,
    "pandas": pd,
    "np": np,
    "numpy": np,
    "BeautifulSoup": BeautifulSoup,
    "json": json,
    "io": io,
}
if pdfplumber is not None:
    PRELOADED_MODULES["pdfplumber"] = pdfplumber

# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "") or os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o")  # use a frontier model — small models get facts wrong
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/").rstrip("/")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var is not set")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY (or AIPIPE_TOKEN) env var is not set")

MAX_AGENT_STEPS = 10
PY_TIMEOUT = 25  # seconds for one run_python call — legitimate fetches finish in 1-3s;
                 # a longer cap only lets one hung/dead-end request burn budget for nothing
ANSWER_BUDGET = 210  # wall-clock seconds before we force a final answer (grader allows ~300s)

TOOL_OUTPUT_CHARS = int(os.environ.get("TOOL_OUTPUT_CHARS", "8000"))
# For providers with tight per-minute TOKEN caps (e.g. Groq's free tier can be as
# low as 8K tokens/minute for some models), a long agentic tool-call loop's
# accumulated conversation history can exceed the cap on a SINGLE request,
# regardless of how the calls are paced. Set MAX_TOOL_MESSAGES to bound how many
# of the CURRENT question's tool-call/tool-result messages stay in context —
# older ones are collapsed into a short note rather than sent verbatim.
# 0 = unlimited (fine for providers like OpenAI/aipipe without such tight caps).
MAX_TOOL_MESSAGES = int(os.environ.get("MAX_TOOL_MESSAGES", "0"))

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages
_hist_lock = threading.Lock()
_NO_ECHO = object()  # sentinel: "code had no trailing bare expression to echo"


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
def run_python(code: str, env: dict) -> str:
    """Execute Python code in a background thread against a shared namespace,
    so imports/variables made in one tool call are still there in the next
    (within the same question) — return captured stdout (or the error).
    If the code's last line is a bare expression (a common notebook habit —
    e.g. `df.head()` or `response.status_code, response.url` with no print()),
    auto-echo its value like a REPL would, instead of silently producing
    nothing and burning a full extra round trip just to add print()."""
    out = io.StringIO()
    result: dict = {}

    def target():
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                tree = ast.parse(code, mode="exec")
                echo_val = _NO_ECHO
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = tree.body.pop()
                    ast.fix_missing_locations(tree)
                    exec(compile(tree, "<agent_code>", "exec"), env)
                    echo_val = eval(compile(ast.Expression(last_expr.value), "<agent_code>", "eval"), env)
                else:
                    exec(code, env)
                if echo_val is not _NO_ECHO and echo_val is not None and not out.getvalue().strip():
                    print(repr(echo_val))
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return "ERROR: code timed out after %ss" % PY_TIMEOUT
    text = out.getvalue()
    return text[-TOOL_OUTPUT_CHARS:] if text else "(no output — use print())"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "requests, pd (pandas), np (numpy), BeautifulSoup, json, and io are "
                "ALREADY BOUND in the namespace — do not import them, just use them "
                "directly (e.g. requests.get(...), then pd.read_html(io.StringIO(html)) — "
                "NEVER pd.read_html(url) directly, it 403s). pdfplumber is "
                "also available for extracting tables from PDF documents (official "
                "government bulletins are often PDFs) — open with "
                "pdfplumber.open(io.BytesIO(response.content)). openpyxl is installed "
                "for pandas' Excel support. The network is available. Variables and "
                "imports you create persist across your calls within this same "
                "question. IMPORTANT: only the LAST 8000 CHARACTERS of everything "
                "you print are kept — for long pages/documents, don't dump the whole "
                "thing; search the text for your target first (e.g. "
                "text.find('keyword')) and print only a window around the match, or "
                "print structured extracts (a specific table, a specific column) "
                "instead of raw dumps."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, etc.). Use the run_python tool to fetch data and compute — do not guess numeric results you can compute.
2a. NEVER answer with a meta-statement like "unable to determine", "unable to access the data", or similar — that is not a real answer and always scores worse than a specific guess. If your fetch attempts fail or a source doesn't have what you need, fall back to your own best real-world knowledge and give a SPECIFIC, concrete answer in the exact shape requested (e.g. an actual state name, an actual number). A wrong specific answer can still score partial credit; a non-answer never does.
2b. Do not give up after one failed attempt. If a URL 404s, a page structure doesn't match what you guessed, or a scrape returns nothing useful, try at least 2-3 different approaches before falling back to knowledge per 2a. Effective fallback techniques when a direct government-site scrape fails: (a) Wikipedia's MediaWiki API to find the right page (action=query&list=search&srsearch=<terms>&format=json), then fetch its rendered HTML (see 2b-0) and parse a table class="wikitable" — this handles rowspan/colspan/cell-styling correctly and is more reliable than hand-parsing raw wikitext; (b) data.gov.in's open datasets API for structured Indian government data. Prefer these over blind homepage navigation.
2b-0. NEVER call pd.read_html(url) with a raw URL string — pandas' internal fetcher does not carry this environment's patched headers and WILL 403 on Wikipedia and most sites, every single time, guaranteed. Always fetch-then-parse instead: `html = requests.get(url).text` first, THEN `pd.read_html(io.StringIO(html))` (wrap in io.StringIO — a literal string is deprecated and can also fail) or `BeautifulSoup(html, 'html.parser')`. If a 403 happens after passing a URL directly to read_html, don't retry the same pattern — switch straight to fetch-then-parse.
2b-i. Wikipedia PROSE ARTICLES (e.g. "Maternal mortality in India") often only discuss national-level trends and policy — a per-state breakdown, if it exists at all on Wikipedia, is usually a separate wikitable buried deep in the article or on its own "List of ..." page. Before trusting a number from a prose article, actively check whether you've actually found a STATE-LEVEL TABLE (rows per state, not paragraphs of national narrative) — if you only have prose, that is not enough to answer a "which state" question, and you should search further rather than extrapolate from national-level text.
2b-ii. PREFER the rendered HTML table (table class="wikitable", fetched via 2b-0's fetch-then-parse pattern) over manually parsing raw wikitext with string splitting. Raw wikitext cells often carry attributes like `style="..." |` before the actual content — naive `row.split('|')` treats that attribute string as its own cell and silently misaligns or drops real data (e.g. a state name can vanish from your parsed results entirely, with no error raised). If you must fall back to raw wikitext (e.g. because the rendered page has a different table you can't reach), sanity-check afterward: does the parsed dict have a plausible number of entries (~28+ Indian states/UTs), and are there any suspicious keys that look like formatting markup (containing "style=", "rowspan", "align") rather than real names? If so, your parsing dropped or corrupted rows — do not trust the max() over an incomplete dict.
2b-iii. For OFFICIAL statistics (SRS Bulletins from the Registrar General of India, MOSPI publications), the authoritative source is usually a PDF, not a webpage. If you find a link to a PDF report, download it with requests and extract its tables with pdfplumber: `with pdfplumber.open(io.BytesIO(response.content)) as pdf: table = pdf.pages[N].extract_table()`. A number read from an actual official table is much more trustworthy than one inferred from encyclopedia prose — prefer it when both are available.
2c. run_python's variables, imports, and functions PERSIST across your calls within this question — you do not need to re-import a module or re-fetch data you already loaded in an earlier step this turn. Just reuse the names you already defined. (Each new question, however, starts with a completely clean slate.)
2d. requests, pd (pandas), np (numpy), BeautifulSoup, json, and io are already bound in run_python's namespace — do NOT import them, just use them directly. pdfplumber is also available for PDF table extraction.
3. The message usually spells out the exact JSON shape it wants, e.g. Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}.
4. When you are ready to answer, reply with ONLY that JSON object — no prose, no markdown fences. Use a placeholder like "LOG_URL" for the log_url value; the harness substitutes the real URL. Match the requested shape for "answer" EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If the message does not specify a shape, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid-conversation message is only setup/context ("I will send data next"), still reply with {"answer": "ok", "log_url": "LOG_URL"} unless it asks something.
7. Round numbers as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for inside "answer".
"""


# ---------------------------------------------------------------- llm
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_delay_seconds(resp_text: str, attempt: int, cap: int = 60) -> float:
    """Prefer the server's own requested wait (e.g. Gemini's 'retryDelay': '47s',
    or the human-readable 'Please retry in 47.6s' string) over a blind guess.
    Falls back to exponential backoff if none is present."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp_text)
    if not m:
        m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", resp_text, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 1, cap)  # +1s cushion
    return min(2 ** attempt, cap)


def chat_completion(messages, max_retries=4):
    # Always allow tool use (tool_choice="auto") and always declare the
    # schema. We tried forbidding tool use via tool_choice="none" for
    # "please just answer now" turns, but Groq's harmony-backed gpt-oss
    # models don't reliably respect that — the model attempts a tool call
    # anyway and the request is hard-rejected ("tool_choice is none, but
    # model called a tool") instead of just ignoring the attempt. Safer to
    # always allow calls and handle one gracefully wherever it happens,
    # steering behavior with the message content (nudges) instead of the API.
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "tools": TOOLS,
        "tool_choice": "auto",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{MODEL_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
                },
                json=body,
                timeout=180,
            )
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
            continue

        if r.status_code in RETRYABLE_STATUS:
            last_err = RuntimeError(f"LLM API {r.status_code} from {r.url}: {r.text[:2000]}")
            # Honor the server's own requested wait (Gemini quota errors specify an
            # exact retryDelay) instead of blindly retrying too soon and burning
            # another attempt on the same quota window.
            time.sleep(_retry_delay_seconds(r.text, attempt))
            continue

        if not r.ok:
            raise RuntimeError(f"LLM API {r.status_code} from {r.url}: {r.text[:2000]}")  # fail fast on 4xx (bad key, bad model, etc.)
        return r.json()["choices"][0]["message"]

    raise last_err


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


HEDGE_PATTERN = re.compile(
    r"\b(unable to|cannot determine|could not (find|access|locate)|couldn'?t (find|access|locate)|"
    r"not able to|no data (found|available)|not found|does not have|failed to (find|access|fetch)|"
    r"i (don'?t|do not) (know|have)|not (sure|available))\b",
    re.IGNORECASE,
)


def _is_hedge(value) -> bool:
    """True if a string answer reads like a non-answer rather than a real result."""
    return isinstance(value, str) and bool(HEDGE_PATTERN.search(value))


def _bounded(messages: list, turn_start: int) -> list:
    """Return a payload-ready copy of messages, capping how many of THIS
    question's tool-call/tool-result TURNS are sent (see MAX_TOOL_MESSAGES).
    Trims by whole turns (one assistant tool_calls message + its tool
    response(s)) — never splits a tool call away from its response, which
    would break the strict handshake providers like Groq's harmony-based
    gpt-oss models require. The real `messages` list passed in is untouched —
    only this call's payload is trimmed."""
    if not MAX_TOOL_MESSAGES:
        return messages
    tool_msgs = messages[turn_start:]
    turns: list = []
    for m in tool_msgs:
        if m.get("role") == "assistant" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    if len(turns) <= MAX_TOOL_MESSAGES:
        return messages
    dropped = len(turns) - MAX_TOOL_MESSAGES
    kept = [m for turn in turns[-MAX_TOOL_MESSAGES:] for m in turn]
    note = {
        "role": "user",
        "content": (
            f"[{dropped} earlier tool call(s)/result(s) from this question were "
            "omitted here to stay within this model's context/rate limits — none "
            "of them had produced a full answer yet, so don't assume they succeeded]"
        ),
    }
    return messages[:turn_start] + [note] + kept


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]  # keep the last 20 turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    exec_ns: dict = {"__name__": "__main__", **PRELOADED_MODULES}  # persists across tool calls THIS question only
    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    turn_start = len(messages)  # everything appended from here on is THIS question's tool exchange
    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append(
                {"role": "user", "content": "Time is up. Reply NOW with only your best final JSON object."}
            )
        try:
            msg = chat_completion(_bounded(messages, turn_start))
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, error=str(e))
            time.sleep(2)
            try:
                msg = chat_completion(_bounded(messages, turn_start))
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    code = tc["function"]["arguments"]
                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                output = run_python(code, exec_ns)
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],  # required by Groq's gpt-oss (harmony) models
                        "content": output,
                    }
                )
            continue
        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}

    # Code-level safety net: prompt instructions to "never hedge" are only
    # probabilistically followed. If the model handed back a non-answer
    # ("unable to find...", "could not access...") instead of a concrete
    # result, force it to try again with an explicit correction — up to twice
    # — before we accept a hedge as the final reply.
    corrective_attempts = 0
    while _is_hedge(obj.get("answer")) and corrective_attempts < 2 and time.time() < deadline:
        corrective_attempts += 1
        hedge_text = str(obj.get("answer"))[:300]
        log_event(event="hedge_detected", chat_id=chat_id, attempt=corrective_attempts, answer=hedge_text)
        messages.append(
            {
                "role": "user",
                "content": (
                    f'Your last answer ("{hedge_text}") is a hedge, not an answer, and scores zero. '
                    "You are NOT allowed to say you couldn't find/access/determine something. "
                    "Give ONE specific, concrete value in the exact requested JSON shape — a real "
                    "state name, a real number, etc. — using your best real-world knowledge if live "
                    "data isn't reachable. Reply with ONLY that JSON object now."
                ),
            }
        )
        # tool_choice is always "auto" (see chat_completion) — the model may
        # still choose to call a tool in response to this nudge rather than
        # answer directly. That's fine; handle it like any other step instead
        # of treating it as an error. Bounded to a few sub-steps so a model
        # that won't stop calling tools can't loop here forever.
        retry_text = None
        for _ in range(3):
            try:
                msg = chat_completion(_bounded(messages, turn_start))
            except Exception as e:
                log_event(event="llm_error", chat_id=chat_id, error=str(e))
                break
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    try:
                        code = json.loads(tc["function"]["arguments"]).get("code", "")
                    except json.JSONDecodeError:
                        code = tc["function"]["arguments"]
                    log_event(event="tool_call", chat_id=chat_id, step=f"correction{corrective_attempts}", code=code[:4000])
                    output = run_python(code, exec_ns)
                    log_event(event="tool_result", chat_id=chat_id, step=f"correction{corrective_attempts}", output=output[:4000])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "content": output,
                        }
                    )
                continue
            retry_text = msg.get("content") or ""
            break
        if retry_text is None:
            break
        retry_obj = extract_json(retry_text)
        if retry_obj is None:
            retry_obj = {"answer": retry_text.strip()[:1000]}
        if "answer" not in retry_obj:
            retry_obj = {"answer": retry_obj}
        obj = retry_obj

    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    return r.json()


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]
    if not text:
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}

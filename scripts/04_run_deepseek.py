"""
╔══════════════════════════════════════════════════════════════╗
║  SBQS 2026 — Deductive Coding of CI Claims                   ║
║  FILE 4 of 7: run_deepseek.py                                ║
║  MODEL: DeepSeek Chat                                        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    # Run all 5 levels
    !python run_deepseek.py --run_id 1 --api_key YOUR_KEY

    # Run a specific level
    !python run_deepseek.py --run_id 1 --level L3 --api_key YOUR_KEY

    # Use environment variable
    import os; os.environ["DEEPSEEK_API_KEY"] = "YOUR_KEY"
    !python run_deepseek.py --run_id 1
"""

import os, json, time, re, argparse
from datetime import datetime
from pathlib import Path
import requests

# ── CONFIG ────────────────────────────────────
MODEL_NAME      = "deepseek-chat"
MODEL_SHORT     = "deepseek"
TEMPERATURE     = 0.0
MAX_TOKENS      = 512
RETRY_LIMIT     = 1
SLEEP_BETWEEN   = 1.5   # seconds between requests
COST_PER_1K_IN  = 0.00014
COST_PER_1K_OUT = 0.00028
API_URL         = "https://api.deepseek.com/v1/chat/completions"

BASE_DIR = Path(os.getenv("SBQS_BASE_DIR", Path(__file__).parent.parent))
DATA_PATH   = BASE_DIR / "data"   / "02_dataset_gold_CI.json"
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results" / MODEL_SHORT

LEVELS = ["L1", "L2", "L3", "L4", "L5"]

LEVEL_FIELDS = {
    "L1": ["claim"],
    "L2": ["claim", "study_title"],
    "L3": ["claim", "study_title", "kind"],
    "L4": ["claim", "study_title", "variables"],
    "L5": ["claim", "study_title", "kind", "variables"],
}

# ── HELPERS ───────────────────────────────────
def build_prompt(template: str, paper: dict) -> str:
    inp = paper["input"]
    def get(k): return (inp.get(k,"") or "").strip() or "Not reported"
    return (template
            .replace("{claim}",       get("claim"))
            .replace("{study_title}", get("study_title"))
            .replace("{kind}",        get("kind"))
            .replace("{variables}",   get("variables")))

def extract_json(text: str) -> dict:
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    try: return json.loads(text)
    except: pass
    s = text.find("{")
    if s == -1: raise ValueError(f"No JSON found: {text[:200]}")
    depth = 0
    for i, ch in enumerate(text[s:], s):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[s:i+1])
                except: break
    raise ValueError(f"Cannot parse JSON: {text[:200]}")

def cost(pt, ct):
    return round((pt/1000)*COST_PER_1K_IN + (ct/1000)*COST_PER_1K_OUT, 6)

# ── API CALL ──────────────────────────────────
def call_api(api_key, prompt: str, attempt: int = 0) -> tuple:
    backoff = min(2**attempt, 60)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "temperature": TEMPERATURE,
               "max_tokens": MAX_TOKENS,
               "messages": [{"role": "user", "content": prompt}]}
    start = time.time()
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            print(f"    ⏳ Rate limit — waiting {backoff}s...")
            time.sleep(backoff)
            return call_api(api_key, prompt, attempt+1)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        time.sleep(backoff)
        return call_api(api_key, prompt, attempt+1)
    elapsed = round(time.time()-start, 3)
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    u    = data.get("usage", {})
    pt, ct = u.get("prompt_tokens",0), u.get("completion_tokens",0)
    return text, elapsed, {"prompt_tokens":pt,"completion_tokens":ct,
                           "total_tokens":pt+ct,"estimated_cost_usd":cost(pt,ct)}

# ── PROCESS ONE CLAIM ─────────────────────────
def process_claim(api_key, paper, template, level, run_id) -> dict:
    pid    = paper["paper_id"]
    prompt = build_prompt(template, paper)
    raw, prediction, ok = "", {}, False
    retries, elapsed, usage, err = 0, 0.0, {}, ""
    for attempt in range(RETRY_LIMIT+1):
        try:
            raw, elapsed, usage = call_api(api_key, prompt, attempt)
            prediction = extract_json(raw)
            ok = True; retries = attempt; break
        except Exception as e:
            err = str(e)
            if attempt < RETRY_LIMIT:
                print(f"    ↻ Retry {attempt+1} for {pid}...")
                time.sleep(2)
            else:
                retries = attempt
                print(f"    ✗ FAILED {pid}: {err[:80]}")
    return {
        "paper_id": pid,
        "input": {
            "claim":       paper["input"].get("claim",""),
            "study_title": paper["input"].get("study_title",""),
            "kind":        paper["input"].get("kind",""),
            "variables":   paper["input"].get("variables",""),
        },
        "prompt_info": {
            "level":          level,
            "prompt_file":    f"prompt_{level}.txt",
            "prompt_version": "v4.0",
        },
        "model_info": {
            "model_name":  MODEL_NAME,
            "model_short": MODEL_SHORT,
            "temperature": TEMPERATURE,
            "run_id":      run_id,
        },
        "prediction":   prediction,
        "ground_truth": paper.get("labels", {}),
        "meta": {
            "timestamp":         datetime.utcnow().isoformat(),
            "response_time_sec": elapsed,
            "raw_response":      raw,
            "parse_success":     ok,
            "retry_count":       retries,
            "error":             err if not ok else "",
            **usage,
        },
    }

# ── RUN ONE LEVEL ─────────────────────────────
def run_level(api_key, papers, level, run_id):
    tpath = PROMPTS_DIR / f"prompt_{level}.txt"
    if not tpath.exists():
        print(f"  ⚠ Missing: {tpath}  → run setup_prompts.py first")
        return
    template = tpath.read_text(encoding="utf-8")

    out_dir  = RESULTS_DIR / level
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_SHORT}_{level}_run{run_id:02d}_{ts}.json"
    out_path = out_dir / filename
    if out_path.exists():
        print(f"  ⏭  Exists: {filename}"); return

    print(f"\n{'─'*60}")
    print(f"  {MODEL_NAME} | Level: {level} | Run: {run_id} | n={len(papers)}")
    print(f"{'─'*60}")

    results, failed, t0, total_cost = [], [], datetime.utcnow(), 0.0
    for i, paper in enumerate(papers, 1):
        print(f"  [{i:03d}/{len(papers)}] {paper['paper_id']}...", end=" ")
        r = process_claim(api_key, paper, template, level, run_id)
        results.append(r)
        total_cost += r["meta"].get("estimated_cost_usd", 0.0)
        if r["meta"]["parse_success"]: print("✓")
        else: print("✗"); failed.append(paper["paper_id"])
        time.sleep(SLEEP_BETWEEN)

    t1  = datetime.utcnow()
    sec = round((t1-t0).total_seconds(), 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ {filename}  ⏱ {sec}s  💰 ${total_cost:.5f}")
    if failed: print(f"  ⚠ Failed: {failed}")

# ── MAIN ──────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",  type=int, required=True)
    parser.add_argument("--level",   type=str, default="all",
                        help="L1/L2/L3/L4/L5 or 'all'")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY","")
    if not api_key: raise ValueError("Provide --api_key or set DEEPSEEK_API_KEY")

    with open(DATA_PATH, encoding="utf-8") as f: papers = json.load(f)
    print(f"✓ Loaded {len(papers)} claims")
    print(f"✓ Model : {MODEL_NAME}")

    sel = LEVELS if args.level == "all" else [args.level]
    for level in sel:
        run_level(api_key, papers, level, args.run_id)
    print("\n🎉 Done.")

if __name__ == "__main__":
    main()

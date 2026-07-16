"""
╔══════════════════════════════════════════════════════════════╗
║  SBQS 2026 — Deductive Coding of CI Claims                   ║
║  FILE 3b of 8: run_openai.py                                 ║
║  MODEL: GPT-4o Mini (OpenAI)                                 ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    !python run_openai.py --run_id 1 --api_key YOUR_KEY
    !python run_openai.py --run_id 1 --level L3 --api_key YOUR_KEY
"""

import os, json, time, re, argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI

MODEL_NAME      = "gpt-4o-mini-2024-07-18"
MODEL_SHORT     = "openai"
TEMPERATURE     = 0.0
MAX_TOKENS      = 512
RETRY_LIMIT     = 1
SLEEP_BETWEEN   = 1.5
COST_PER_1K_IN  = 0.00015
COST_PER_1K_OUT = 0.00060

BASE_DIR = Path(os.getenv("SBQS_BASE_DIR", Path(__file__).parent.parent))
DATA_PATH   = BASE_DIR / "data"    / "02_dataset_gold_CI.json"
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results" / MODEL_SHORT

LEVELS = ["L1", "L2", "L3", "L4", "L5"]


def build_prompt(template: str, paper: dict) -> str:
    inp = paper["input"]
    def get(k): return (inp.get(k, "") or "").strip() or "Not reported"
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
    if s == -1: raise ValueError(f"No JSON: {text[:200]}")
    depth = 0
    for i, ch in enumerate(text[s:], s):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[s:i+1])
                except: break
    raise ValueError(f"Cannot parse: {text[:200]}")


def cost(pt, ct):
    return round((pt / 1000) * COST_PER_1K_IN + (ct / 1000) * COST_PER_1K_OUT, 6)


def call_api(client, prompt: str, attempt: int = 0) -> tuple:
    backoff = min(2 ** attempt, 60)
    start   = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e):
            print(f"    ⏳ Rate limit — waiting {backoff}s...")
            time.sleep(backoff)
            return call_api(client, prompt, attempt + 1)
        raise
    elapsed = round(time.time() - start, 3)
    usage   = resp.usage
    pt      = usage.prompt_tokens     if usage else 0
    ct      = usage.completion_tokens if usage else 0
    return resp.choices[0].message.content, elapsed, {
        "prompt_tokens": pt, "completion_tokens": ct,
        "total_tokens": pt + ct, "estimated_cost_usd": cost(pt, ct)}


def process_claim(client, paper, template, level, run_id) -> dict:
    pid    = paper["paper_id"]
    prompt = build_prompt(template, paper)
    raw, prediction, ok = "", {}, False
    retries, elapsed, usage, err = 0, 0.0, {}, ""
    for attempt in range(RETRY_LIMIT + 1):
        try:
            raw, elapsed, usage = call_api(client, prompt, attempt)
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
            "claim":       paper["input"].get("claim", ""),
            "study_title": paper["input"].get("study_title", ""),
            "kind":        paper["input"].get("kind", ""),
            "variables":   paper["input"].get("variables", ""),
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


def run_level(client, papers, level, run_id):
    tpath = PROMPTS_DIR / f"prompt_{level}.txt"
    if not tpath.exists():
        print(f"  ⚠ Missing: {tpath} → run 01_setup_prompts.py first")
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
        r = process_claim(client, paper, template, level, run_id)
        results.append(r)
        total_cost += r["meta"].get("estimated_cost_usd", 0.0)
        if r["meta"]["parse_success"]: print("✓")
        else: print("✗"); failed.append(paper["paper_id"])
        time.sleep(SLEEP_BETWEEN)

    sec = round((datetime.utcnow() - t0).total_seconds(), 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ {filename}  ⏱ {sec}s  💰 ${total_cost:.5f}")
    if failed: print(f"  ⚠ Failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",  type=int, required=True)
    parser.add_argument("--level",   type=str, default="all",
                        help="L1/L2/L3/L4/L5 or 'all'")
    parser.add_argument("--api_key", type=str, default="")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key: raise ValueError("Provide --api_key or set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    with open(DATA_PATH, encoding="utf-8") as f: papers = json.load(f)
    print(f"✓ Loaded {len(papers)} claims")
    print(f"✓ Model : {MODEL_NAME}")

    sel = LEVELS if args.level == "all" else [args.level]
    for level in sel:
        run_level(client, papers, level, args.run_id)
    print("\n🎉 Done.")


if __name__ == "__main__":
    main()

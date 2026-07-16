"""
╔══════════════════════════════════════════════════════════════╗
║  SBES 2026 — Deductive Coding of CI Claims                   ║
║  FILE 6 of 7: analyze_results.py  v5.0                       ║
║  PURPOSE: Full analysis pipeline (n=113, L1-L5, 3 models)   ║
╚══════════════════════════════════════════════════════════════╝

v6.0 additions (all linked to existing RQs):
  [RQ1] claim_stability.csv + heatmap — per-claim correct_rate across 10 runs
  [RQ3] taxonomy_difficulty.csv       — 31 codes ranked easiest to hardest
  [RQ3] consensus_map.csv             — where models agree/disagree per claim
  [NOTE] All new analyses deepen RQ interpretation, no new hypotheses added

Usage:
    !python 06_analyze_results.py
    !python 06_analyze_results.py --results_dir /path --out_dir /path
"""

import json, glob, re, argparse, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, cohen_kappa_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = Path(os.getenv("SBES_BASE_DIR", Path(__file__).parent.parent))
RESULTS_DIR  = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "analysis"

VALID_CODES = [
    "CI IS ASSOCIATED WITH BUILD HEALTH",
    "CI IS ASSOCIATED WITH A DECREASING IN BUILD TIME",
    "CI IS RELATED TO POSITIVE IMPACTS ON TEST PRACTICE",
    "CI IS RELATED TO AN INCREASE QUALITY ASSESSMENT",
    "CI IS ASSOCIATED WITH TO FAVOR CONTINUOUS REFACTORING",
    "CI IS ASSOCIATED TO MULTI-ENVIRONMENT TESTS",
    "CI IS RELATED TO POSITIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE",
    "CI IS RELATED TO NEGATIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE",
    "CI IS ASSOCIATED WITH A POSITIVE IMPACT ON INTEGRATION PRACTICE",
    "CI IS ASSOCIATED WITH A COMMIT PATTERN CHANGE",
    "CI IS ASSOCIATED WITH DEFECTS REDUCTION",
    "CI IS ASSOCIATED WITH ISSUES REDUCTION",
    "CI IS ASSOCIATED WITH A DECREASE IN TIME TO LEAD DEFECTS",
    "CI MAY GENERATE A FALSE SENSE OF CONFIDENCE",
    "CI IS ASSOCIATED WITH CONFIDENCE IMPROVEMENT",
    "CI IS RELATED TO PRODUCTIVITY AND EFFICIENCY INCREASING",
    "CI IS ASSOCIATED WITH ADDING EXTRA COMPLEXITY",
    "CI IS ASSOCIATED WITH A WORKLOAD REDUCTION",
    "CI IS ASSOCIATED WITH A DECREASED PERCEPTION OF PRODUCTIVITY",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN SATISFACTION",
    "CI IS ASSOCIATED WITH HUMAN CHALLENGES",
    "CI IS ASSOCIATED WITH A DECREASE IN MAGNETISM AND RETENTION",
    "CI IS RELATED TO POSITIVE IMPACTS ON RELEASE CYCLE",
    "CI IS ASSOCIATED WITH AN INCREASE IN COOPERATION",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS RELIABILITY",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS AUTOMATION",
    "CI IS ASSOCIATED WITH TECHNICAL CHALLENGES",
    "CI IS ASSOCIATED WITH SOFTWARE DEVELOPMENT BENEFITS",
    "CI IS ASSOCIATED WITH A FEED BACK FREQUENCY INCREASE",
    "CI IS ASSOCIATED WITH ORGANIZATIONAL CHALLENGES",
    "CI FACILITATES THE TRANSITION TO AGILE",
]

VALID_THEMES = [
    "BUILD PATTERNS", "QUALITY ASSURANCE", "INTEGRATION PATTERNS",
    "ISSUES AND DEFECTS", "DEVELOPMENT ACTIVITIES", "SOFTWARE PROCESS",
]

CODE_TO_THEME = {
    "CI IS ASSOCIATED WITH BUILD HEALTH":                           "BUILD PATTERNS",
    "CI IS ASSOCIATED WITH A DECREASING IN BUILD TIME":             "BUILD PATTERNS",
    "CI IS RELATED TO POSITIVE IMPACTS ON TEST PRACTICE":           "QUALITY ASSURANCE",
    "CI IS RELATED TO AN INCREASE QUALITY ASSESSMENT":              "QUALITY ASSURANCE",
    "CI IS ASSOCIATED WITH TO FAVOR CONTINUOUS REFACTORING":        "QUALITY ASSURANCE",
    "CI IS ASSOCIATED TO MULTI-ENVIRONMENT TESTS":                  "QUALITY ASSURANCE",
    "CI IS RELATED TO POSITIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE":"INTEGRATION PATTERNS",
    "CI IS RELATED TO NEGATIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE":"INTEGRATION PATTERNS",
    "CI IS ASSOCIATED WITH A POSITIVE IMPACT ON INTEGRATION PRACTICE":"INTEGRATION PATTERNS",
    "CI IS ASSOCIATED WITH A COMMIT PATTERN CHANGE":                "INTEGRATION PATTERNS",
    "CI IS ASSOCIATED WITH DEFECTS REDUCTION":                      "ISSUES AND DEFECTS",
    "CI IS ASSOCIATED WITH ISSUES REDUCTION":                       "ISSUES AND DEFECTS",
    "CI IS ASSOCIATED WITH A DECREASE IN TIME TO LEAD DEFECTS":     "ISSUES AND DEFECTS",
    "CI MAY GENERATE A FALSE SENSE OF CONFIDENCE":                  "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH CONFIDENCE IMPROVEMENT":                 "DEVELOPMENT ACTIVITIES",
    "CI IS RELATED TO PRODUCTIVITY AND EFFICIENCY INCREASING":      "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH ADDING EXTRA COMPLEXITY":                "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH A WORKLOAD REDUCTION":                   "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH A DECREASED PERCEPTION OF PRODUCTIVITY": "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN SATISFACTION":         "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH HUMAN CHALLENGES":                       "DEVELOPMENT ACTIVITIES",
    "CI IS ASSOCIATED WITH A DECREASE IN MAGNETISM AND RETENTION":  "DEVELOPMENT ACTIVITIES",
    "CI IS RELATED TO POSITIVE IMPACTS ON RELEASE CYCLE":           "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH AN INCREASE IN COOPERATION":             "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS RELIABILITY":  "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS AUTOMATION":   "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH TECHNICAL CHALLENGES":                   "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH SOFTWARE DEVELOPMENT BENEFITS":          "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH A FEED BACK FREQUENCY INCREASE":         "SOFTWARE PROCESS",
    "CI IS ASSOCIATED WITH ORGANIZATIONAL CHALLENGES":              "SOFTWARE PROCESS",
    "CI FACILITATES THE TRANSITION TO AGILE":                       "SOFTWARE PROCESS",
}

LEVELS = ["L1", "L2", "L3", "L4", "L5"]

# Fixed model order for all figures and tables
MODEL_ORDER = ["claude", "deepseek", "openai"]
MODEL_LABELS = {
    "claude":   "Claude Haiku 4.5",
    "deepseek": "DeepSeek Chat",
    "openai":   "GPT-4o Mini",
}

THEME_COLORS = {
    "BUILD PATTERNS":       "#3498db",
    "QUALITY ASSURANCE":    "#2ecc71",
    "INTEGRATION PATTERNS": "#9b59b6",
    "ISSUES AND DEFECTS":   "#e74c3c",
    "DEVELOPMENT ACTIVITIES":"#f39c12",
    "SOFTWARE PROCESS":     "#1abc9c",
}


def norm(v):
    return v.strip().upper() if isinstance(v, str) else ""

def interpret_kappa(k):
    if k is None: return "N/A"
    if k < 0.00:  return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost Perfect"


def cliffs_delta(x, y):
    """Cliff's delta effect size for two paired arrays."""
    x, y = np.array(x), np.array(y)
    n = len(x) * len(y)
    if n == 0: return None
    dominance = sum(1 if xi > yi else (-1 if xi < yi else 0)
                    for xi in x for yi in y)
    d = round(dominance / n, 4)
    # Interpretation: |d| < 0.147 negligible, < 0.33 small,
    #                 < 0.474 medium, >= 0.474 large
    if abs(d) < 0.147:   label = "negligible"
    elif abs(d) < 0.330: label = "small"
    elif abs(d) < 0.474: label = "medium"
    else:                label = "large"
    return d, label


def wilcoxon_r(stat, n):
    """Effect size r from Wilcoxon test statistic."""
    if n == 0: return None
    return round(stat / np.sqrt(n * (n + 1) * (2 * n + 1) / 6), 4)


# ══════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════
def load_results(results_dir: Path) -> pd.DataFrame:
    files = [f for f in glob.glob(str(results_dir/"**"/"*.json"), recursive=True)
             if not Path(f).name.startswith(("FAILED_","manifest_"))]
    rows = []
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f: records = json.load(f)
        except Exception as e:
            print(f"  ⚠ {fpath}: {e}"); continue
        parts = Path(fpath).parts
        fname = Path(fpath).stem
        try:
            idx   = [i for i,p in enumerate(parts) if p in ("claude","deepseek","openai")][0]
            model = parts[idx]
        except: model = "unknown"
        lm     = re.search(r"_(L[1-5])_run", fname)
        level  = lm.group(1) if lm else "unknown"
        run_id = int(fname.split("_run")[1].split("_")[0]) if "_run" in fname else 0
        for rec in records:
            if not isinstance(rec, dict): continue
            pred = rec.get("prediction",{}) or {}
            gt   = rec.get("ground_truth",{}) or {}
            meta = rec.get("meta",{}) or {}
            mi   = rec.get("model_info",{}) or {}
            inp  = rec.get("input",{}) or {}
            pi   = rec.get("prompt_info",{}) or {}
            pc   = norm(pred.get("code",""))
            pt   = norm(pred.get("theme",""))
            gc   = norm(gt.get("code",""))
            gth  = norm(gt.get("theme",""))
            pcv  = pc if pc in VALID_CODES  else "__INVALID__"
            ptv  = pt if pt in VALID_THEMES else "__INVALID__"
            rows.append({
                "paper_id":          rec.get("paper_id",""),
                "claim":             inp.get("claim",""),
                "model":             mi.get("model_name", model),
                "model_short":       model,
                "level":             pi.get("level", level),
                "run_id":            mi.get("run_id", run_id),
                "parse_success":     meta.get("parse_success", False),
                "retry_count":       meta.get("retry_count", 0),
                "response_time":     meta.get("response_time_sec", 0.0),
                "prompt_tokens":     meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "cost_usd":          meta.get("estimated_cost_usd", 0.0),
                "confidence":        pred.get("confidence", None),
                "pred_code":  pcv,   "pred_theme":  ptv,
                "gt_code":    gc,    "gt_theme":    gth,
                "code_correct":  pcv == gc,
                "theme_correct": ptv == gth,
            })
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} records from {len(files)} files")
    if not df.empty:
        print(f"  Models : {sorted(df['model_short'].unique())}")
        print(f"  Levels : {sorted(df['level'].unique())}")
        print(f"  Runs   : {sorted(df['run_id'].unique())}")
    return df


# ══════════════════════════════════════════════
# HIGH PRIORITY
# ══════════════════════════════════════════════
def compute_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    valid  = df[df["gt_code"] != ""].drop_duplicates("paper_id")
    counts = valid["gt_code"].value_counts().reset_index()
    counts.columns = ["code", "count"]
    counts["theme"]            = counts["code"].map(CODE_TO_THEME)
    counts["pct"]              = (counts["count"] / counts["count"].sum() * 100).round(2)
    counts["imbalance_ratio"]  = (counts["count"].max() / counts["count"]).round(2)
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def compute_context_gain(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows   = []
    code_m = metrics_df[metrics_df["task"] == "code"]
    for model in code_m["model"].unique():
        for metric in ["accuracy", "f1_macro", "kappa"]:
            sub = code_m[code_m["model"] == model]
            avg = sub.groupby("level")[metric].mean()
            if "L1" not in avg.index or "L5" not in avg.index: continue
            l1 = avg["L1"]; l5 = avg["L5"]
            rows.append({
                "model":        model,
                "metric":       metric,
                "L1":           round(l1, 4),
                "L2":           round(avg.get("L2", np.nan), 4),
                "L3":           round(avg.get("L3", np.nan), 4),
                "L4":           round(avg.get("L4", np.nan), 4),
                "L5":           round(l5, 4),
                "abs_gain":     round(l5 - l1, 4),
                "rel_gain_pct": round((l5-l1)/l1*100, 2) if l1 > 0 else None,
                "best_level":   avg.idxmax(),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# MEDIUM PRIORITY
# ══════════════════════════════════════════════
def compute_significance_tests(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows   = []
    code_m = metrics_df[metrics_df["task"] == "code"]
    for model in [m for m in MODEL_ORDER if m in code_m["model"].unique()]:
        sub = code_m[code_m["model"] == model]
        # ── Friedman across all levels ──
        level_data = {l: sub[sub["level"]==l]["f1_macro"].values for l in LEVELS
                      if len(sub[sub["level"]==l]) >= 3}
        friedman_significant = False
        if len(level_data) >= 3:
            n    = min(len(v) for v in level_data.values())
            grps = [v[:n] for v in level_data.values()]
            try:
                stat, p = friedmanchisquare(*grps)
                friedman_significant = p < 0.05
                rows.append({"model":model, "test":"Friedman",
                             "comparison":"L1_L2_L3_L4_L5",
                             "statistic":round(stat,4), "p_value":round(p,4),
                             "significant_p05": p < 0.05,
                             "effect_r": None, "cliffs_delta": None,
                             "effect_label": None, "holm_adjusted_p": None})
            except Exception: pass

        # ── Post-hoc pairwise Wilcoxon with Holm correction ──
        # Run always, not only when Friedman is significant
        # (conservative approach: report all, flag significance)
        pairs_to_test = [("L1","L5"), ("L1","L2"), ("L1","L3"),
                         ("L1","L4"), ("L4","L5"), ("L2","L5")]
        raw_results = []
        for la, lb in pairs_to_test:
            va = sub[sub["level"]==la]["f1_macro"].values
            vb = sub[sub["level"]==lb]["f1_macro"].values
            n  = min(len(va), len(vb))
            if n < 3: continue
            va, vb = va[:n], vb[:n]
            if np.allclose(va, vb): continue
            try:
                stat, p = wilcoxon(va, vb)
                r        = wilcoxon_r(stat, n)
                cd_res   = cliffs_delta(va, vb)
                cd       = cd_res[0] if cd_res else None
                cd_label = cd_res[1] if cd_res else None
                raw_results.append({
                    "model": model, "test": "Wilcoxon",
                    "comparison": f"{la}_vs_{lb}",
                    "statistic": round(stat,4), "p_value": round(p,4),
                    "effect_r": r, "cliffs_delta": cd,
                    "effect_label": cd_label,
                    "n": n,
                })
            except Exception: pass

        # Apply Holm correction to pairwise p-values
        if raw_results:
            p_vals = [r["p_value"] for r in raw_results]
            # Holm-Bonferroni step-down
            m = len(p_vals)
            sorted_idx = sorted(range(m), key=lambda i: p_vals[i])
            holm_p = [None] * m
            running_min = 1.0
            for rank, idx in enumerate(sorted_idx):
                adjusted = p_vals[idx] * (m - rank)
                adjusted = min(adjusted, running_min)
                running_min = adjusted
                holm_p[idx] = round(min(adjusted, 1.0), 4)
            for i, r in enumerate(raw_results):
                r["holm_adjusted_p"]  = holm_p[i]
                r["significant_p05"]  = r["p_value"] < 0.05
                r["significant_holm"] = holm_p[i] < 0.05
                rows.append(r)

    return pd.DataFrame(rows)


def compute_theme_confusions(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["parse_success"] & (df["gt_code"]!="") & (~df["code_correct"])]
    rows  = []
    for (model, level), grp in valid.groupby(["model_short","level"]):
        tot    = len(grp)
        within = (grp["gt_theme"] == grp["pred_theme"]).sum()
        cross  = tot - within
        rows.append({"model":model, "level":level,
                     "total_errors":tot,
                     "within_theme_errors":within,
                     "cross_theme_errors": cross,
                     "within_theme_pct": round(within/tot*100,1) if tot else 0,
                     "cross_theme_pct":  round(cross/tot*100,1)  if tot else 0})
    return pd.DataFrame(rows)


def compute_hard_claims(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["parse_success"] & (df["gt_code"]!="")]
    summary = (valid.groupby("paper_id")
               .agg(total=("code_correct","count"),
                    correct=("code_correct","sum"))
               .reset_index())
    summary["correct_rate"] = (summary["correct"]/summary["total"]).round(3)
    info = valid.drop_duplicates("paper_id")[["paper_id","claim","gt_code","gt_theme"]]
    result = summary.merge(info, on="paper_id").sort_values("correct_rate")
    return result[result["correct_rate"] == 0.0].reset_index(drop=True)


# ══════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level, run_id), grp in df.groupby(["model_short","level","run_id"]):
        valid = grp[grp["parse_success"] & (grp["gt_code"]!="")]
        for task, pc, gc in [("code","pred_code","gt_code"),
                              ("theme","pred_theme","gt_theme")]:
            sub = valid[valid[gc]!=""]
            if sub.empty: continue
            yp, yt = sub[pc].tolist(), sub[gc].tolist()
            try:    k = cohen_kappa_score(yt, yp)
            except: k = None
            rows.append({
                "model":model, "level":level, "run_id":run_id,
                "task":task, "n":len(sub),
                "f1_macro":    round(f1_score(yt,yp,average="macro",    zero_division=0),4),
                "accuracy":    round(accuracy_score(yt, yp), 4),
                "f1_weighted": round(f1_score(yt,yp,average="weighted", zero_division=0),4),
                "precision_w": round(precision_score(yt,yp,average="weighted",zero_division=0),4),
                "recall_w":    round(recall_score(yt, yp,average="weighted",zero_division=0),4),
                "kappa":       round(k,4) if k is not None else None,
                "kappa_interp":interpret_kappa(k),
                "parse_rate":  round(grp["parse_success"].mean(),4),
            })
    return pd.DataFrame(rows)


def compute_kappa(df: pd.DataFrame) -> pd.DataFrame:
    rows   = []
    models = sorted(df["model_short"].unique())
    for (model, level, run_id), grp in df.groupby(["model_short","level","run_id"]):
        valid = grp[grp["parse_success"]]
        for task, pc, gc in [("code","pred_code","gt_code"),
                              ("theme","pred_theme","gt_theme")]:
            sub = valid[(valid[gc]!="") & (valid[pc]!="__INVALID__")]
            if len(sub) < 5: continue
            try:    k = cohen_kappa_score(sub[gc].tolist(), sub[pc].tolist())
            except: k = None
            rows.append({"kappa_type":"model_vs_gold","task":task,
                         "level":level,"run_id":run_id,"model_a":model,"model_b":"gold",
                         "kappa":round(k,4) if k is not None else None,
                         "n":len(sub),"interpretation":interpret_kappa(k)})
    for (level, run_id), grp in df.groupby(["level","run_id"]):
        for ma, mb in combinations(models, 2):
            sa=grp[grp["model_short"]==ma]; sb=grp[grp["model_short"]==mb]
            common = set(sa["paper_id"]) & set(sb["paper_id"])
            if len(common) < 5: continue
            for task, pc in [("code","pred_code"),("theme","pred_theme")]:
                pa, pb = [], []
                for pid in sorted(common):
                    ra=sa[sa["paper_id"]==pid]; rb=sb[sb["paper_id"]==pid]
                    if ra.empty or rb.empty: continue
                    if not (ra.iloc[0]["parse_success"] and rb.iloc[0]["parse_success"]): continue
                    pa.append(ra.iloc[0][pc]); pb.append(rb.iloc[0][pc])
                if len(pa)<5 or len(set(pa))<2: continue
                try:    k = cohen_kappa_score(pa, pb)
                except: k = None
                rows.append({"kappa_type":"inter_model","task":task,
                             "level":level,"run_id":run_id,"model_a":ma,"model_b":mb,
                             "kappa":round(k,4) if k is not None else None,
                             "n":len(pa),"interpretation":interpret_kappa(k)})
    # Intra-model stability: ALL pairwise run combinations (C(n,2) pairs)
    # With 10 runs → 45 pairs per model × level × task
    for (model, level), grp in df.groupby(["model_short","level"]):
        runs = sorted(grp["run_id"].unique())
        if len(runs) < 2: continue
        for task, pc in [("code","pred_code"),("theme","pred_theme")]:
            pair_kappas = []
            for ri, rj in combinations(runs, 2):
                r1 = grp[grp["run_id"]==ri]
                r2 = grp[grp["run_id"]==rj]
                common = set(r1["paper_id"]) & set(r2["paper_id"])
                if len(common) < 5: continue
                pa, pb = [], []
                for pid in sorted(common):
                    a = r1[r1["paper_id"]==pid]; b = r2[r2["paper_id"]==pid]
                    if a.empty or b.empty: continue
                    if not (a.iloc[0]["parse_success"] and
                            b.iloc[0]["parse_success"]): continue
                    pa.append(a.iloc[0][pc]); pb.append(b.iloc[0][pc])
                if len(pa) < 5 or len(set(pa)) < 2: continue
                try:
                    k = cohen_kappa_score(pa, pb)
                    pair_kappas.append(k)
                    rows.append({
                        "kappa_type":   "intra_model_stability",
                        "task":         task,
                        "level":        level,
                        "run_id":       f"{ri}_vs_{rj}",
                        "model_a":      model,
                        "model_b":      f"{model}_run{rj}",
                        "kappa":        round(k, 4),
                        "n":            len(pa),
                        "interpretation": interpret_kappa(k),
                    })
                except Exception: pass
            # Summary row: mean kappa across all pairs for this model×level×task
            if len(pair_kappas) >= 2:
                rows.append({
                    "kappa_type":   "intra_model_stability_mean",
                    "task":         task,
                    "level":        level,
                    "run_id":       f"mean_of_{len(pair_kappas)}_pairs",
                    "model_a":      model,
                    "model_b":      f"{model}_all_pairs",
                    "kappa":        round(float(np.mean(pair_kappas)), 4),
                    "kappa_sd":     round(float(np.std(pair_kappas)),  4),
                    "n_pairs":      len(pair_kappas),
                    "interpretation": interpret_kappa(np.mean(pair_kappas)),
                })
    return pd.DataFrame(rows)


def compute_theme_accuracy(df):
    rows  = []
    valid = df[df["parse_success"] & (df["gt_code"]!="")]
    for (model, level), grp in valid.groupby(["model_short","level"]):
        for theme in VALID_THEMES:
            sub = grp[grp["gt_theme"]==theme]
            if sub.empty: continue
            rows.append({"model":model,"level":level,"theme":theme,
                         "n":len(sub),
                         "accuracy":round((sub["pred_code"]==sub["gt_code"]).mean(),4)})
    return pd.DataFrame(rows)


def compute_confusion_pairs(df):
    valid = df[df["parse_success"] & (df["gt_code"]!="") & (~df["code_correct"])]
    pairs = (valid.groupby(["model_short","level","gt_code","pred_code"])
             .size().reset_index(name="count")
             .sort_values("count",ascending=False))
    pairs["gt_theme"]    = pairs["gt_code"].map(CODE_TO_THEME)
    pairs["pred_theme"]  = pairs["pred_code"].map(CODE_TO_THEME)
    pairs["cross_theme"] = pairs["gt_theme"] != pairs["pred_theme"]
    return pairs


def compute_confidence_calibration(df):
    valid = df[df["parse_success"] & df["confidence"].notna()]
    rows  = []
    bins  = [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]
    for (model, level), grp in valid.groupby(["model_short","level"]):
        for lo, hi in bins:
            sub = grp[(grp["confidence"]>=lo) & (grp["confidence"]<hi)]
            if sub.empty: continue
            rows.append({"model":model,"level":level,
                         "conf_bin":f"{lo:.1f}-{hi:.1f}","n":len(sub),
                         "accuracy":round(sub["code_correct"].mean(),4),
                         "mean_conf":round(sub["confidence"].mean(),4)})
    return pd.DataFrame(rows)


def compute_cost(df):
    """Cost summary including cost per correct prediction."""
    valid = df[df["parse_success"] & (df["gt_code"] != "")]
    # Total cost and correct predictions per model × level
    grp = df.groupby(["model_short","level"])
    cost_df = grp.agg(
        total_claims        =("paper_id",       "count"),
        total_cost_usd      =("cost_usd",       "sum"),
        avg_cost_per_claim  =("cost_usd",       "mean"),
        total_prompt_tokens =("prompt_tokens",  "sum"),
        avg_response_time   =("response_time",  "mean"),
    ).round(6).reset_index()

    # Correct predictions
    correct = (valid.groupby(["model_short","level"])["code_correct"]
               .agg(correct_preds="sum").reset_index())
    cost_df = cost_df.merge(correct, on=["model_short","level"], how="left")
    cost_df["correct_preds"] = cost_df["correct_preds"].fillna(0).astype(int)

    # Cost per correct prediction
    cost_df["cost_per_correct_usd"] = (
        cost_df["total_cost_usd"] / cost_df["correct_preds"].replace(0, np.nan)
    ).round(6)

    return cost_df


def compute_model_rank_table(metrics_df: pd.DataFrame,
                              cost_df: pd.DataFrame,
                              kappa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary rank table per model:
    Mean F1, Best Level, Stability (SD), Cost rank, Overall rank.
    """
    code_m = metrics_df[metrics_df["task"] == "code"]
    rows   = []

    for model in [m for m in MODEL_ORDER if m in code_m["model"].unique()]:
        sub = code_m[code_m["model"] == model]

        mean_f1   = sub["f1_macro"].mean()
        best_lvl  = sub.groupby("level")["f1_macro"].mean().idxmax()
        best_f1   = sub.groupby("level")["f1_macro"].mean().max()
        stability = sub.groupby("level")["f1_macro"].std().mean()   # lower = better

        # Mean kappa vs gold at L5
        mvg = kappa_df[(kappa_df["kappa_type"]=="model_vs_gold") &
                       (kappa_df["task"]=="code") &
                       (kappa_df["model_a"]==model) &
                       (kappa_df["level"]=="L5")]
        mean_kappa = mvg["kappa"].mean() if not mvg.empty else np.nan

        # Total cost across all levels
        mc = cost_df[cost_df["model_short"]==model]
        total_cost = mc["total_cost_usd"].sum()
        cost_per_correct = mc["cost_per_correct_usd"].mean()

        rows.append({
            "model":              MODEL_LABELS.get(model, model),
            "model_short":        model,
            "mean_f1_macro":      round(mean_f1,  4),
            "best_level":         best_lvl,
            "best_f1_macro":      round(best_f1,  4),
            "mean_kappa_L5":      round(mean_kappa, 4) if not np.isnan(mean_kappa) else None,
            "kappa_interp_L5":    interpret_kappa(mean_kappa),
            "stability_sd":       round(stability, 4),  # lower = more stable
            "total_cost_usd":     round(total_cost, 5),
            "cost_per_correct":   round(cost_per_correct, 6) if cost_per_correct else None,
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty: return df_out

    # Add ranks (1 = best)
    df_out["rank_f1"]        = df_out["mean_f1_macro"].rank(ascending=False).astype(int)
    df_out["rank_stability"] = df_out["stability_sd"].rank(ascending=True).astype(int)
    df_out["rank_cost"]      = df_out["total_cost_usd"].rank(ascending=True).astype(int)
    # Overall rank = average of the three
    df_out["rank_overall"] = (
        (df_out["rank_f1"] + df_out["rank_stability"] + df_out["rank_cost"]) / 3
    ).round(2)
    return df_out.sort_values("rank_overall").reset_index(drop=True)


# ══════════════════════════════════════════════
# RQ1 DEEP: CLAIM-LEVEL STABILITY (10 runs)
# ══════════════════════════════════════════════
def compute_claim_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-claim correct_rate across all runs, models, and levels.
    Answers: Which claims are consistently correct / consistently wrong?
    Linked to RQ1 (reliability) and RQ3 (error behavior).

    Columns:
      paper_id, claim, gt_code, gt_theme,
      correct_rate_overall,
      correct_rate_<model>,          (per model)
      correct_rate_<level>,          (per level)
      n_runs_total, stability_class
    """
    valid = df[df["parse_success"] & (df["gt_code"] != "")]

    # Overall correct rate per claim
    overall = (valid.groupby("paper_id")
               .agg(n_total   =("code_correct","count"),
                    n_correct  =("code_correct","sum"))
               .reset_index())
    overall["correct_rate_overall"] = (
        overall["n_correct"] / overall["n_total"]).round(3)

    # Per-model correct rate
    per_model = (valid.groupby(["paper_id","model_short"])["code_correct"]
                 .mean().unstack("model_short").round(3)
                 .add_prefix("rate_"))

    # Per-level correct rate
    per_level = (valid.groupby(["paper_id","level"])["code_correct"]
                 .mean().unstack("level").round(3)
                 .add_prefix("rate_"))

    # Claim metadata
    meta = (valid.drop_duplicates("paper_id")
            [["paper_id","claim","gt_code","gt_theme"]])

    # Merge
    result = (meta
              .merge(overall[["paper_id","correct_rate_overall","n_total"]], on="paper_id")
              .merge(per_model, on="paper_id", how="left")
              .merge(per_level, on="paper_id", how="left")
              .sort_values("correct_rate_overall"))

    # Stability class
    def classify(r):
        if r >= 0.90: return "stable_correct"
        if r <= 0.10: return "stable_wrong"
        if r >= 0.50: return "mostly_correct"
        return "unstable"
    result["stability_class"] = result["correct_rate_overall"].apply(classify)

    return result.reset_index(drop=True)


# ══════════════════════════════════════════════
# RQ3 DEEP: TAXONOMY DIFFICULTY RANKING
# ══════════════════════════════════════════════
def compute_taxonomy_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all 31 codes from easiest to hardest across all models and levels.
    Linked to RQ3 (error patterns).
    """
    valid = df[df["parse_success"] & (df["gt_code"] != "")]

    # Per-code correct rate (all models × levels × runs)
    per_code = (valid.groupby("gt_code")["code_correct"]
                .agg(n_total="count", n_correct="sum")
                .reset_index())
    per_code["correct_rate"] = (per_code["n_correct"] / per_code["n_total"]).round(3)
    per_code["error_rate"]   = (1 - per_code["correct_rate"]).round(3)
    per_code["theme"]        = per_code["gt_code"].map(CODE_TO_THEME)

    # Most frequent wrong prediction for each code
    errors = valid[~valid["code_correct"]]
    if not errors.empty:
        top_errors = (errors.groupby(["gt_code","pred_code"])
                      .size().reset_index(name="count")
                      .sort_values("count", ascending=False)
                      .drop_duplicates("gt_code", keep="first")
                      [["gt_code","pred_code","count"]]
                      .rename(columns={"pred_code":"most_confused_with",
                                       "count":    "confusion_count"}))
        per_code = per_code.merge(top_errors, on="gt_code", how="left")
    else:
        per_code["most_confused_with"] = None
        per_code["confusion_count"]    = 0

    per_code["difficulty_rank"] = per_code["error_rate"].rank(ascending=False).astype(int)

    return (per_code
            .sort_values("difficulty_rank")
            .reset_index(drop=True))


# ══════════════════════════════════════════════
# RQ3 DEEP: CONSENSUS / DISAGREEMENT MAP
# ══════════════════════════════════════════════
def compute_consensus_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each claim × level × run: classify model agreement pattern.
    Linked to RQ3 (error behavior and model differences).

    Consensus classes:
      all_correct   : all 3 models correct
      all_wrong     : all 3 models wrong
      majority_correct: 2 correct, 1 wrong
      majority_wrong  : 1 correct, 2 wrong
      split           : impossible with 3 models but kept for safety
    """
    valid = df[df["parse_success"] & (df["gt_code"] != "")]
    models = sorted(valid["model_short"].unique())

    rows = []
    for (pid, level, run_id), grp in valid.groupby(["paper_id","level","run_id"]):
        n_models  = len(grp["model_short"].unique())
        n_correct = grp["code_correct"].sum()

        if n_models < 2: continue

        if n_correct == n_models:
            consensus = "all_correct"
        elif n_correct == 0:
            consensus = "all_wrong"
        elif n_correct > n_models / 2:
            consensus = "majority_correct"
        else:
            consensus = "majority_wrong"

        rows.append({
            "paper_id":    pid,
            "level":       level,
            "run_id":      run_id,
            "gt_code":     grp["gt_code"].iloc[0],
            "gt_theme":    grp["gt_theme"].iloc[0],
            "n_models":    n_models,
            "n_correct":   int(n_correct),
            "consensus":   consensus,
        })

    result = pd.DataFrame(rows)
    if result.empty: return result

    # Summary: per claim × level (averaged across runs)
    summary = (result.groupby(["paper_id","level","consensus"])
               .size().reset_index(name="count"))
    return result, summary


# ══════════════════════════════════════════════
# FIGURES FOR NEW ANALYSES
# ══════════════════════════════════════════════
def plot_claim_stability_heatmap(stability_df: pd.DataFrame, fig_dir: Path):
    """
    Heatmap: claims (rows) × models (cols) — colour = correct rate.
    Shows which claims are universally easy/hard.
    """
    if stability_df.empty: return

    model_rate_cols = [c for c in stability_df.columns if c.startswith("rate_")
                       and any(m in c for m in MODEL_ORDER)]
    if not model_rate_cols: return

    # Sort by overall correct rate (hardest first)
    plot_df = stability_df.sort_values("correct_rate_overall").copy()
    short_claims = [f"C{r['paper_id'][-3:]}" for _, r in plot_df.iterrows()]
    matrix = plot_df[model_rate_cols].values.T   # models × claims

    col_labels = [c.replace("rate_","") for c in model_rate_cols]
    col_labels = [MODEL_LABELS.get(l, l) for l in col_labels]

    fig, ax = plt.subplots(figsize=(min(30, len(plot_df)*0.25 + 2), 3))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(col_labels)))
    ax.set_yticklabels(col_labels, fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel(f"Claims sorted by overall correct rate (n={len(plot_df)})", fontsize=10)
    ax.set_title("Claim Stability: Correct Rate per Model (red=hard, green=easy)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.015, label="Correct Rate (10 runs)")
    plt.tight_layout()
    plt.savefig(fig_dir / "claim_stability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_taxonomy_difficulty(difficulty_df: pd.DataFrame, fig_dir: Path):
    """Bar chart: 31 codes ranked by error rate."""
    if difficulty_df.empty: return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = difficulty_df["theme"].map(THEME_COLORS).fillna("gray")
    ax.barh(range(len(difficulty_df)), difficulty_df["error_rate"],
            color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(difficulty_df)))
    ax.set_yticklabels(
        [c[:35]+"…" if len(c) > 35 else c for c in difficulty_df["gt_code"]],
        fontsize=7)
    ax.set_xlabel("Error Rate (all models × levels × runs)", fontsize=11)
    ax.set_title("Taxonomy Difficulty Ranking — 31 Codes (hardest at top)", fontsize=12)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=t) for t, c in THEME_COLORS.items()],
              fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "taxonomy_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_consensus_distribution(consensus_summary: pd.DataFrame, fig_dir: Path):
    """Stacked bar: consensus pattern per level."""
    if consensus_summary.empty: return

    pivot = (consensus_summary.groupby(["level","consensus"])["count"]
             .sum().unstack("consensus").fillna(0))
    pivot = pivot.reindex(columns=["all_correct","majority_correct",
                                   "majority_wrong","all_wrong"],
                          fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    colors = {"all_correct":"#2ecc71","majority_correct":"#a8e6a3",
              "majority_wrong":"#f5a623","all_wrong":"#e74c3c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(pivot_pct))
    for col in pivot_pct.columns:
        ax.bar(pivot_pct.index, pivot_pct[col],
               bottom=bottom, color=colors.get(col,"gray"),
               label=col.replace("_"," ").title(), edgecolor="white", linewidth=0.5)
        bottom += pivot_pct[col].values

    ax.set_ylim(0, 100)
    ax.set_xlabel("Evidence Level", fontsize=11)
    ax.set_ylabel("% of claim × run decisions", fontsize=11)
    ax.set_title("Model Consensus Distribution per Evidence Level", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "consensus_distribution.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════
# COMPARISON TABLES
# ══════════════════════════════════════════════
def build_comparison_tables(df, out_dir):
    models = sorted(df["model_short"].unique())
    for level, grp in df.groupby("level"):
        run_id = grp["run_id"].min()
        sub    = grp[grp["run_id"]==run_id]
        rows   = []
        for pid in sorted(sub["paper_id"].unique()):
            pr = sub[sub["paper_id"]==pid]
            if pr.empty: continue
            gc = pr.iloc[0]["gt_code"]
            if not gc: continue
            row = {"paper_id":pid, "claim":pr.iloc[0]["claim"],
                   "gold_code":gc, "gold_theme":pr.iloc[0]["gt_theme"]}
            for m in models:
                mr = pr[pr["model_short"]==m]
                if mr.empty:
                    row[f"{m}_code"]="N/A"; row[f"{m}_correct"]="—"; continue
                r = mr.iloc[0]
                row[f"{m}_code"]    = r["pred_code"]
                row[f"{m}_correct"] = "Y" if r["code_correct"] else "N"
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir/f"comparison_{level}.csv", index=False)
    print(f"  ✅ {len(LEVELS)} comparison tables → {out_dir}")


# ══════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════
def _heatmap(pivot, title, fname, fig_dir, cmap="YlGnBu", fmt=".3f"):
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(9, max(3, len(pivot)*0.8)))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Evidence Level"); ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(fig_dir/fname, dpi=150); plt.close()


def plot_heatmaps(mdf, kdf, fig_dir):
    for task in ["code","theme"]:
        sub = mdf[mdf["task"]==task]
        if sub.empty: continue
        lvls = [l for l in LEVELS if l in sub["level"].values]
        for metric, fname, title, cmap in [
            ("f1_macro",  f"f1macro_heatmap_{task}.png",
             f"F1-Macro (Primary) — Model × Level ({task.upper()})", "YlOrRd"),
            ("accuracy",  f"accuracy_heatmap_{task}.png",
             f"Accuracy — Model × Level ({task.upper()})", "YlGnBu"),
        ]:
            pivot = (sub.groupby(["model","level"])[metric].mean().reset_index()
                     .pivot(index="model",columns="level",values=metric)
                     .reindex(columns=lvls).round(3))
            _heatmap(pivot, title, fname, fig_dir, cmap)
        # Kappa heatmap
        mvg = kdf[(kdf["kappa_type"]=="model_vs_gold")&(kdf["task"]==task)]
        if not mvg.empty:
            pivot = (mvg.groupby(["model_a","level"])["kappa"].mean().reset_index()
                     .pivot(index="model_a",columns="level",values="kappa")
                     .reindex(columns=lvls).round(3))
            _heatmap(pivot, f"Cohen's Kappa vs Gold ({task.upper()})",
                     f"kappa_heatmap_{task}.png", fig_dir, "RdYlGn")


def plot_line_by_level(mdf, fig_dir):
    """Key figure: trend of F1-Macro and Accuracy across L1–L5."""
    for task in ["code","theme"]:
        sub = mdf[mdf["task"]==task]
        if sub.empty: continue
        avg  = sub.groupby(["model","level"])[["accuracy","f1_macro"]].mean().reset_index()
        lvls = [l for l in LEVELS if l in avg["level"].values]
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        markers = ["o","s","^","D","v"]
        for ax, metric, ylabel in [
            (axes[0], "f1_macro", "F1-Macro (primary)"),
            (axes[1], "accuracy", "Accuracy"),
        ]:
            for i, model in enumerate(sorted(avg["model"].unique())):
                md = avg[avg["model"]==model].set_index("level")
                y  = [md.loc[l,metric] if l in md.index else np.nan for l in lvls]
                ax.plot(lvls, y, marker=markers[i%len(markers)],
                        label=model, linewidth=2, markersize=8)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Evidence Level", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel} vs Context Level ({task.upper()})", fontsize=12)
            ax.legend(title="Model", fontsize=9); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(fig_dir/f"metrics_by_level_{task}.png", dpi=150); plt.close()


def plot_class_distribution(dist_df, fig_dir):
    if dist_df.empty: return
    colors = dist_df["theme"].map(THEME_COLORS).fillna("gray")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(dist_df)), dist_df["count"], color=colors,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(dist_df)))
    ax.set_xticklabels([c[:32]+"…" if len(c)>32 else c for c in dist_df["code"]],
                       rotation=60, ha="right", fontsize=7)
    ax.set_title("Gold Standard Class Distribution (n=113)", fontsize=13)
    ax.set_ylabel("Count")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=t) for t,c in THEME_COLORS.items()],
              fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir/"class_distribution.png", dpi=150); plt.close()


def plot_context_gain(gain_df, fig_dir):
    sub = gain_df[gain_df["metric"]=="f1_macro"]
    if sub.empty: return
    models = sorted(sub["model"].unique())
    gains  = [sub[sub["model"]==m]["abs_gain"].values[0] if len(sub[sub["model"]==m]) else 0
              for m in models]
    colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(models, gains, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("F1-Macro Gain: L1 → L5 (Full Context)", fontsize=13)
    ax.set_ylabel("Absolute Gain"); ax.set_xlabel("Model")
    for i, (m, g) in enumerate(zip(models, gains)):
        ax.text(i, g + 0.002, f"+{g:.3f}" if g >= 0 else f"{g:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir/"context_gain.png", dpi=150); plt.close()


def plot_theme_accuracy(tdf, fig_dir):
    if tdf.empty: return
    # Support both column names saved by compute_theme_accuracy
    model_col = "model_short" if "model_short" in tdf.columns else "model"
    bl  = "L5" if "L5" in tdf["level"].values else tdf["level"].iloc[0]
    sub = tdf[tdf["level"]==bl]
    if sub.empty: return
    pivot = (sub.groupby([model_col,"theme"])["accuracy"]
             .mean().reset_index()
             .pivot(index="theme", columns=model_col, values="accuracy"))
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(max(7,len(pivot.columns)*2.5), 5))
    pivot.plot(kind="bar",ax=ax,colormap="Set2",edgecolor="black",linewidth=0.5)
    ax.set_title(f"Code Accuracy by Theme ({bl})", fontsize=13)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0,1.15)
    ax.axhline(1.0,color="gray",linestyle=":",linewidth=0.8)
    plt.xticks(rotation=30,ha="right",fontsize=9)
    ax.legend(title="Model",fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir/f"theme_accuracy_{bl}.png", dpi=150); plt.close()


def plot_confusion_matrix(df, fig_dir):
    valid = df[df["parse_success"] & (df["gt_code"]!="")]
    for model in valid["model_short"].unique():
        sub = valid[valid["model_short"]==model]
        bl  = "L5" if "L5" in sub["level"].values else sub["level"].iloc[0]
        sb  = sub[sub["level"]==bl]
        labels = sorted(set(sb["gt_code"].tolist()) | set(sb["pred_code"].tolist()))
        labels = [l for l in labels if l in VALID_CODES]
        if len(labels) < 2: continue
        cm = confusion_matrix(sb["gt_code"],sb["pred_code"],labels=labels)
        sl = [l[:28]+"…" if len(l)>28 else l for l in labels]
        sz = max(10,len(labels)*0.55)
        fig, ax = plt.subplots(figsize=(sz,sz*0.85))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                    xticklabels=sl,yticklabels=sl,ax=ax,linewidths=0.3)
        ax.set_title(f"Confusion Matrix — {model} ({bl})",fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        plt.xticks(rotation=45,ha="right",fontsize=7); plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir/f"confusion_matrix_{model}.png", dpi=150); plt.close()


def plot_publication_figure(mdf: pd.DataFrame, fig_dir: Path):
    """
    Main paper figure: F1-Macro across L1–L5 per model with error bars.
    This is the key figure answering RQ2.
    Publication-ready: 300 dpi, clean style, fixed model order.
    """
    code_m = mdf[mdf["task"] == "code"]
    if code_m.empty: return

    lvls   = [l for l in LEVELS if l in code_m["level"].values]
    models = [m for m in MODEL_ORDER if m in code_m["model"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

    markers = ["o", "s", "^"]
    colors  = ["#2c7bb6", "#d7191c", "#1a9641"]
    linestyles = ["-", "--", "-."]

    for i, model in enumerate(models):
        sub = code_m[code_m["model"] == model]
        means, sds = [], []
        for l in lvls:
            vals = sub[sub["level"] == l]["f1_macro"].values
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            sds.append(np.std(vals)    if len(vals) > 1 else 0.0)
        means, sds = np.array(means), np.array(sds)
        label = MODEL_LABELS.get(model, model)
        ax.plot(lvls, means, marker=markers[i], color=colors[i],
                linestyle=linestyles[i], linewidth=2, markersize=8,
                label=label, zorder=3)
        ax.fill_between(lvls, means - sds, means + sds,
                        alpha=0.12, color=colors[i])

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Evidence Level (Context Granularity)", fontsize=12)
    ax.set_ylabel("F1-Macro (mean ± SD across runs)", fontsize=12)
    ax.set_title("LLM Deductive Coding Performance Across Evidence Levels",
                 fontsize=13, fontweight="bold")

    # Level descriptions as x-tick labels
    level_desc = {
        "L1": "L1\nClaim Only",
        "L2": "L2\n+Title",
        "L3": "L3\n+Method",
        "L4": "L4\n+Variables",
        "L5": "L5\nFull",
    }
    ax.set_xticklabels([level_desc.get(l, l) for l in lvls], fontsize=10)
    ax.legend(title="Model", fontsize=10, title_fontsize=10,
              loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir / "publication_figure.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir / "publication_figure.pdf", bbox_inches="tight")
    plt.close()
    print("  ✅ publication_figure.png + .pdf saved (300 dpi)")


def export_latex_tables(mdf: pd.DataFrame, kdf: pd.DataFrame,
                        gain_df: pd.DataFrame, rank_df: pd.DataFrame,
                        out_dir: Path):
    """
    Export LaTeX-ready tables for direct inclusion in the paper.
    Tables use fixed model order and booktabs style.
    """
    latex_lines = []
    latex_lines.append("% ══════════════════════════════════════════")
    latex_lines.append("% SBES 2026 — Auto-generated LaTeX Tables")
    latex_lines.append("% Run: python 06_analyze_results.py")
    latex_lines.append("% ══════════════════════════════════════════")
    latex_lines.append("")

    code_m = mdf[mdf["task"] == "code"]
    models = [m for m in MODEL_ORDER if m in code_m["model"].unique()]
    lvls   = [l for l in LEVELS if l in code_m["level"].values]

    # ── Table 1: F1-Macro by model × level ──
    latex_lines.append("% Table 1: F1-Macro (primary metric)")
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{F1-Macro scores per model and evidence level (mean across runs). "
                       r"Bold = best per row.}")
    latex_lines.append(r"\label{tab:f1macro}")
    latex_lines.append(r"\begin{tabular}{l" + "c" * len(lvls) + "}")
    latex_lines.append(r"\toprule")
    latex_lines.append("Model & " + " & ".join(lvls) + r" \\")
    latex_lines.append(r"\midrule")

    pivot = (code_m.groupby(["model","level"])["f1_macro"]
             .mean().unstack("level")
             .reindex(index=models, columns=lvls).round(3))

    for model in models:
        label = MODEL_LABELS.get(model, model)
        if model not in pivot.index:
            continue
        row_vals = pivot.loc[model]
        best     = row_vals.max()
        cells    = []
        for l in lvls:
            v = row_vals.get(l, float("nan"))
            if pd.isna(v):
                cells.append("—")
            elif abs(v - best) < 0.0001:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        latex_lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    # ── Table 2: Cohen's Kappa model vs gold ──
    mvg = kdf[(kdf["kappa_type"] == "model_vs_gold") & (kdf["task"] == "code")]
    if not mvg.empty:
        latex_lines.append("% Table 2: Cohen's Kappa (model vs gold)")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Cohen's $\kappa$ (model vs. gold standard) per level. "
                           r"Interpretation: $<$0.40 Fair, 0.40--0.60 Moderate, $>$0.60 Substantial.}")
        latex_lines.append(r"\label{tab:kappa}")
        latex_lines.append(r"\begin{tabular}{l" + "c" * len(lvls) + "}")
        latex_lines.append(r"\toprule")
        latex_lines.append("Model & " + " & ".join(lvls) + r" \\")
        latex_lines.append(r"\midrule")

        kpivot = (mvg.groupby(["model_a","level"])["kappa"]
                  .mean().unstack("level")
                  .reindex(index=models, columns=lvls).round(3))

        for model in models:
            label = MODEL_LABELS.get(model, model)
            if model not in kpivot.index: continue
            row_vals = kpivot.loc[model]
            best     = row_vals.max()
            cells    = []
            for l in lvls:
                v = row_vals.get(l, float("nan"))
                if pd.isna(v): cells.append("—")
                elif abs(v - best) < 0.0001: cells.append(f"\\textbf{{{v:.3f}}}")
                else: cells.append(f"{v:.3f}")
            latex_lines.append(f"{label} & " + " & ".join(cells) + r" \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

    # ── Table 3: Context Gain ──
    sub = gain_df[gain_df["metric"] == "f1_macro"]
    if not sub.empty:
        latex_lines.append("% Table 3: Context Gain L1→L5")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Absolute and relative F1-Macro gain from L1 to L5.}")
        latex_lines.append(r"\label{tab:gain}")
        latex_lines.append(r"\begin{tabular}{lccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Model & L1 & L5 & $\Delta$F1 & $\Delta$\% & Best Level \\")
        latex_lines.append(r"\midrule")
        for model in models:
            row = sub[sub["model"] == model]
            if row.empty: continue
            r   = row.iloc[0]
            lbl = MODEL_LABELS.get(model, model)
            latex_lines.append(
                f"{lbl} & {r['L1']:.3f} & {r['L5']:.3f} & "
                f"{r['abs_gain']:+.3f} & "
                f"{r['rel_gain_pct']:+.1f}\\% & {r['best_level']} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

    # ── Table 4: Model Rank Summary ──
    if not rank_df.empty:
        latex_lines.append("% Table 4: Model Rank Summary")
        latex_lines.append(r"\begin{table}[t]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Model ranking summary across F1-Macro, stability, and cost. "
                           r"Rank 1 = best. Overall rank = mean of three individual ranks.}")
        latex_lines.append(r"\label{tab:ranks}")
        latex_lines.append(r"\begin{tabular}{lcccccc}")
        latex_lines.append(r"\toprule")
        latex_lines.append(r"Model & Mean F1 & Best Level & Stability SD & "
                           r"Total Cost & Cost/Correct & Overall Rank \\")
        latex_lines.append(r"\midrule")
        for _, r in rank_df.iterrows():
            cc = f"\\${r['cost_per_correct']:.5f}" if r['cost_per_correct'] else "—"
            latex_lines.append(
                f"{r['model']} & {r['mean_f1_macro']:.3f} & {r['best_level']} & "
                f"{r['stability_sd']:.4f} & \\${r['total_cost_usd']:.4f} & "
                f"{cc} & {r['rank_overall']} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

    out_path = out_dir / "latex_tables.tex"
    out_path.write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"  ✅ latex_tables.tex → {out_path}")


# ══════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════
def write_report(df, mdf, kdf, gain_df, sig_df, rank_df, path):
    lines = [
        "="*65,
        "SBES 2026 — Deductive Coding of CI Claims",
        "Summary Report  (n=113, L1–L5, 3 models)",
        "="*65,"",
        f"Records : {len(df)}  |  Claims: {df['paper_id'].nunique()}",
        f"Models  : {', '.join(sorted(df['model_short'].unique()))}",
        f"Levels  : {', '.join(sorted(df['level'].unique()))}",
        f"Runs    : {sorted(df['run_id'].unique())}",
        "",
        "Note: Multi-label claims (n=12) excluded. See paper limitations.",
        "Note: F1-Macro is the primary metric (31 classes, imbalanced).",
        "",
    ]

    # F1-Macro FIRST (primary)
    if not mdf.empty:
        cm = mdf[mdf["task"]=="code"]
        lines.append("── F1-Macro (PRIMARY) — mean across runs ──")
        if not cm.empty:
            pv = cm.groupby(["model","level"])["f1_macro"].mean().unstack("level").round(4)
            lines.append(pv.to_string())
        lines.append("")

        lines.append("── Accuracy — mean across runs ──")
        if not cm.empty:
            pv = cm.groupby(["model","level"])["accuracy"].mean().unstack("level").round(4)
            lines.append(pv.to_string())
        lines.append("")

    if not kdf.empty:
        lines.append("── Cohen's Kappa (Model vs Gold, code) ──")
        mvg = kdf[(kdf["kappa_type"]=="model_vs_gold")&(kdf["task"]=="code")]
        if not mvg.empty:
            pv = mvg.groupby(["model_a","level"])["kappa"].mean().unstack("level").round(4)
            lines.append(pv.to_string())
        lines.append("")

    if not gain_df.empty:
        lines.append("── Context Gain L1→L5 (F1-Macro) ──")
        sub = gain_df[gain_df["metric"]=="f1_macro"][
            ["model","L1","L5","abs_gain","rel_gain_pct","best_level"]]
        lines.append(sub.to_string(index=False))
        lines.append("")

    if not sig_df.empty:
        lines.append("── Statistical Significance ──")
        lines.append(sig_df.to_string(index=False))
        lines.append("")

    if not rank_df.empty:
        lines.append("── Model Rank Summary ──")
        cols = ["model","mean_f1_macro","best_level","stability_sd",
                "total_cost_usd","cost_per_correct","rank_overall"]
        cols = [c for c in cols if c in rank_df.columns]
        lines.append(rank_df[cols].to_string(index=False))
        lines.append("")

    lines += ["="*65,"End of report","="*65]
    path.write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--out_dir",     default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    fig_dir  = out_dir / "figures"
    comp_dir = out_dir / "comparisons"
    for d in [out_dir, fig_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n"+"="*65)
    print("SBES 2026 — 06_analyze_results.py  v5.0")
    print("n=113 | L1–L5 | 3 models | F1-Macro primary")
    print("="*65)

    print("\n[1/9] Loading results...")
    df = load_results(Path(args.results_dir))
    if df.empty: print("  ❌ No results found."); return

    print("\n[2/9] Core metrics (F1-Macro, Accuracy, Kappa)...")
    mdf = compute_metrics(df)
    mdf.to_csv(out_dir/"metrics_by_model_level.csv", index=False)
    print(f"  ✅ {len(mdf)} rows → metrics_by_model_level.csv")

    print("\n[3/9] Kappa analysis...")
    kdf = compute_kappa(df)
    kdf.to_csv(out_dir/"kappa_analysis.csv", index=False)
    print(f"  ✅ {len(kdf)} rows → kappa_analysis.csv")

    print("\n[4/9] Class distribution...")
    dist_df = compute_class_distribution(df)
    dist_df.to_csv(out_dir/"class_distribution.csv", index=False)
    print(f"  ✅ {len(dist_df)} classes → class_distribution.csv")

    print("\n[5/9] Context gain (L1→L5)...")
    gain_df = compute_context_gain(mdf)
    gain_df.to_csv(out_dir/"context_gain.csv", index=False)
    print(f"  ✅ {len(gain_df)} rows → context_gain.csv")

    print("\n[6/9] Significance tests...")
    sig_df = compute_significance_tests(mdf)
    sig_df.to_csv(out_dir/"significance_tests.csv", index=False)
    if sig_df.empty:
        print("  ⚠  Need ≥3 runs per level for significance tests")
    else:
        print(f"  ✅ {len(sig_df)} tests → significance_tests.csv")

    print("\n[7/9] Detailed analysis...")
    compute_theme_accuracy(df).to_csv(out_dir/"theme_accuracy.csv", index=False)
    compute_theme_confusions(df).to_csv(out_dir/"theme_confusions.csv", index=False)
    compute_confusion_pairs(df).to_csv(out_dir/"confusion_pairs.csv", index=False)
    compute_hard_claims(df).to_csv(out_dir/"hard_claims.csv", index=False)
    compute_confidence_calibration(df).to_csv(out_dir/"confidence_calibration.csv", index=False)
    cost_df = compute_cost(df)
    cost_df.to_csv(out_dir/"cost_summary.csv", index=False)
    rank_df = compute_model_rank_table(mdf, cost_df, kdf)
    rank_df.to_csv(out_dir/"model_rank_table.csv", index=False)
    (df.groupby(["model_short","level"])
     .agg(total=("parse_success","count"),
          ok=("parse_success","sum"),
          rate=("parse_success","mean"))
     .round(3).reset_index()
     .to_csv(out_dir/"parse_success.csv", index=False))

    # ── Deep analyses (linked to RQs, not new hypotheses) ──
    print("  Computing deep analyses (RQ1: stability, RQ3: difficulty + consensus)...")
    stability_df = compute_claim_stability(df)
    stability_df.to_csv(out_dir/"claim_stability.csv", index=False)

    difficulty_df = compute_taxonomy_difficulty(df)
    difficulty_df.to_csv(out_dir/"taxonomy_difficulty.csv", index=False)

    consensus_result = compute_consensus_map(df)
    if isinstance(consensus_result, tuple):
        consensus_df, consensus_summary = consensus_result
        consensus_df.to_csv(out_dir/"consensus_map.csv", index=False)
        consensus_summary.to_csv(out_dir/"consensus_summary.csv", index=False)
    else:
        consensus_df      = pd.DataFrame()
        consensus_summary = pd.DataFrame()

    print("  ✅ All detailed CSVs written")

    print("\n[8/9] Figures...")
    tdf = pd.read_csv(out_dir/"theme_accuracy.csv")
    plot_heatmaps(mdf, kdf, fig_dir)
    plot_line_by_level(mdf, fig_dir)
    plot_class_distribution(dist_df, fig_dir)
    plot_context_gain(gain_df, fig_dir)
    plot_theme_accuracy(tdf, fig_dir)
    # Confusion matrix → appendix
    appendix_dir = out_dir / "appendix"
    appendix_dir.mkdir(exist_ok=True)
    plot_confusion_matrix(df, appendix_dir)
    # Publication-ready main figure
    plot_publication_figure(mdf, fig_dir)
    # Deep analysis figures
    plot_claim_stability_heatmap(stability_df, fig_dir)
    plot_taxonomy_difficulty(difficulty_df, fig_dir)
    if not consensus_summary.empty:
        plot_consensus_distribution(consensus_summary, fig_dir)
    build_comparison_tables(df, comp_dir)
    print(f"  ✅ {len(list(fig_dir.glob('*.png')))} figures → figures/")
    print(f"  ✅ confusion matrices → appendix/")

    print("\n[9/9] Summary report + LaTeX tables...")
    write_report(df, mdf, kdf, gain_df, sig_df, rank_df, out_dir/"summary_report.txt")
    export_latex_tables(mdf, kdf, gain_df, rank_df, out_dir)
    print("  ✅ summary_report.txt + latex_tables.tex")

    print("\n"+"="*65)
    print(f"Analysis complete → {out_dir}")
    print("="*65+"\n")


if __name__ == "__main__":
    main()

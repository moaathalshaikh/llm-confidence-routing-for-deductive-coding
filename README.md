# Confidence-Aware HITL Deductive Coding for Software Engineering Evidence Claims

> Replication package submitted for double-anonymous peer review to a software engineering venue.  
> **Author information has been omitted for double-anonymous review.**  
> Full author information and affiliations will be disclosed upon acceptance.

---

# Overview

This repository contains the complete replication package for an empirical study investigating whether **self-reported confidence scores from Large Language Models (LLMs) can serve as reliable routing signals in Human-in-the-Loop (HITL) deductive coding workflows** for software engineering evidence claims.

The study evaluates confidence calibration, routing behavior, and human–AI collaboration strategies using three LLMs across five levels of context granularity, using 113 expert-annotated Continuous Integration (CI) evidence claims derived from a published systematic literature review.

Each model–context configuration was executed ten times, resulting in **16,950 individual classification decisions** used for calibration and routing analysis.

---

# Research Questions

### RQ1 — Confidence Calibration

How well calibrated are self-reported confidence scores from LLMs in deductive coding tasks?

### RQ2 — Context Granularity

How does context granularity affect confidence calibration quality across different models?

### RQ3 — Model Behavior

How do confidence calibration patterns differ across models and what implications do these differences have for HITL workflows?

### RQ4 — Routing Design

What confidence-aware routing strategy can support trustworthy Human-in-the-Loop deductive coding pipelines?

---

# Main Contributions

This study contributes:

- An empirical evaluation of confidence calibration for LLM-based deductive coding.
- A comparison of calibration behavior across three LLMs and five context levels.
- A human baseline for comparison against model agreement levels.
- A confidence-aware routing simulation exploring automation–accuracy trade-offs.
- A practical design artifact: the **Confidence-Action Taxonomy (CAT)** for HITL workflows.

---

# Key Findings

| Finding | Observation |
|----------|-------------|
| Best calibrated model | Claude Haiku 4.5 |
| Lowest observed calibration error | Claude Haiku 4.5 at L3 (ECE = 0.020) |
| Best predictive performance | Claude Haiku 4.5 at L1 |
| Confidence distribution issue | DeepSeek assigns high confidence to more than 91% of predictions |
| Context effect | Additional bibliographic context degrades calibration and predictive performance in this dataset |
| Human agreement | Human coders outperform all evaluated models |
| Best routing configuration | Claude L1 with θ = 0.80 |
| Automation rate | 56.8% of claims automatically accepted |
| Automated decision accuracy | 90.6% |
| Overall pipeline accuracy | 94.7% |

---

# Experimental Design

| Parameter | Value |
|-----------|-------|
| Dataset | 113 CI evidence claims |
| Deductive codes | 31 |
| Themes | 6 |
| Models | Claude Haiku 4.5 · DeepSeek Chat · GPT-4o Mini |
| Context levels | L1–L5 |
| Independent runs | 10 |
| Total evaluations | 150 |
| Total model decisions | 16,950 |
| Human annotators | 2 independent coders |
| Temperature | 0.0 |
| Output schema | Closed-set single-label classification |
| Confidence reporting | Self-reported confidence score per prediction |

---

# Context Granularity Levels

| Level | Name | Content |
|-------|------|---------|
| L1 | Claim Only | Claim text only |
| L2 | Source Context | Claim + study title |
| L3 | Methodological Context | Claim + title + research method |
| L4 | Empirical Context | Claim + title + measured variables |
| L5 | Full Context | Claim + title + method + variables |

---

# Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
│
├── SBQS2026_Confidence_Aware_HITL_Routing.ipynb
│
├── data/
│   ├── dataset_gold_v1_2_frozen.json
│   └── README_data.md
│
├── prompts/
│   ├── prompt_L1.txt
│   ├── prompt_L2.txt
│   ├── prompt_L3.txt
│   ├── prompt_L4.txt
│   └── prompt_L5.txt
│
├── scripts/
│   ├── 01_setup_prompts.py
│   ├── 03_run_claude.py
│   ├── 04_run_deepseek.py
│   ├── 05_run_openai.py
│   ├── 07_analyze_results.py
│   ├── 08_calibration_analysis.py
│   ├── 09_threshold_sweep.py
│   └── 10_human_baseline.py
│
├── analysis/
│   ├── figures/
│   ├── calibration/
│   ├── routing/
│   ├── human_baseline/
│   └── appendix/
│
└── results/
    ├── claude/
    ├── deepseek/
    └── openai/
```

---

# Gold Standard Dataset

The dataset contains **113 expert-annotated Continuous Integration evidence claims** derived from:

> Soares, E., Sizilio, G., Santos, J., da Costa, D. A., & Kulesza, U. (2022).  
> *The effects of continuous integration on software development: a systematic literature review.*  
> Empirical Software Engineering, 27(3), 78.

Original artefacts are archived at:

https://doi.org/10.5281/zenodo.4545623

The dataset contains:

- 113 single-label claims
- 31 deductive codes
- 6 themes

Twelve multi-label claims present in the original dataset were excluded to preserve a single-label setting.

---

# Human Baseline

To establish a human reference baseline, two independent graduate researchers in Software Engineering annotated all 113 claims using the original 31-code taxonomy.

The annotators:

- worked independently,
- had no access to LLM predictions,
- had no access to confidence scores,
- had no access to gold labels during annotation.

Inter-rater agreement was measured using Cohen's κ.

---

# Confidence Calibration

Calibration quality is evaluated using:

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability Diagrams
- Confidence Histograms

The study investigates whether:

> a confidence score of 0.80 corresponds to approximately 80% empirical correctness.

---

# Confidence-Aware Routing

The repository includes threshold sweep simulations for confidence-aware routing strategies.

Example routing policy:

| Confidence | Action |
|------------|--------|
| confidence ≥ θ | Automatic acceptance |
| confidence < θ | Human review |

The resulting automation–accuracy trade-offs are reported for multiple thresholds.

---

# Confidence-Action Taxonomy (CAT)

The proposed Confidence-Action Taxonomy maps confidence–accuracy patterns to routing decisions.

| Category | Recommended Action |
|----------|-------------------|
| Confident-Correct | Automatic acceptance |
| Confident-Wrong | Mandatory review |
| Uncertain-Correct | Prompt refinement |
| Uncertain-Wrong | Human review |
| Systematic Failure | Taxonomy inspection |

---

# Reproducing the Results

## Google Colab

Open:

```text
SBQS2026_Confidence_Aware_HITL_Routing.ipynb
```

The notebook reproduces:

- model execution,
- calibration analysis,
- routing simulation,
- human baseline analysis,
- publication figures.

---

## Local Execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate prompts:

```bash
python scripts/01_setup_prompts.py
```

Run all experiments:

```bash
for i in $(seq 1 10); do
    python scripts/03_run_claude.py --run_id $i
    python scripts/04_run_deepseek.py --run_id $i
    python scripts/05_run_openai.py --run_id $i
done
```

Generate analyses:

```bash
python scripts/07_analyze_results.py
python scripts/08_calibration_analysis.py
python scripts/09_threshold_sweep.py
python scripts/10_human_baseline.py
```

---

# Dependencies

```text
anthropic
openai
requests
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
python-dotenv
```

---

# Double-Anonymous Review Notice

This repository has been prepared for double-anonymous peer review.

Author names, affiliations, commit history, and identifying information have been removed from this version.

A permanent public repository will be released upon publication.

---

# License

This repository is distributed under the MIT License.

The gold-standard dataset remains subject to the licensing terms of the original Zenodo artefact released by Soares et al.

---

# Citation

If this work is accepted, citation information will be added here.

Please cite the original CI evidence dataset as:

```bibtex
@misc{soares2021zenodo,
  author    = {Soares, Eliezio and Sizilio, Gustavo and Santos, Jadson and da Costa, Daniel Alencar and Kulesza, Uirá},
  title     = {SLR Artifacts -- Continuous Integration Quality Impacts},
  year      = {2021},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4545623},
  url       = {https://doi.org/10.5281/zenodo.4545623}
}
```
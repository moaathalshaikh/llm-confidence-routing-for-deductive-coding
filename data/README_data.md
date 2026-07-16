# Data Provenance

## Gold-Standard Dataset

The file `dataset_gold_v1_2_frozen.json` contains 113 expert-annotated
Continuous Integration (CI) evidence claims used as the reference dataset
for evaluating LLM confidence calibration, confidence-aware routing, and
human–AI collaboration in deductive coding workflows.

## Source

The evidence claims originate from the replication package of:

> Soares, E., Sizilio, G., Santos, J., da Costa, D. A., & Kulesza, U. (2022).
> *The effects of continuous integration on software development: a systematic
> literature review.* Empirical Software Engineering, 27(3), 78.
> https://doi.org/10.1007/s10664-021-10114-1

The original artefacts are archived at Zenodo (v1.0.2):

https://doi.org/10.5281/zenodo.4545623

## Role of the Dataset in This Study

The dataset serves as the common reference benchmark for four complementary
analyses:

1. **Deductive coding performance evaluation** across multiple LLMs and
   context granularity levels.

2. **Confidence calibration analysis**, including Expected Calibration Error
   (ECE) estimation and reliability diagrams.

3. **Confidence-aware Human-in-the-Loop routing simulation**, evaluating
   automation–accuracy trade-offs under different confidence thresholds.

4. **Human baseline estimation**, using independent human coders to establish
   reference agreement levels for the deductive coding task.

## What We Used

- **113 single-label evidence claims**, each manually annotated by expert
  reviewers in the original SLR.
- Each claim is assigned to:
  - one of **31 deductive codes**, and
  - one of **6 thematic categories**.
- **12 multi-label claims** present in the original dataset were excluded to
  preserve a single-label classification setting and ensure comparability
  between human and LLM annotators.

## Fields in `dataset_gold_v1_2_frozen.json`

```json
{
  "paper_id": "unique identifier for source primary study",
  "claim": "evidence claim text extracted from the original SLR",
  "code": "gold-standard deductive code (one of 31 codes)",
  "theme": "gold-standard theme (one of 6 themes)",
  "title": "title of the originating primary study",
  "kind": "research method (case study, MSR, survey, experiment, ...)",
  "variables": "measured constructs / dependent variables (or 'Not reported')"
}
```

## Taxonomy Structure

The deductive coding taxonomy contains **31 codes** organized into **6 themes**.

| Theme | Number of Codes |
|-------|-----------------|
| Build Patterns | 2 |
| Quality Assurance | 4 |
| Integration Patterns | 4 |
| Issues and Defects | 3 |
| Development Activities | 9 |
| Software Process | 9 |

The complete taxonomy definitions are available in:

- `codebook_31_codes.csv`
- `taxonomy_definitions.md`

## Human Baseline Protocol

To establish a human reference baseline for the task, two independent
graduate researchers in Software Engineering annotated the 113 claims using
the original 31-code taxonomy.

The annotators:

- worked independently,
- had no access to LLM predictions,
- had no access to confidence scores,
- had no access to the gold-standard labels during annotation.

Inter-rater agreement was subsequently quantified using Cohen's κ.

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Evidence claims | 113 |
| Deductive codes | 31 |
| Themes | 6 |
| Primary study domain | Continuous Integration |
| Classification setting | Single-label |
| Human coders | 2 |
| LLM models evaluated | 3 |
| Context levels | 5 |
| Total model decisions | 16,950 |

## License and Attribution

This dataset is derived from research artefacts released by Soares et al.
under an open research artefact license.

Users of this repository should cite both:

1. The original EMSE paper:
   Soares et al. (2022), *The effects of continuous integration on software
   development: a systematic literature review*.

2. The original Zenodo artefact:
   https://doi.org/10.5281/zenodo.4545623

Please also cite this repository when reusing the calibration analyses,
human baseline annotations, or confidence-aware routing artefacts.
"""
╔══════════════════════════════════════════════════════════════╗
║  SBES 2026 — Deductive Coding of CI Claims                   ║
║  FILE 1 of 7: setup_prompts.py                               ║
║  PURPOSE: Create the 5 prompt files on Google Drive          ║
║  RUN ONCE before any experiments                             ║
╚══════════════════════════════════════════════════════════════╝

Evidence Levels (the only variable in the experiment):
  L1 — Claim Only            : claim text only
  L2 — Source Context        : claim + study title
  L3 — Methodological Context: claim + title + research method
  L4 — Empirical Context     : claim + title + variables
  L5 — Full Context          : claim + title + method + variables

Prompt strategy: FIXED — Structured Closed-set Deductive Coding
  (same role, rules, codebook, disambiguation, output format)

Usage in Colab:
    !python setup_prompts.py
"""

from pathlib import Path

BASE_DIR = Path(os.getenv("SBES_BASE_DIR", Path(__file__).parent.parent))
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# FIXED COMPONENTS — identical in all 5 prompts
# ══════════════════════════════════════════════════════════════
ROLE = (
    "You are a systematic literature review researcher performing "
    "controlled deductive coding of empirical claims about Continuous "
    "Integration (CI) in software engineering."
)

DECISION_RULES = """\
=== DECISION RULES ===
- Assign EXACTLY ONE code from the codebook below.
- Choose the code whose definition BEST matches the core claim.
- Base your decision ONLY on explicit content in the provided text.
  Do not infer intent, causality, or context not stated.
- Return the code label EXACTLY as written in the codebook,
  character by character, uppercase.
- If two codes seem equally applicable, apply the disambiguation rules.
- If no code fits well, choose the closest and set confidence below 0.50.\
"""

CODEBOOK = """\
=== CODEBOOK ===

-- THEME: BUILD PATTERNS --
- CI IS ASSOCIATED WITH BUILD HEALTH
- CI IS ASSOCIATED WITH A DECREASING IN BUILD TIME

-- THEME: QUALITY ASSURANCE --
- CI IS RELATED TO POSITIVE IMPACTS ON TEST PRACTICE
- CI IS RELATED TO AN INCREASE QUALITY ASSESSMENT
- CI IS ASSOCIATED WITH TO FAVOR CONTINUOUS REFACTORING
- CI IS ASSOCIATED TO MULTI-ENVIRONMENT TESTS

-- THEME: INTEGRATION PATTERNS --
- CI IS RELATED TO POSITIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE
- CI IS RELATED TO NEGATIVE IMPACTS ON PULL REQUESTS LIFE-CYCLE
- CI IS ASSOCIATED WITH A POSITIVE IMPACT ON INTEGRATION PRACTICE
- CI IS ASSOCIATED WITH A COMMIT PATTERN CHANGE

-- THEME: ISSUES AND DEFECTS --
- CI IS ASSOCIATED WITH DEFECTS REDUCTION
- CI IS ASSOCIATED WITH ISSUES REDUCTION
- CI IS ASSOCIATED WITH A DECREASE IN TIME TO LEAD DEFECTS

-- THEME: DEVELOPMENT ACTIVITIES --
- CI MAY GENERATE A FALSE SENSE OF CONFIDENCE
- CI IS ASSOCIATED WITH CONFIDENCE IMPROVEMENT
- CI IS RELATED TO PRODUCTIVITY AND EFFICIENCY INCREASING
- CI IS ASSOCIATED WITH ADDING EXTRA COMPLEXITY
- CI IS ASSOCIATED WITH A WORKLOAD REDUCTION
- CI IS ASSOCIATED WITH A DECREASED PERCEPTION OF PRODUCTIVITY
- CI IS ASSOCIATED WITH AN IMPROVEMENT IN SATISFACTION
- CI IS ASSOCIATED WITH HUMAN CHALLENGES
- CI IS ASSOCIATED WITH A DECREASE IN MAGNETISM AND RETENTION

-- THEME: SOFTWARE PROCESS --
- CI IS RELATED TO POSITIVE IMPACTS ON RELEASE CYCLE
- CI IS ASSOCIATED WITH AN INCREASE IN COOPERATION
- CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS RELIABILITY
- CI IS ASSOCIATED WITH AN IMPROVEMENT IN PROCESS AUTOMATION
- CI IS ASSOCIATED WITH TECHNICAL CHALLENGES
- CI IS ASSOCIATED WITH SOFTWARE DEVELOPMENT BENEFITS
- CI IS ASSOCIATED WITH A FEED BACK FREQUENCY INCREASE
- CI IS ASSOCIATED WITH ORGANIZATIONAL CHALLENGES
- CI FACILITATES THE TRANSITION TO AGILE\
"""

DISAMBIGUATION = """\
=== DISAMBIGUATION RULES ===
- TECHNICAL vs HUMAN vs ORGANIZATIONAL CHALLENGES:
    TECHNICAL     : tools, infrastructure, or configuration.
    HUMAN         : mindset, skills, or individual resistance.
    ORGANIZATIONAL: processes, culture, or management.
- DEFECTS REDUCTION vs ISSUES REDUCTION:
    DEFECTS: bugs or errors found in the code.
    ISSUES : reported problems in issue trackers or tickets.
- PRODUCTIVITY INCREASING vs DECREASED PERCEPTION OF PRODUCTIVITY:
    INCREASING         : objective output or throughput improvements.
    DECREASED PERCEPTION: subjective feelings of being less productive.
- FALSE SENSE OF CONFIDENCE vs CONFIDENCE IMPROVEMENT:
    FALSE SENSE : CI misleads (green build does not mean correct code).
    IMPROVEMENT : CI genuinely and correctly increases trust.
- PROCESS RELIABILITY vs PROCESS AUTOMATION:
    RELIABILITY: consistency, predictability, or stability.
    AUTOMATION : automation of steps, pipelines, or deployments.
- BUILD HEALTH vs DECREASING IN BUILD TIME:
    BUILD HEALTH: pass/fail status, stability, or flakiness.
    DECREASING  : speed or duration of builds.
- SOFTWARE DEVELOPMENT BENEFITS: use ONLY for broad general claims
    that do not fit any more specific code above.\
"""

OUTPUT_FORMAT = """\
=== OUTPUT FORMAT ===
Return ONLY the following JSON object. No preamble. No explanation.

{
  "code": "EXACT CODE LABEL FROM CODEBOOK",
  "theme": "EXACT THEME NAME FROM CODEBOOK",
  "confidence": 0.00
}

- code      : must match a codebook entry exactly, character by character.
- theme     : must match a theme name exactly (e.g., "SOFTWARE PROCESS").
- confidence: float 0.00–1.00.
              Use > 0.80 only when the match is unambiguous.
              Use < 0.50 when the best code is uncertain.\
"""

# ══════════════════════════════════════════════════════════════
# VARIABLE COMPONENT — context block per level
# ══════════════════════════════════════════════════════════════
CONTEXT_BLOCKS = {
    "L1": """\
=== CLAIM ===
{claim}\
""",
    "L2": """\
=== STUDY INFORMATION ===
Title: {study_title}

=== CLAIM ===
{claim}\
""",
    "L3": """\
=== STUDY INFORMATION ===
Title: {study_title}
Research Method: {kind}

=== CLAIM ===
{claim}\
""",
    "L4": """\
=== STUDY INFORMATION ===
Title: {study_title}
Variables / Measured Constructs: {variables}

=== CLAIM ===
{claim}\
""",
    "L5": """\
=== STUDY INFORMATION ===
Title: {study_title}
Research Method: {kind}
Variables / Measured Constructs: {variables}

=== CLAIM ===
{claim}\
""",
}

# ══════════════════════════════════════════════════════════════
# BUILD AND WRITE
# ══════════════════════════════════════════════════════════════
for level, ctx in CONTEXT_BLOCKS.items():
    text = "\n\n".join([ROLE, DECISION_RULES, CODEBOOK,
                        DISAMBIGUATION, ctx, OUTPUT_FORMAT]) + "\n"
    path = PROMPTS_DIR / f"prompt_{level}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  ✅ prompt_{level}.txt  ({len(text)} chars)")

print(f"\n✓ 5 prompt files → {PROMPTS_DIR}")
print("✓ Next step: run the experiment scripts")

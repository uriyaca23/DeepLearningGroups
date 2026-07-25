# HW4 project state

Last reconciled: 2026-07-25

## Objective

Complete HW4 collaboratively as a learning exercise. The student's reasoning
must lead each answer; Codex challenges, hints, formalizes, implements only
approved designs, and writes approved Hebrew answers.

## Governing files

- `HOMEWORK_STYLE_GUIDE.md`: authoritative constant collaboration, writing,
  Word, RTL, and verification rules.
- `style/homework_style_contract.json`: machine-checkable Word/OOXML values.
- `AGENTS.md`: mandatory entry point for future sessions.
- `HW4.pdf`: authoritative assignment wording.
- `q3_set_networks.py`: current approved Question 3(a) implementation.

Do not duplicate stable style rules here. Do not use or modify
LocationPipeline.

## Preserved Q1-Q2 visual baseline

- File: `old/HW4-Solution.pre-q3a-20260724-230819.docx`
- SHA-256:
  `6197D53B5522953BDFEF486372541E9AB082FBF259471BED34AE11891EEE4AE2`
- Accepted scope: Question 1 through Question 2(e), including the corrected
  Q2(c) visual and the approved Q2(e) conclusion: the linear layer has eight
  free parameters; if "layer" means an affine layer, the invariant bias adds
  one, for nine parameters in total.

This remains the immutable recovery baseline through Question 2(e). It must
remain unchanged.

## Current accepted checkpoint

- File: `HW4-Solution.docx`
- SHA-256:
  `0D2F314D00BE2668A5EE1C13867D2F026B26E290EA91535DBAA01717EA853916`
- Accepted scope: Question 1 through Question 3(b).
- User visual approval: granted on 2026-07-25 after the full-page Microsoft
  Word renders of pages 11 and 12 were shown.
- Git checkpoint: `842c46e` on `main`.
- Publication status: published to `origin/main`.
- Accepted Q3(a) predecessor archived as
  `old/HW4-Solution.pre-q3b-20260725-150335.docx`, SHA-256
  `29691CF9182C9FE438FDDC78EDC73217E8E95A0B1B80E78BEAB2115C952D85A5`.
- The earlier rejected Q3(a) checkpoint remains archived as
  `old/HW4-Solution.rejected-q3a-20260725-125359.docx`, SHA-256
  `5BB090D5A6504D45716B5E764F75725B32750515E5760AFA9C192461826EA42C`.

## Gate record

| Part | Reasoning | Formal text | Code/tests | DOCX insertion | Mechanical audit | User visual approval | Publication |
|---|---|---|---|---|---|---|---|
| Q1 | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(a) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(b) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(c) | approved | approved | n/a | accepted baseline; only required block-type visual retained | baseline passes | approved | published |
| Q2(d) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(e) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q3(a) | approved | approved | passed with revised `n=5` configuration | approved, revised, and promoted | passes | approved | published |
| Q3(b) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(c-e) | not started | not started | not started | not started | not started | not started | not started |
| Q4 | not started | not started | not started | not started | not started | not started | not started |

Semantic approval never implies DOCX-format approval.

## Q3(a-b) approved technical checkpoint

- Shared experimental configuration: seed 2319, `n=5`, `d=3`, `p=4`, hidden
  width 32, `atol=1e-5`, and `rtol=0`.
- Q3(a): canonize rows lexicographically, flatten the canonical matrix, and
  apply a two-layer MLP with one ReLU.
- Q3(b): apply the same ordinary order-sensitive two-layer MLP to every one of
  the `5! = 120` row permutations and average the resulting output vectors.
- Implementation: `q3_set_networks.py`.
- Recorded Q3(a) results: required invariance test passes; the
  partial-tie/duplicate-row test also passes.
- Recorded Q3(b) results: required invariance test passes; the ordinary base
  MLP is non-invariant on the test input, while the averaged model has maximum
  absolute permutation error `2.980232239e-08`. An exhaustive check of all 120
  input permutations and a finite-gradient check also pass.

These technical results may be reused in later comparisons; they do not need
to be re-derived unless the design changes.

## Resolved Q3(a) format repair

- The complete original English prompt is now preserved as two readable,
  high-resolution excerpts from the two source pages of `HW4.pdf`.
- The rejected `Original Question` style is absent.
- Question and subsection titles use the accepted Heading 1 and Heading 2
  structures.
- Hebrew prose inherits the accepted Word-native RTL paragraph behavior. The
  seven direct `right` alignment overrides that Word rendered on the physical
  left were removed; short lines now terminate at the physical right edge.
- Display equations remain centered LTR Word equations.
- The contract audit passes, and two independent Microsoft Word exports were
  pixel-identical on all eleven pages.
- The promoted main file was rendered again after promotion and matched the
  visually approved candidate on all eleven pages.

## Approved Q3(b) document addition

- Q3(a)'s experiment was revised from `n=20` to the user-approved `n=5` so
  parts (a) and (b) use the same setting. Its displayed MLP architecture now
  records the explicitly approved hidden width 32.
- Q3(b) uses an accepted Heading 2 clone, concise Hebrew prose in the
  HW1-HW3 style, and three centered Word-native equations. No answer figure
  was added.
- All new Hebrew paragraphs inherit the accepted native RTL behavior and
  DejaVu Sans body style; no direct `right` alignment override was introduced.
- The mechanical contract passes. Microsoft Word updated the TOC, saved,
  closed, and reopened the candidate successfully.
- Live Word inspection confirmed the accepted RTL reading order and alignment.
  Two independent Word exports were pixel-identical on all twelve pages.
- Pages 1-10 remained pixel-identical to the prior accepted checkpoint. The
  user visually approved the changed full-page renders of pages 11 and 12.
- After promotion, a fresh read-only Word export of `HW4-Solution.docx`
  matched the approved candidate pixel-for-pixel on all twelve pages.

## Infrastructure

- Repository: `https://github.com/uriyaca23/DeepLearningGroups.git`
- Branch: `main`
- Project virtual environment: `.venv`, Python 3.12.13.
- Project dependencies recorded in `requirements.txt`.
- Superseded DOCX checkpoints belong only under `old/`.
- Candidates, renders, and generated reports belong under ignored `_qa/`.

## Precise next action

Continue collaboratively with Question 3(c): present its original English
wording from the approved prompt image, give only brief course-oriented
context, and ask one focused question about sampled symmetrization and why it
is generally only approximately invariant. Do not write a Q3(c) answer into
the DOCX until its formal Hebrew answer is verified and explicitly approved
under the normal content and visual gates.

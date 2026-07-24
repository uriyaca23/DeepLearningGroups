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

## Accepted visual baseline

- File: `old/HW4-Solution.pre-q3a-20260724-230819.docx`
- SHA-256:
  `6197D53B5522953BDFEF486372541E9AB082FBF259471BED34AE11891EEE4AE2`
- Accepted scope: Question 1 through Question 2(e), including the corrected
  Q2(c) visual and the approved Q2(e) conclusion: the linear layer has eight
  free parameters; if "layer" means an affine layer, the invariant bias adds
  one, for nine parameters in total.

This is the last accepted style authority. It must remain unchanged.

## Current working checkpoint

- File: `HW4-Solution.docx`
- SHA-256:
  `5BB090D5A6504D45716B5E764F75725B32750515E5760AFA9C192461826EA42C`
- Git checkpoint: `5e52ec5` on `main`
- Publication status: pushed, but explicitly rejected as a DOCX-format
  checkpoint.

Question 3(a)'s reasoning, formal Hebrew content, approved design, code, and
tests remain valid. Its document structure and visual formatting are not
approved. The current file is therefore a repair source, not a style
authority and not a finished submission checkpoint.

## Gate record

| Part | Reasoning | Formal text | Code/tests | DOCX insertion | Mechanical audit | User visual approval | Publication |
|---|---|---|---|---|---|---|---|
| Q1 | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(a) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(b) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(c) | approved | approved | n/a | accepted baseline; only required block-type visual retained | baseline passes | approved | published |
| Q2(d) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q2(e) | approved | approved | n/a | accepted baseline | baseline passes | approved | published |
| Q3(a) | approved | approved | passed | present, but rejected formatting | fails prompt/forbidden-style gates | rejected | pushed only as rejected checkpoint |
| Q3(b-e) | not started | not started | not started | not started | not started | not started | not started |
| Q4 | not started | not started | not started | not started | not started | not started | not started |

Semantic approval never implies DOCX-format approval.

## Q3(a) approved technical checkpoint

- Canonize rows lexicographically, flatten the canonical matrix, and apply a
  two-layer MLP with one ReLU.
- Implementation: `q3_set_networks.py`.
- Approved experiment values: seed 2319, `n=20`, `d=3`, `p=4`, hidden width
  32, `atol=1e-5`, `rtol=0`.
- Recorded results: required invariance test passes with maximum absolute
  error 0; partial-tie/duplicate-row test, 125 additional permutation checks,
  gradient check, and CUDA smoke test also passed.

These technical results may be reused when repairing the document; they do not
need to be re-derived unless the design changes.

## Identified Q3(a) format failure

The Q3 prompt was retyped into eleven paragraphs using a newly invented
`Original Question` style (Aptos 9 pt, gray). HW1-HW3 and accepted HW4 Q1-Q2
instead place a centered, readable source excerpt immediately after the
Question Heading 1. This structural mismatch is the primary rejected-format
defect. Any additional RTL, font, heading, spacing, TOC, or pagination defects
must be resolved under the full guide and verified in Word.

## Infrastructure

- Repository: `https://github.com/uriyaca23/DeepLearningGroups.git`
- Branch: `main`
- Project virtual environment: `.venv`, Python 3.12.13.
- Project dependencies recorded in `requirements.txt`.
- Superseded DOCX checkpoints belong only under `old/`.
- Candidates, renders, and generated reports belong under ignored `_qa/`.

## Precise next action

Do not proceed to Q3(b) yet.

First, create a Q3(a) repair candidate from the accepted pre-Q3(a) baseline,
reuse the already approved Hebrew content and code results, insert the complete
Q3 prompt as a high-resolution excerpt from `HW4.pdf`, and reproduce the
accepted Heading 1, Heading 2, body, equation, header, TOC, font, and RTL
patterns. Then:

1. run `scripts/audit_homework_docx.py`;
2. save, close, reopen, and export the exact candidate with Microsoft Word;
3. inspect all changed/reflowed full pages and their neighbors;
4. show those full-page renders to the user;
5. wait for explicit visual approval before promotion, commit, or push.

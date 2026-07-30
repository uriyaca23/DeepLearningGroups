# HW4 project state

Last reconciled: 2026-07-30

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
- `q3_set_networks.py`: current approved Question 3(a-d) implementation.

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
  `EED29FB0BB6BFECBD1D2656D8CF10A4A38027419B958554262E93F555577A326`
- Accepted scope: Question 1 through Question 3(d).
- User visual approval: granted on 2026-07-30 for the unified
  input/output-feature revision through Q3(d), including the updated
  calculations and Word rendering.
- Git checkpoint: the accepted revision is the repository `HEAD` on `main`.
- Publication status: published to `origin/main` on 2026-07-30.
- The previous accepted Q3(a-d) checkpoint is archived as
  `old/HW4-Solution.pre-output3-consistency-20260730-160856.docx`, SHA-256
  `7B8C1B62C049CB55DB4D146DB1796E1FAFAB954064430E4E509D948ACD6EF2A1`.
- The accepted Q3(a-c) predecessor is archived as
  `old/HW4-Solution.pre-q3d-review-20260727-184452.docx`, SHA-256
  `61A32D4B2B95821F273C5677669F76D80838B1D14D8B302238141589CCE54856`.
- The accepted mixed-`n` Q3(c) predecessor is archived as
  `old/HW4-Solution.pre-q3-n7-revision-20260727-182303.docx`, SHA-256
  `E9C60FDFDD073A9414850646BFBE33F87C6CFA2252BAB2BE71F8B712D7711CE1`.
- The accepted Q3(b) predecessor is archived as
  `old/HW4-Solution.pre-q3c-review-20260727-173243.docx`, SHA-256
  `0D2F314D00BE2668A5EE1C13867D2F026B26E290EA91535DBAA01717EA853916`.
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
| Q3(a) | approved | approved | passed with unified `n=7` configuration | approved, revised, and promoted | passes | approved | published |
| Q3(b) | approved | approved | passed with exact `7!` symmetrization | approved, revised, and promoted | passes | approved | published |
| Q3(c) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(d) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(e) | not started | not started | not started | not started | not started | not started | not started |
| Q4 | not started | not started | not started | not started | not started | not started | not started |

Semantic approval never implies DOCX-format approval.

## Q3(a-c) approved technical checkpoint

- Shared experimental configuration: seed 2319, `n=7`, `d=3`, `p=3`, and
  hidden width 32.
- Q3(a-b) use `atol=1e-5` and `rtol=0`; Q3(c) uses the separately approved
  pure maximum-coordinate absolute tolerance `1e-2`.
- Q3(a): canonize rows lexicographically, flatten the canonical matrix, and
  apply a two-layer MLP with one ReLU.
- Q3(b): apply the same ordinary order-sensitive two-layer MLP to every one of
  the `7! = 5040` row permutations and average the resulting output vectors.
- Implementation: `q3_set_networks.py`.
- Recorded Q3(a) results: required invariance test passes; the
  partial-tie/duplicate-row test and an exhaustive check over all 5040 row
  permutations also pass. The maximum absolute error is zero.
- Recorded Q3(b) results: required invariance test passes; the ordinary base
  MLP is non-invariant on the test input, while the averaged model has maximum
  absolute permutation error `2.98023223877e-08`. The structural check confirms
  all 5040 elements of `S_7` occur exactly once, and the finite-gradient check
  also passes.
- Q3(c): sample one fixed subset of `B=2700` distinct permutations uniformly
  without replacement from `S_7`, and reuse that same subset for both compared
  inputs.
- The finite-population Monte Carlo calculation uses `N=7!=5040` and the
  calibrated 95th-percentile coefficient `0.7618`. Solving the resulting
  absolute-error estimate for tolerance `10^-2` gives `B >= 2697.7`; the
  approved rounded choice is `B=2700`, or about 53.6% of the full group and a
  1.87-fold computation reduction.
- Recorded Q3(c) results: the pure maximum-coordinate absolute test passes
  with error `0.00395260751247 < 10^-2`. Structural tests also confirm that
  the subset contains 2700 unique permutations, remains fixed between forward
  passes, and is reproduced by seed 2319.

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

- Q3(a-b) now use the user-approved common value `n=7`. The flattened MLP
  input dimension is therefore 21, while the explicitly approved hidden width
  remains 32 and the output dimension is 3.
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

## Approved Q3(c) document addition

- Q3(c) uses an accepted Heading 2 clone, concise Hebrew prose in the
  HW1-HW3 style, and Word-native displayed equations. No answer figure was
  added.
- The answer explains why a fixed proper subset is generally only
  approximately invariant, derives the finite-population Monte Carlo estimate,
  records how it led to `B=2700`, and reports the approved implementation and
  absolute-error test.
- All new Hebrew prose and the heading use the accepted native RTL behavior
  and DejaVu Sans styles; displayed equations remain centered LTR Word
  equations.
- The mechanical contract passes. Microsoft Word saved, closed, reopened, and
  exported the 13-page candidate. The changed pages were inspected at full
  page, including a high-resolution check of the page-13 header.
- The user explicitly approved the exact Word review candidate on 2026-07-27.
  After promotion, a fresh read-only Word export of `HW4-Solution.docx`
  matched the approved candidate pixel-for-pixel on all thirteen pages.

## Approved Q3(a-c) `n=7` unification revision

- At the user's request, Q3(a) and Q3(b) were revised from `n=5` to `n=7`;
  Q3(c) already used `n=7`.
- Q3(b) now records the domain `R^(7x3)`, flattened dimension 21, exact
  averaging over `7! = 5040` permutations, the proof over `S_7`, and the new
  measured maximum error `3.73 * 10^-9`.
- The code batches the 5040 evaluations for efficiency without changing the
  exact full-group average.
- The mechanical contract passes. Microsoft Word saved, reopened, and exported
  the 13-page candidate; pages 1-10 and 13 remained pixel-identical to the
  prior checkpoint, while revised pages 11-12 passed full-page RTL, header,
  font, heading, and equation inspection.
- The user explicitly approved the exact Word candidate on 2026-07-27. After
  promotion, a fresh read-only Word export matched the approved candidate
  pixel-for-pixel on all thirteen pages.

## Q3(d) approved technical checkpoint

- The equivariant linear layer uses two independent feature maps and one
  shared bias:
  `L(X)_i = X_i A + sum_{j != i} X_j B + b`. Equivalently, with the row
  mean, it is implemented using the two matrices
  `W_1 = A - B` and `W_2 = nB`.
- The approved equivariant network has widths `3 -> 62 -> 3`, with a
  pointwise ReLU between its two equivariant linear layers. Mean pooling over
  the seven rows produces the final invariant three-dimensional output.
- The width 62 gives 809 trainable DeepSets parameters, close to the 803
  parameters of the approved ordinary `n=7` MLP.
- Shared configuration: seed 2319, `n=7`, `d=3`, `p=3`, `atol=1e-5`, and
  `rtol=0`.
- Recorded maximum absolute errors are `1.19209289551e-7` for one equivariant
  layer, `1.19209289551e-7` for the full equivariant stack, and
  `0` after invariant mean pooling. All are below the approved
  tolerance, and the finite-gradient test passes.
- The full Q3(a-d) script and all Q3(a-c) regression checks pass.

## Approved Q3(d) document addition

- Q3(d) uses an accepted Heading 2 clone, concise Hebrew prose in the
  HW1-HW3 style, and five centered Word-native display equations. No answer
  figure was added.
- The Q3(c) title was corrected to exactly
  `סעיף ג. סימטריה דגומה.`; the rejected wording is absent.
- The new Hebrew prose and headings inherit the accepted native RTL behavior
  and DejaVu Sans styles. No forbidden direct right-alignment override was
  introduced.
- The mechanical contract passes. Microsoft Word saved, closed, reopened,
  and exported the 13-page candidate successfully.
- Pages 1-11 remained pixel-identical to the previous checkpoint. The
  corrected Q3(c) heading on page 12 and the complete Q3(d) answer on page 13
  passed full-page inspection; page 13 remained pixel-identical after the
  heading-only correction.
- The user explicitly approved the corrected Word candidate on 2026-07-27.
  After promotion, a fresh read-only Word export of `HW4-Solution.docx`
  matched the approved candidate pixel-for-pixel on all thirteen pages.

## Approved Q3 output-dimension consistency revision

- At the user's request, the Q3(a-c) ordinary MLP and the Q3(d) DeepSets
  network now share the same input- and output-feature requirement:
  `n=7`, `d=3`, and `p=d=3`.
- The ordinary flattened MLP is `21 -> 32 -> 3` with 803 trainable
  parameters. The comparison DeepSets network is `3 -> 62 -> 3` with 809
  trainable parameters, a difference of six parameters.
- All Q3(a-d) calculations and tests were rerun. The revised Q3(c)
  finite-population estimate keeps `B=2700`, now using coefficient `0.7618`
  and the bound `B >= 2697.7`.
- The user explicitly approved the exact revised Word review copy on
  2026-07-30. Its mechanical audit passed; Microsoft Word saved, reopened,
  and exported it successfully; and all 13 pages passed visual inspection.
- After promotion, a fresh read-only Word export of `HW4-Solution.docx`
  matched the approved candidate pixel-for-pixel on all 13 pages.

## Infrastructure

- Repository: `https://github.com/uriyaca23/DeepLearningGroups.git`
- Branch: `main`
- Project virtual environment: `.venv`, Python 3.12.13.
- Project dependencies recorded in `requirements.txt`.
- Superseded DOCX checkpoints belong only under `old/`.
- Candidates, renders, and generated reports belong under ignored `_qa/`.
- Give the user the review-file path rather than opening Word on the screen;
  open Word visibly only when the user explicitly requests it.

## Precise next action

Continue collaboratively with Question 3(e): present its original English
wording from `HW4.pdf`, give only brief orienting context, and ask for the
student's reasoning. Challenge mistakes with progressively stronger hints and
formalize only the answer the student understands and approves. Do not
implement or write the Q3(e) answer into the DOCX before its design and formal
Hebrew answer pass the normal explicit approval gates.

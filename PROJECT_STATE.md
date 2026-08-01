# HW4 project state

Last reconciled: 2026-08-01

## Objective

Complete HW4 collaboratively as a learning exercise. The student's reasoning
must lead each answer; Codex challenges, gives progressively stronger hints,
formalizes approved reasoning, implements only approved designs, and writes
approved final answers in polished Hebrew.

## Governing files

- `HOMEWORK_STYLE_GUIDE.md`: authoritative stable collaboration, writing,
  Word, RTL, and verification rules.
- `style/homework_style_contract.json`: machine-checkable Word/OOXML values
  and the exact approved media exceptions.
- `AGENTS.md`: mandatory entry point for future sessions.
- `HW4.pdf`: authoritative assignment wording.
- `q3_set_networks.py`: approved Question 3(a-e) implementation and recorded
  experiments.
- `q4_modelnet40.py`: approved Question 4 mandatory benchmark and
  surface-normal implementation, plus the rotation and Transformer Bonus
  implementations now awaiting final document approval.

Do not duplicate stable style rules here. Never use, inspect, or modify
LocationPipeline for this project.

## Preserved Q1-Q2 visual baseline

- File: `old/HW4-Solution.pre-q3a-20260724-230819.docx`
- SHA-256:
  `6197D53B5522953BDFEF486372541E9AB082FBF259471BED34AE11891EEE4AE2`
- Accepted scope: Question 1 through Question 2(e), including the corrected
  Q2(c) block-type visual and the approved Q2(e) conclusion: the linear layer
  has eight free parameters; an invariant bias adds one parameter if "layer"
  means an affine layer.

This remains the immutable recovery baseline through Question 2(e).

## Current accepted checkpoint

- File: `HW4-Solution.docx`
- SHA-256:
  `092C293D3F0F96569B1EA02AD205CE1F18C2A504E434C541099FF92FFA5BA646`
- Accepted scope: the complete HW4 report, including all Q3/Q4 experiments,
  the three Q4 Bonus method sections, unified 13-variant result tables, and
  the complete six-part joint analysis.
- User visual approval: granted on 2026-08-02 for the exact 27-page Word
  review candidate.
- Verification: the permanent contract audit passed; the promoted file is
  byte-identical to the approved candidate; the fresh read-only Word export
  has 27 A4 pages and is pixel-identical to the approved render on every page.
- Publication status: approved and ready for the final commit and push.
- Previous accepted checkpoint is preserved as
  `old/HW4-Solution.pre-q4-rotation-20260801-211939.docx`, SHA-256
  `9B9579BF37F8AB238D9E037A49F675C3856259F8DCEC92EA7775A594C06568C1`.

## Gate record

| Part | Reasoning | Formal text | Code/tests | DOCX insertion | Mechanical audit | User visual approval | Publication |
|---|---|---|---|---|---|---|---|
| Q1 | approved | approved | n/a | accepted baseline | passes | approved | published |
| Q2(a-e) | approved | approved | n/a | accepted baseline | passes | approved | published |
| Q3(a) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(b) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(c) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(d) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q3(e) | approved | approved | passed | approved and promoted | passes | approved | published |
| Q4 mandatory | approved | approved | 15 mandatory approximately 56k-parameter runs pass | approved and promoted | passes | approved | published |
| Q4 normals bonus | approved | approved | 15 d=6 runs pass | approved and promoted | passes | approved | published |
| Q4 rotation bonus | autonomous implementation requested | approved | 3 runs and symmetry checks pass | approved and promoted | passes | approved | pending final push |
| Q4 Transformer bonus | autonomous implementation requested | approved | 5 complete runs plus one OOT run and symmetry checks pass | approved and promoted | passes | approved | pending final push |
| Q4 final consolidation | requested natural joint structure | approved | 39 saved runs validated | approved and promoted | passes | approved | pending final push |

Semantic approval never implies DOCX-format approval.

## Current Q3/Q4-aligned design

The following values supersede all earlier Q3 technical configurations.

### Common rationale

- Seed: 2319.
- All neural-network experiments require the CUDA GPU; the target machine is
  the user's RTX 3080.
- Architectures were selected with the PointNet task in Q4 in mind, while
  keeping comparisons within each Q3 subsection as close as the mathematical
  requirements allow.
- Q3(a-d) use output dimension 40 to match the Q4 classification target. Q3(e)
  alone uses output dimension 3 because it predicts the three coordinate-wise
  empirical variances.
- The approximate-invariance metric in Q3(c) is the mean output-space RMS
  error and uses the approved threshold `atol=1e-2`; maximum error is only a
  supplementary diagnostic.

### Q3(a): canonicalization

- `n=256`, `d=3`, `p=40`.
- Architecture: `768 -> 64 -> 64 -> 40`, ReLU activations, 55,976 trainable
  parameters.
- Rows are sorted lexicographically, then the canonical matrix is flattened
  and passed to the ordinary MLP.
- Recorded maximum absolute invariance error: zero; duplicate-row and
  deterministic-canonicalization checks pass.

### Q3(b): exact symmetrization

- `n=7`, `d=3`, `p=40`.
- Base architecture: `21 -> 207 -> 207 -> 40`, ReLU activations, 55,930
  trainable parameters.
- The model averages the base MLP over all `7! = 5040` permutations for each
  input; evaluation is chunked in groups of 1024 without changing the exact
  average.
- Recorded base-MLP invariance error: `0.206946`; exact symmetrized error:
  `1.49e-8`. Enumeration and finite-gradient checks pass.

### Q3(c): sampled symmetrization

- Uses the same `n=7`, `d=3`, `p=40` base MLP as Q3(b).
- One fixed subset of `B=252` distinct permutations is sampled uniformly
  without replacement and reused for both compared inputs. This is 5% of
  `S_7` and a 20-fold reduction from exact symmetrization.
- The finite-population Monte Carlo prediction for the mean RMS invariance
  error is `0.005811`.
- Across 100 comparisons, the empirical mean RMS error is `0.005727`, the
  95th percentile is `0.008496`, and the supplementary maximum diagnostic is
  `0.026787`; the approved mean-error criterion passes `atol=1e-2`.

### Q3(d): invariant DeepSets network

- `n=256`, `d=3`, `p=40`.
- Equivariant block: `3 -> 128 -> 128`, followed by mean pooling and an
  invariant head `128 -> 128 -> 40`.
- Total trainable parameters: 55,464.
- Recorded maximum absolute errors are `2.38e-7` for one equivariant layer,
  `1.79e-7` for the full equivariant stack, and `1.68e-8` after invariant
  pooling. Equivariance, invariance, and finite-gradient checks pass.

### Q3(e): augmentation study and permutation coverage

- Dataset: 1,000 examples split 700/150/150 into train/validation/test.
- Each example independently samples its mean from `Normal(0,1)` and variance
  from `Rayleigh(scale=1)`, then samples Gaussian rows. The target is the
  coordinate-wise empirical variance with divisor `n`.
- Both comparisons use an order-sensitive flattened MLP with 53,571
  parameters, MSELoss, Adam with learning rate `1e-4`, batch size 64, at most
  2,000 epochs, early-stopping patience 100, and restoration of the best
  validation checkpoint.
- For `n=256`, the architecture is `768 -> 64 -> 64 -> 3`. For the controlled
  `n=7` experiment, it is `21 -> 349 -> 130 -> 3`.
- The augmented loader applies a fresh independent random permutation to every
  example in every batch; it does not share one permutation across the batch.
- If `N=n!` and a particular example has been shown `k` times, the expected
  number of distinct permutations seen is
  `E[D_k] = N(1 - (1 - 1/N)^k)`.
- At the best augmented checkpoints, `n=7`, `k=1643` gives about 1,402
  distinct permutations, or 27.8% of `7!`; `n=256`, `k=1042` gives about
  1,042 distinct permutations, a negligible fraction of `256!`.
- Recorded `n=256` results:
  - Initial: test MSE `2.119865`, mean invariance error `0.187865`.
  - Augmented: test MSE `0.177690`, mean invariance error `0.710025`, best
    epoch 1042.
  - Control: test MSE `0.581739`, mean invariance error `0.486882`, best epoch
    45.
- Recorded `n=7` results:
  - Initial: test MSE `1.998376`, mean invariance error `0.157501`.
  - Augmented: test MSE `0.006223`, mean invariance error `0.164389`, best
    epoch 1643.
  - Control: test MSE `0.141360`, mean invariance error `0.505162`, best epoch
    544.
- The answer explains that data augmentation improves task generalization but
  does not impose exact architectural invariance. It also explains why the
  initial model can have a deceptively small invariance error: a nearly
  constant random output changes little under permutation while still being a
  poor predictor.

## Approved Q3 document structure

- Question 3 begins with a common overview covering the RTX 3080, design
  consistency with Q4, common testing principles, and shared implementation
  choices.
- Each subsection answers its own prompt first, then separates the unique
  implementation/hyperparameters, numerical results, and focused discussion.
- Hyperparameter tables use one Hebrew-labeled hyperparameter per row; results
  are in separate tables.
- Repeated architecture statements and other duplicated prose were removed.
- The Q3(e) training/validation graph is approved media with SHA-256
  `5B9ECA79502947795176D6A44B1A430D9F8A181F4B3C3372D0FC7445F16211E4`.
- The Word table of contents contains the three completed Heading 1 question
  entries and all 15 Heading 2 subsection entries. Heading 3 implementation,
  results, and discussion labels are intentionally excluded.

## Q4 mandatory benchmark checkpoint

The student approved the complete mandatory experimental design, numerical
interpretation, trade-off analysis, recommendation, formal Hebrew text, and
Word presentation on 2026-08-01. The exact 25-page approved review candidate
was promoted to `HW4-Solution.docx`, SHA-256
`21753F3AD7BB8D12FBDEEFE1F94F8A711C05E1EB5995FB4B127E0B88D2068F38`.

The promoted checkpoint includes the complete original English Question 4 wording as two
source images, the mandatory Hebrew answer only, the approved sanity and
accuracy plots, separate accuracy and runtime tables, four question-level TOC
entries, and 22 subsection-level TOC entries. Architecture and related technical
terms are written directly in standard English, including `equivariant network`
and `linear equivariant layers`. The analysis-and-reporting section now mirrors
the six required deliverables in six numbered Heading 3 sections: accuracy plot,
wall-clock runtime, trade-offs, implementation challenges, recommendation, and
the special suitability of linear equivariant layers. The permanent contract,
OOXML audit, invisible Word TOC update/reopen, read-only Word PDF
export, read-only Word direction audit, and full-page inspection of the
neighboring page and all seven Question 4 pages pass. Formal-content and visual
approval were granted, and the exact candidate was promoted to the main file.

### Data and protocol

- Source: the assignment-provided ModelNet40 `normal resampled` archive,
  SHA-256
  `DCA19B495658331BDD1656527AF7EA6D8CC4162D871E00444A7BFD945C96C9D3`.
- A compact ignored cache contains the approved 2,000 selected training-pool
  clouds and all 2,468 official test clouds, each with the first 256 XYZ rows.
- No centering, rescaling, surface normals, or rotation features are used in
  the mandatory benchmark.
- The class-stratified selected pools are nested at 5, 10, and 50 examples per
  class. Every consecutive block of five contributes four training examples
  and one validation example, yielding 4/1, 8/2, and 40/10 optimization and
  validation examples per class.
- All architectures use the same selected examples, splits, seed 2319, batch
  order, and matching initial weights where their base MLPs agree.
- Training uses CrossEntropyLoss and Adam with learning rate `1e-4`,
  `weight_decay=1e-4`, batch size 64, no dropout, and no scheduler. Early
  stopping monitors validation cross-entropy with patience 100 and restores
  the best checkpoint. The maximum is 2,000 epochs and 30 minutes per run.
- The official test set does not affect training or checkpoint selection.
  Overall accuracy is primary; mean per-class accuracy is secondary. The
  augmentation model also receives one fixed-permutation diagnostic pass.

### Architectures and verification

- Canonization: 256 points, lexicographic sorting, then
  `768 -> 64 -> 64 -> 40`, 55,976 parameters.
- Full symmetrization: the fixed first seven points and all `7! = 5040`
  permutations, with averaged logits from `21 -> 207 -> 207 -> 40`, 55,930
  parameters.
- Sampled symmetrization: the same seven-point base MLP and one fixed seeded
  subset of 252 distinct permutations, 55,930 parameters.
- Equivariant network: 256 points, equivariant `3 -> 128 -> 128`, mean pooling,
  and head `128 -> 128 -> 40`, 55,464 parameters.
- Augmentation: 256 points and the ordinary
  `768 -> 64 -> 64 -> 40` MLP, with a fresh independent row permutation for
  every training example, 55,976 parameters.
- The compact dataset, nested splits, matching initializations, parameter
  counts, logits, finite gradients, fixed sampled subset, and symmetry behavior
  all pass smoke tests on the RTX 3080. Exact-method errors are at numerical
  precision; sampled symmetrization is approximate; the ordinary MLP is
  order-sensitive.

### Approved mandatory results and interpretation

All 15 runs completed normally; none reached the 30-minute cap.

| Architecture | Selected/class | Overall accuracy | Mean class accuracy | Training time (s) |
|---|---:|---:|---:|---:|
| Canonization | 5 | 13.98% | 13.58% | 3.94 |
| Canonization | 10 | 21.27% | 19.12% | 4.95 |
| Canonization | 50 | 33.27% | 31.25% | 17.32 |
| Full symmetrization | 5 | 10.98% | 11.61% | 28.47 |
| Full symmetrization | 10 | 23.42% | 21.83% | 146.84 |
| Full symmetrization | 50 | 41.98% | 41.84% | 751.42 |
| Sampled symmetrization | 5 | 10.53% | 11.43% | 5.93 |
| Sampled symmetrization | 10 | 24.23% | 22.16% | 32.26 |
| Sampled symmetrization | 50 | 41.98% | 40.79% | 134.91 |
| Equivariant + mean pooling | 5 | 13.98% | 14.91% | 6.82 |
| Equivariant + mean pooling | 10 | 27.71% | 26.83% | 13.36 |
| Equivariant + mean pooling | 50 | 58.31% | 55.82% | 147.82 |
| Permutation augmentation | 5 | 2.11% | 5.40% | 3.75 |
| Permutation augmentation | 10 | 5.11% | 7.86% | 5.51 |
| Permutation augmentation | 50 | 24.68% | 20.77% | 68.17 |

- The augmentation model's fixed-permutation accuracies are 2.35%, 5.63%, and
  25.32%, close to its original-order accuracies but not an architectural
  invariance guarantee.
- Sampled symmetrization is 4.55-5.57 times faster than full symmetrization in
  measured training wall time, despite using 20 times fewer permutation
  evaluations; the GPU executes the larger exact batches more efficiently.
- At 50 selected examples per class, the equivariant model leads by 16.33
  percentage points over full/sampled symmetrization and by 25.04 points over
  canonization.
- Learning curves show genuine overfitting, especially for canonization. At
  its best 50-example checkpoint, training cross-entropy is 1.4985 while
  validation cross-entropy is 2.6025.
- By explicit instruction on 2026-08-01, a later approximately 5,600-parameter
  sensitivity experiment was discarded in full and must not be used in the
  homework or future comparisons. Q4 now uses only the approximately
  56,000-parameter benchmark recorded above.
- Ignored artifacts under `_qa/q4_modelnet40/` include the required sanity
  plot, accuracy plot, five learning-curve figures, per-run JSON/checkpoints,
  consolidated results JSON, and runtime CSV.

## Q4 surface-normal bonus checkpoint

The student approved the complete experimental design on 2026-08-01. Every
point is represented by `(x, y, z, nx, ny, nz)`; positions and their attached
normals are always permuted together. Canonization sorts by XYZ only and carries
the complete six-feature row. The archive contains no duplicated XYZ rows among
the 4,468 cached clouds, so this sorting rule is unambiguous on the experiment
data.

All mandatory hyperparameters, splits, seeds, point counts, hidden widths, and
stopping rules are unchanged. This is intentionally not parameter-matched:

- Canonization and augmentation use `1536 -> 64 -> 64 -> 40`, 105,128
  parameters.
- Full and sampled symmetrization use `42 -> 207 -> 207 -> 40`, 60,277
  parameters.
- The equivariant network uses `6 -> 128 -> 128`, mean pooling, and
  `128 -> 128 -> 40`, 56,232 parameters.

Both the unchanged d=3 mode and the new d=6 mode pass CUDA smoke tests. All 15
d=6 runs completed normally on the RTX 3080, and a checkpoint-only reevaluation
reproduced the saved test metrics. Raw overall-accuracy results are:

| Architecture | d=6 at 5 | d=6 at 10 | d=6 at 50 | Delta from d=3 at 5/10/50 |
|---|---:|---:|---:|---:|
| Canonization | 19.69% | 22.85% | 35.01% | +5.71 / +1.58 / +1.74 pp |
| Full symmetrization | 3.40% | 3.85% | 38.13% | -7.58 / -19.57 / -3.85 pp |
| Sampled symmetrization | 3.77% | 3.77% | 38.01% | -6.77 / -20.46 / -3.97 pp |
| Equivariant network | 21.07% | 33.87% | 68.64% | +7.09 / +6.16 / +10.33 pp |
| Permutation augmentation | 2.31% | 4.38% | 25.69% | +0.20 / -0.73 / +1.01 pp |

The student approved the interpretation and formal Hebrew answer on 2026-08-01.
The answer explains that the equivariant network benefits most because its
internal pointwise weight sharing learns reusable position-normal geometry over
all 256 points. Full and sampled symmetrization average the output but retain a
flattened base MLP without internal pointwise weight sharing; they also use only
seven points. Their nearly identical results show that the deterioration is not
caused by the Monte Carlo approximation. In the 5- and 10-example regimes their
validation cross-entropy remained close to `log(40)`, indicating that they
barely learned beyond random prediction. The interpretation also records that
the larger parameter counts of the non-equivariant models did not solve this
problem and explicitly limits the conclusion because point counts, parameter
increases, and per-architecture retuning are not controlled.

The exact 26-page review candidate is
`_qa/q4-normals/HW4-Solution.q4-normals-REVIEW.docx`, SHA-256
`9B9579BF37F8AB238D9E037A49F675C3856259F8DCEC92EA7775A594C06568C1`.
Per the student's instruction, it retains the original Q4 accuracy table and
adds the d=6 overall-accuracy delta in parentheses on a second line inside each
existing result cell; it does not add a separate bonus-results table or figure.
The approved explanation appears under the Heading 2 section
`Bonus. שימוש ב-surface normals.` The permanent OOXML audit, invisible Word TOC
update and read-only reopen, read-only Word PDF export, Q4 Word direction audit,
and full-page visual inspection all pass. The TOC contains four Heading 1 and
23 Heading 2 entries, and pages 18-21 are pixel-identical to the accepted
render. The student visually approved this exact candidate, after which it was
promoted byte-for-byte to `HW4-Solution.docx`. A fresh read-only Word export of
the promoted file is pixel-identical to the approved render on all 26 pages.
The preceding accepted checkpoint was copied to
`old/HW4-Solution.pre-q4-normals-20260801-203055.docx` before candidate work.
Generated caches, checkpoints, plots, and result tables remain isolated under
ignored `_qa/q4_modelnet40_normals/`.

## Q4 rotation-symmetry bonus checkpoint

On 2026-08-01 the student explicitly asked Codex to stop the question-by-question
interview and complete the remaining rotation Bonus autonomously because the
submission was urgent. The implemented symmetry group is `O(3)`: `SO(3)` is the
rotation subgroup, while pairwise distances also remain invariant under
reflections.

For each XYZ cloud, the implementation subtracts the point mean and divides by
the maximum centered point norm. It then forms the normalized squared-distance
matrix
`D_ij = ||x_hat_i - x_hat_j||_2^2`. For every point `i`, its per-point feature
is the sorted row `sort(D_i1, ..., D_in)`. Sorting removes the column order, so
the features are `O(3)`-invariant and permutation-equivariant: a point
permutation only permutes the feature rows.

The classifier uses all 256 XYZ points. Its DeepSets backbone is
`256 -> 72 -> 72`, followed by mean pooling and a `72 -> 72 -> 40` head. It has
55,552 parameters, closely matching the original 55,464-parameter equivariant
network. The mandatory seed, splits, optimizer, learning rate, weight decay,
batch size, epoch cap, patience, and time limit are unchanged.

CUDA verification passes:

- Maximum feature error under a combined orthogonal transformation including a
  reflection, translation, and global scaling: `1.43e-6`.
- Maximum output-logit error under that transformation: `4.47e-8`.
- Maximum permutation-equivariance error of the per-point features: `4.77e-7`.
- Maximum permutation-invariance error of the classifier output: `2.98e-8`.
- Output shape, finite loss, and finite-gradient checks pass.

All three clean final runs completed and checkpoint-only reevaluation reproduced
the saved test metrics:

| Selected/class | Overall accuracy | Mean-class accuracy | Training time (s) | Completed/best epoch |
|---:|---:|---:|---:|---:|
| 5 | 18.52% | 20.36% | 22.2 | 631 / 531 |
| 10 | 20.83% | 22.82% | 50.4 | 804 / 704 |
| 50 | 35.98% | 37.17% | 980.9 | 1230 / 1130 |

Relative to the original equivariant network, overall accuracy changes by
`+4.54`, `-6.88`, and `-22.33` percentage points. The draft interpretation is
that exact `O(3)` invariance provides a helpful inductive bias in the smallest
data regime, but independently sorting every distance row loses correspondence
between distances in different rows. The aligned ModelNet40 data may also
contain useful absolute-orientation information that this representation
removes. Runtime is about 3.3-3.9 times higher because building and sorting the
`n x n` matrix is at least quadratic in set size, while the original pointwise
backbone scales linearly in `n`.

The earlier rotation-only Word draft contained stale hard-coded size-50 values
(`42.02%`, `43.30%`, and `576.1` seconds) that did not match the final saved
checkpoint. It is superseded and must not be used. The corrected values above
are read directly from
`_qa/q4_modelnet40_rotation/runs/rotation_invariant-selected-50.json` and are
included in the combined final candidate described below.

`HW4-Solution.docx` remains unchanged pending visual approval. The accepted
pre-rotation checkpoint is archived as
`old/HW4-Solution.pre-q4-rotation-20260801-211939.docx`. Generated experiment
artifacts are isolated under ignored `_qa/q4_modelnet40_rotation/`.

## Q4 Transformer Bonus checkpoint

The two required Transformer variants use one shared order-sensitive base
model: input projection `d -> 56`, fixed sinusoidal positional encoding, two
`TransformerEncoder` layers with four attention heads and feed-forward width
104, ReLU, no dropout, final encoder `LayerNorm`, mean pooling, and a
`56 -> 56 -> 40` head. Each variant has 55,408 trainable parameters, close to
the common approximately 56,000-parameter budget.

- Canonization + Transformer sorts all 256 XYZ points lexicographically before
  the Transformer and is exactly invariant.
- Full symmetrization + Transformer uses the first seven points and averages
  logits over all `7! = 5040` permutations. Permutations are processed in CUDA
  chunks of 512; training uses microbatch size 32 and gradient accumulation to
  preserve effective batch size 64.
- CUDA smoke tests confirm that the base Transformer is order-sensitive, the
  canonized model has zero maximum permutation error, and the exact averaged
  model has maximum error `2.98e-8`. Output-shape, positional-encoding,
  finite-loss, and finite-gradient checks pass.

Recorded results are:

| Variant | Selected/class | Overall accuracy | Mean-class accuracy | Training time (s) | Status |
|---|---:|---:|---:|---:|---|
| Canonization + Transformer | 5 | 21.03% | 21.91% | 24.98 | complete |
| Canonization + Transformer | 10 | 31.89% | 30.37% | 32.50 | complete |
| Canonization + Transformer | 50 | 56.08% | 54.72% | 120.52 | complete |
| Full symmetrization + Transformer | 5 | 17.75% | 16.90% | 760.74 | complete |
| Full symmetrization + Transformer | 10 | 27.15% | 27.58% | 1495.32 | complete |
| Full symmetrization + Transformer | 50 | 42.67% | 40.36% | 1800.2 | OOT |

The size-50 exact-symmetrization run reached the 30-minute cap during epoch 94;
the reported official-test result comes from the best completed checkpoint at
epoch 93. Generated artifacts are isolated under ignored
`_qa/q4_transformer_bonus/`.

## Combined Q3/Q4 final review candidate

The full Q3/Q4 requirements were audited against `HW4.pdf`. Question 3 already
contains every requested construction, test, tolerance/result, before/after
Q3(e) measurement, and focused discussion. Question 4 is reorganized in the
natural assignment order: shared protocol; five mandatory architecture
subsections; separate method subsections for Transformer, surface normals, and
rotation/pairwise-distance Bonuses; one unified accuracy table; one unified
runtime table; and a joint six-part analysis matching the requested reporting
items.

The unified tables contain all 13 variants across the three data sizes, and
every extension row begins with `Bonus:`. The mandatory five-curve accuracy
plot is retained for readability; Bonus results are reported in the joint
tables rather than adding eight overlapping curves. The consolidated result
ledger validates all 39 saved experiment records under
`_qa/q4_joint_results/`.

The exact approved 27-page review candidate is
`_qa/q4-final/HW4-Solution.q4-final-REVIEW.docx`, SHA-256
`092C293D3F0F96569B1EA02AD205CE1F18C2A504E434C541099FF92FFA5BA646`.
The permanent OOXML audit passes, Word updated the TOC and reopened the file
read-only, and the TOC contains four Heading 1 entries and 25 Heading 2 entries.
The read-only Microsoft Word export contains 27 A4 pages. Every page was
inspected at full-page resolution: the RTL direction, headings, native Word
equations, figures, and both unified tables are clean, with no clipping,
overlap, or stray rotation-only table. The paragraph-level Word audit found no
Hebrew paragraph with a non-RTL reading order. The user approved this exact
candidate on 2026-08-02, and it was promoted byte-for-byte to
`HW4-Solution.docx`. A fresh promoted-file audit and Word render pass; all 27
promoted pages are pixel-identical to the approved render.

## Final submission package

`HW4.pdf` requires exactly two uploaded files: one PDF report and one ID-named
ZIP containing all runnable Q3/Q4 code with dependencies stated at the top.
For the single student listed on the report, the final files are:

- `submission/209517846_hw4.pdf`, SHA-256
  `1417139D8492E4EE0FAF61EB2691381BE69B6441D9B68EAF3B669128C5385C49`.
  It is a fresh read-only Microsoft Word export of the promoted DOCX and has 27
  valid A4 pages.
- `submission/209517846_hw4.zip`, SHA-256
  `CA78FB7D70926C403D063A1EB6C4BEAB6BB0E1CDB7D15250011292EE65803465`.
  Its archive root contains exactly `q3_set_networks.py`,
  `q4_modelnet40.py`, `q4_joint_results.py`, and `requirements.txt`.

The code was simplified per the student's final instruction: each Python file
has only a one-line dependency header, and the Q3/Q4 implementation contains
only two short comments. The extracted archive is byte-identical to the source
files, compiles successfully, exposes the expected command-line entry points,
and passes the Q3 symmetry checks and full Q4 CUDA smoke suite from the packed
copy. No datasets, checkpoints, generated plots, QA files, or outer directory
are present in the ZIP. The `submission/` directory contains exactly the two
files required for Moodle.

## Infrastructure

- Repository: `https://github.com/uriyaca23/DeepLearningGroups.git`
- Branch: `main`
- Project virtual environment: `.venv`, Python 3.12.13.
- `requirements.txt` records the CUDA 12.6 PyTorch build.
- Superseded DOCX checkpoints belong only under `old/`.
- Candidates, renders, and generated reports belong under ignored `_qa/`.
- Give the user the review-file path rather than opening Word visibly; open it
  only when the user explicitly requests that.

## Precise next action

Commit and push the approved implementation, promoted report, preserved prior
checkpoint, and final two-file submission package. Then give the student the
two exact paths under `submission/` for Moodle upload.

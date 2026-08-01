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
- `q4_modelnet40.py`: approved Question 4 mandatory benchmark implementation;
  the mandatory results and their interpretation are approved.

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
  `21753F3AD7BB8D12FBDEEFE1F94F8A711C05E1EB5995FB4B127E0B88D2068F38`
- Accepted scope: Question 1 through the mandatory part of Question 4,
  including the complete Q3/Q4-aligned experiments, Q4 analysis-and-reporting
  structure, and expanded Heading 1-2 table of contents.
- User visual approval: granted on 2026-08-01 for the exact 25-page Word
  review candidate.
- Verification: the permanent contract audit passed; the read-only Word
  export has 25 A4 pages; pages 2-18 are pixel-identical to the previous
  accepted checkpoint; the updated contents page and pages 19-25 are
  pixel-identical to the approved review render; the Q4 Word direction audit
  reports no Hebrew-direction anomalies; the promoted file is byte-identical
  to the approved candidate.
- Publication status: approved and promoted locally; Git publication is
  pending.
- Previous accepted checkpoint archived as
  `old/HW4-Solution.pre-q4-draft-20260801-162500.docx`, SHA-256
  `7D382D3C8A0069BAF66506CCB45CFE3941AEE19F01C7EC1E64ED1AA2AE2DE39F`.

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
| Q4 mandatory | approved | approved | 15 mandatory approximately 56k-parameter runs pass | approved and promoted | passes | approved | pending publication |
| Q4 bonuses | not started | not started | not started | not started | not started | not started | not published |

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
  - Initial: test MSE `2.119865`, RMS invariance error `0.187865`.
  - Augmented: test MSE `0.177690`, RMS invariance error `0.710025`, best
    epoch 1042.
  - Control: test MSE `0.581739`, RMS invariance error `0.486882`, best epoch
    45.
- Recorded `n=7` results:
  - Initial: test MSE `1.998376`, RMS invariance error `0.157501`.
  - Augmented: test MSE `0.006223`, RMS invariance error `0.164389`, best
    epoch 1643.
  - Control: test MSE `0.141360`, RMS invariance error `0.505162`, best epoch
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

Publish the approved mandatory Q4 checkpoint to `origin/main`. Then continue
the collaborative workflow with the first Q4 bonus: present its original
English wording, give brief orientation only, and ask the student one focused
question about how surface normals should change the input and the controlled
comparison. Do not implement or insert the bonus before the normal reasoning,
formal-answer, and approval gates pass.

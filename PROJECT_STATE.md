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
  `7D382D3C8A0069BAF66506CCB45CFE3941AEE19F01C7EC1E64ED1AA2AE2DE39F`
- Accepted scope: Question 1 through Question 3(e), including the complete
  Q3/Q4-aligned rewrite and expanded Heading 1-2 table of contents.
- User visual approval: granted on 2026-08-01 for the exact 18-page Word
  review candidate.
- Verification: the contract audit passed; two independent read-only Word
  exports were pixel-identical on all 18 pages; every page was visually
  inspected; the promoted file is byte-identical to the approved candidate.
- Publication status: published to `origin/main` on 2026-08-01.
- Previous accepted checkpoint archived as
  `old/HW4-Solution.pre-q3-aligned-20260801-131231.docx`, SHA-256
  `F7F654E5CA53FFB74C628229BD95DEE2A21D697F0368DCBCA1D77261FB1EF2CB`.

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
| Q4 | in progress | not started | not started | not started | not started | not started | not started |

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

Present Question 4 verbatim from `HW4.pdf`, give only brief orienting context,
and ask the student for the first part of their reasoning. Continue one focused
question at a time, challenge mistakes with progressively stronger hints, and
formalize only the answer the student understands and approves. Do not
implement Q4 or insert it into the DOCX before the design and formal Hebrew
answer pass the normal explicit approval gates.

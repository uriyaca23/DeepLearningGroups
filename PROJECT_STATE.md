# HW4 Project State

## Objective

Complete HW4 collaboratively as a learning exercise.

Codex must not independently solve new questions for the student. Each answer
and algorithm must originate from the student's reasoning and be developed
through questions, progressively stronger hints, correction, and explicit
approval.

## Sources of truth

- `HW4.pdf`: authoritative assignment wording and requirements.
- `HW4-Solution.docx`: authoritative written-solution checkpoint.
- `q3_set_networks.py`: authoritative runnable implementation checkpoint for
  the approved Question 3 components.
- `requirements.txt`: reproducible Python dependency manifest.
- Course lecture notes and tutorials available through the course-materials
  shortcut: foundation for expected knowledge and solution methods.
- Previous homework documents: formatting and stylistic reference.

Do not use or modify LocationPipeline.

## Communication and writing

- Discuss reasoning in English.
- Write final submitted answers in polished Hebrew.
- Present each assignment question verbatim in its original English.
- Include the original English question wording in the final document.
- Use the existing Hebrew RTL formatting and style.
- Match the concise, professional explanatory style established in HW1-HW3:
  short logical paragraphs, one claim per paragraph, brief inline course
  references, only essential displayed equations, and a direct conclusion.
- Write complete, rigorous, well-presented final answers without redundant
  restatement.
- Do not add a figure, sketch, diagram, or other visual to a written answer
  unless the user explicitly requests it. An assignment prompt mentioning a
  possible sketch is not, by itself, authorization to add one.
- The formal answer need not preserve the student's original wording.

## Mandatory Word formatting implementation

- Every Hebrew prose paragraph must use Word-native RTL paragraph direction
  and visual right alignment explicitly. In this document, Word automation
  reports `ReadingOrder = 0` and logical-start `Alignment = 0` for a visually
  right-aligned RTL paragraph; forcing `Alignment = 2` moves RTL headings to
  the visual left. Therefore verify the Word values after reopening, preserve
  the native `w:rtl` heading formatting used by the approved Q1/Q2 headings,
  and confirm the physical right edge in a rendered full-page view.
- Do not infer RTL from Hebrew characters, run direction, inherited defaults,
  or visual inspection of a cropped text selection.
- Hebrew prose must inherit the document's established DejaVu Sans 12 pt
  formatting. Do not introduce a direct Times New Roman font override.
- Displayed mathematics must be a real Word equation in its own centered LTR
  paragraph (`ReadingOrder = 1`), never ordinary text inside a Hebrew
  paragraph.
- Use Word's semantic heading styles throughout: each question title is
  `Heading 1`, each named or lettered subsection within a question is
  `Heading 2`, and any genuinely nested subsection introduced later should
  continue the hierarchy with `Heading 3`.
- Match the actual HW1 heading definitions rather than Word's generic
  defaults: `Heading 1` is regular-weight 16 pt with 12 pt before and 0 pt
  after; `Heading 2` is regular-weight 13 pt with 2 pt before and 0 pt after.
  Both use the HW1 accent color and the document's DejaVu Sans typography.
- Every heading paragraph must remain explicitly RTL/right in Word. In this
  document, preserve the verified native setting (`ReadingOrder = 0`,
  logical-start `Alignment = 0`) together with the Hebrew `w:rtl` runs and
  inline Word equations; never clear or rebuild their run formatting merely
  to apply a heading style.
- Before delivering a DOCX edit, compare the affected page against an approved
  neighboring page in a full Word-window view, then reopen the saved file and
  audit the paragraph properties above.

## Required learning workflow

For each theoretical subpart:

1. Present the original question verbatim.
2. Give brief general context and point to relevant course material.
3. Ask the student for their reasoning.
4. Challenge mistakes, omissions, and unjustified steps.
5. Give progressively stronger hints rather than immediately revealing the
   solution.
6. Continue until the reasoning is completely correct.
7. Draft a full formal Hebrew answer.
8. Verify its correctness with the student.
9. Wait for the student's explicit `approved`.
10. Only then update and save `HW4-Solution.docx`.

Never write a new subpart into the document before explicit approval.

## Course-material references

Build from the lectures and tutorials instead of unnecessarily rederiving
course results.

When an answer uses a result or solved example shown in class, include a brief
inline reference in the Hebrew solution. Identify the lecture or tutorial and,
when reasonably available, the relevant page, slide, theorem, or exercise.
Keep references accurate but unobtrusive.

## Programming workflow

For every Q3 and Q4 programming subpart:

1. Present the question verbatim.
2. Ask the student to propose the algorithm, architecture, or pseudocode.
3. Challenge conceptual mistakes using progressively stronger hints.
4. Agree on a correct design.
5. Wait for explicit approval before implementing it.
6. Translate the approved design into clean, runnable PyTorch code.
7. Run approved tests and experiments autonomously.
8. Report all results.

If testing exposes a conceptual error, return to the student with hints before
changing the approved design.

Codex may autonomously fix purely mechanical problems such as syntax errors,
paths, serialization, or formatting, but must explain what changed.

## Approved work

- [x] Q1: discussed, understood, formally written, and approved.
- [x] Q2(a): discussed, understood, formally written, and approved.
- [x] Q2(b): discussed, understood, formally written, and approved.
- [x] Q2(c) written explanation: discussed, understood, formally written, and
      approved.
- [x] Q2(c) visual correction: the four unnecessary small matrices were
      removed and only the required block-type parameter-sharing visual was
      retained.
- [x] Q2(d): discussed, understood, formally written, and approved.
- [x] Q2(e): discussed, corrected, formally written, and approved. The written
      answer gives 8 parameters for the linear map and separately explains
      that an affine extension has 9 parameters in total. It is written
      concisely in the HW1-HW3 style and contains no redundant figure.
- [x] Q3(a) design: discussed and explicitly approved. The network
      lexicographically canonizes the rows, flattens the canonical matrix, and
      applies a two-layer MLP with one ReLU.
- [x] Q3(a) implementation and tests: implemented in `q3_set_networks.py`.
      With seed 2319, n=20, d=3, p=4, hidden width 32, and
      `atol=1e-5, rtol=0`, the required invariance test passes with maximum
      absolute error 0. A partial-tie/duplicate-row test, 125 additional
      permutation checks, a gradient check, and a CUDA smoke test also pass.
- [x] Q3(a) Hebrew report text and insertion into `HW4-Solution.docx`:
      explicitly approved and written. The original English prompt is included
      in the established `Original Question` style; the Hebrew answer uses
      `Heading 1`/`Heading 2`, native RTL formatting, and centered Word
      equations. The Word-rendered ten-page document passed full visual QA.
- [ ] Q3(b-e).
- [ ] Q4.
- [ ] Final report verification and export.
- [ ] Runnable-code ZIP.

## Project infrastructure

- Local Git repository initialized on branch `main`.
- Remote `origin` is
  `https://github.com/uriyaca23/DeepLearningGroups.git`.
- The approved project checkpoint is published directly to the public
  repository's `main` branch.
- Project virtual environment: `.venv`, Python 3.12.13.
- Installed project dependencies: PyTorch 2.13.0 CPU and NumPy 2.5.1.
- A separate installed Python 3.8 runtime has PyTorch 2.4.1 with CUDA 12.4
  and an NVIDIA GeForce RTX 3080; Q3(a) also passed on that GPU.

## Current document checkpoint

- `HW4-Solution.docx` is the authoritative current solution.
- It contains completed work through Q3(a).
- Q2(e), including its affine clarification, is on page 9.
- Q3(a), including the original English prompt, proof, architecture, and test
  report, begins on page 9 and continues on page 10.
- Superseded DOCX checkpoints are stored under `old/`, not in the main folder.

## Next learning step

Continue interactively to Q3(b), beginning with the original English wording,
brief course context, and the student's proposed design.

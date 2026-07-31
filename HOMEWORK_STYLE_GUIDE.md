# Homework collaboration and document style guide

Version: 1.0
Status: authoritative for this project
Scope: HW4 conversation, reasoning, writing, code, and Word document work

This file consolidates the user's repeated instructions and the actual
structure of the accepted homework documents. It is deliberately fail-closed:
when a detail is not covered or two authorities appear inconsistent, ask the
user rather than inventing a new style.

## 1. Authority and change control

### R-AUTH-1 — Precedence

Use this order:

1. The user's latest explicit instruction.
2. This guide.
3. `PROJECT_STATE.md` for current progress only.
4. The accepted baseline and named exemplars below.
5. Generic tool or Word behavior.

Never infer a rule from a rejected candidate or from agent memory when an
authority above is available.

### R-AUTH-2 — Stable guide

This guide contains constant rules. Do not modify it merely because an agent
used a different implementation. Change it only after explicit user approval.

### R-AUTH-3 — Resolved exemplar profile

HW1-HW3 represent one intended homework format. Their underlying Word package
internals contain a few minor historical differences, so the exact
machine-enforced profile is resolved explicitly:

- `old/HW4-Solution.pre-q3a-20260724-230819.docx` is the accepted visual and
  structural baseline through Q2(e), SHA-256
  `6197D53B5522953BDFEF486372541E9AB082FBF259471BED34AE11891EEE4AE2`.
- The accepted HW4 baseline controls page geometry, body typography, header,
  table of contents, equation placement, prompt-image placement, and the
  user-corrected Heading 2 pattern.
- The user-selected HW1 treatment controls question-heading appearance; its
  resolved Heading 1 profile is already present in the accepted HW4 baseline.
- HW1–HW3 establish the writing rhythm and the rule that the original English
  assignment wording appears as a source excerpt, not as newly typeset prose.
- The current Q3(a) block is rejected as a formatting exemplar. In particular,
  its invented `Original Question` style must never be copied.
- The mechanical contract records a few exact, text-hashed legacy deviations
  in unchanged accepted Q2(d) prose. They are preserved to avoid overwriting
  the user's approved manual edits. They are not style exemplars, may not be
  copied into new content, and may not be expanded without explicit approval.

Reference fingerprints used to derive this guide:

- HW1 solution:
  Drive file ID `1S0Xhfjlxo43XhIL8zrVH1O0Oe_JNub_V`, SHA-256
  `79BD42F277093CE6D8407AB43A275EFF8D1A558AF4FC681591DB1E1E3730AE79`
- HW2 solution:
  Drive file ID `1UaXcGNvNPbwL7vCj66zNTbZjx-fDAGhi`, SHA-256
  `5417789D40383DA3D984114250B49351B9118A7B3E4B9F1CB44CF0A6902DA45E`
- HW3 solution:
  Drive file ID `1W6HVI3E8rmXpbTIggysv2CH1xLdGFPHc`, SHA-256
  `0FEA138B3C0061B89A207DC5B1126F8D7F7D32E700112320F23B12380AAB3636`

Retrieve these exact files through the connected Google Drive when the local
QA copies are absent. Do not substitute similarly named homework from another
course or year.

## 2. Collaboration and learning workflow

### R-COLLAB-1 — Language

- Discuss the homework with the user in English.
- Write the submitted solution in polished Hebrew.
- Present assignment wording verbatim in its original English.

### R-COLLAB-2 — The reasoning must come from the student

Codex is a tutor and editor, not an independent solver.

For every new theoretical question or subpart:

1. Present the original English wording.
2. Give only brief orientation and point to the relevant lecture or tutorial.
3. Ask one focused question about the student's reasoning.
4. Challenge errors, missing cases, unjustified steps, and imprecise language.
5. Give progressively stronger hints only as needed.
6. Continue until the student's reasoning is fully correct.
7. Formalize that reasoning into a complete Hebrew answer.
8. Independently verify that the formal answer is correct, complete, and
   consistent with the assignment and course material.
9. Show the verified formal answer and wait for the student's explicit
   approval and confirmation of understanding.
10. Only after both gates pass may it enter a DOCX candidate.

The final prose does not need to repeat the student's words verbatim. It must
faithfully formalize the approved reasoning and may improve organization,
notation, completeness, and precision.

### R-COLLAB-3 — Build from course knowledge

- Use lecture notes, tutorials, and already-solved class examples as the
  student's starting knowledge.
- Do not rederive an established course result merely to make the answer
  longer.
- When the student asks for a refresher, present the closest studied example
  and the transferable method before returning to the homework question.
- Brief general direction is welcome; giving away the unapproved solution is
  not.

### R-COLLAB-4 — Course references

When the solution uses a theorem, construction, or solved example from class,
include one brief inline Hebrew reference. Name the lecture or tutorial and,
when reasonably available, the exact slide, page, theorem, or exercise.
References must be accurate and useful, but not overdone.

### R-COLLAB-5 — Programming parts

For Q3 and Q4:

1. Present the exact prompt.
2. Ask the student for the algorithm, architecture, or pseudocode.
3. Correct the concept through questions and progressively stronger hints.
4. Agree on the complete design.
5. Wait for explicit approval before implementation.
6. Implement the approved design cleanly in PyTorch.
7. Run the approved tests and experiments autonomously.
8. Report the results and any conceptual failure honestly.

Syntax, paths, serialization, and other purely mechanical defects may be fixed
autonomously, but the user must be told what changed. A conceptual change
returns to the interactive reasoning workflow.

## 3. Hebrew writing standard

### R-WRITE-1 — Voice

Use the concise, professional style established by HW1-HW3 and the user's
accepted Claude Fable answers:

- formal mathematical Hebrew;
- short logical paragraphs;
- one main claim or transition per paragraph;
- necessary definitions before use;
- displayed equations only when they clarify an actual step;
- a direct conclusion.

The answer must still be full, rigorous, and pleasant to read. Concise does not
mean omitting a proof obligation, case, assumption, definition, test choice, or
conclusion.

### R-WRITE-2 — What to avoid

Do not:

- restate the entire prompt in Hebrew;
- add generic introductions or summaries;
- repeat the same justification in prose and equations;
- call a step "obvious" or "trivial" when its justification matters;
- add tangential derivations;
- mimic the student's brief wording when it is incomplete or informal;
- introduce unnecessary section headings, bullets, tables, or figures;
- use inflated language, chatty filler, or meta-commentary inside the answer.

### R-WRITE-3 — Proof organization

Use the lightest structure that makes the proof rigorous:

- state the setup or intended implication;
- prove each necessary direction or case;
- explain the key inference immediately beside it;
- conclude once.

Use a proof-ending marker only for an actual proof and follow the convention in
the accepted neighboring answer. Do not append one to a programming report or
an explanatory calculation merely for decoration.

### R-WRITE-4 — Terminology and mixed language

- Prefer the terminology used in the course.
- Use the standard English name for technical terms when a Hebrew translation
  or transliteration would sound uncommon or awkward. Examples include `MLP`,
  `PyTorch`, `torch.allclose`, `argsort`, `batch size`, and `batch order`.
- Keep notation consistent throughout an answer.
- Isolate English and mathematics correctly inside RTL prose; never switch the
  whole Hebrew paragraph to LTR merely because it contains an English token.

## 4. Document structure

### R-STRUCT-1 — Existing document is the template

Always edit a copy of the current accepted DOCX. Clone the nearest accepted
element with the same semantic role. Never:

- start from a blank DOCX;
- apply generic Word defaults;
- rebuild the document globally;
- normalize all direct formatting;
- overwrite or discard the user's manual corrections.

### R-STRUCT-2 — Front matter and header

Preserve the accepted title, table of contents, page system, and header. The
header remains one tab-positioned line with:

- student name at the visual right;
- student ID in the visual center;
- date at the visual left.

Do not recreate or restyle the header while adding an answer. Verify all three
fields on every rendered page.

### R-STRUCT-3 — Heading hierarchy

Use real Word paragraph styles:

- every question title: `Heading 1`, e.g. `שאלה 3`;
- every named or lettered subsection: `Heading 2`, e.g.
  `סעיף א. רשת מבוססת קנוניזציה.`;
- a genuinely nested subsection only: `Heading 3`.

Do not simulate a heading with bold text, a larger font, or manual numbering.
Do not use Heading 2 for ordinary transitions inside a proof.

### R-STRUCT-4 — Original English prompt

The submitted DOCX must contain the complete original English wording.

The established form is a high-resolution source excerpt rendered directly
from `HW4.pdf`, not retyped or restyled text. This is how HW1–HW3 and accepted
HW4 Q1–Q2 preserve the prompt.

For each question:

- place the complete prompt excerpt immediately after its `Heading 1`;
- include all wording, hints, subparts, and prompt-supplied diagrams;
- use consecutive excerpts if the source spans more than one PDF page;
- crop only surrounding page whitespace, never content;
- preserve aspect ratio;
- center the excerpt with no border or caption;
- use the accepted prompt-image paragraph pattern (`Body Text`, centered,
  0 pt before, 8 pt after, keep lines);
- fit it within the accepted content width (normally 6.05–6.30 inches);
- inspect it at 100% to ensure every symbol remains readable.

A source excerpt is not an answer figure. It is required verbatim assignment
context.

Forbidden:

- typing the prompt into an invented paragraph style;
- OCR substitution;
- reducing it to small gray text;
- using Aptos 9 pt or the rejected `Original Question` style;
- including only part of the wording while calling it complete.

### R-STRUCT-5 — Answer order

The order is:

1. `Heading 1` for the question;
2. full original English source excerpt(s);
3. `Heading 2` for the relevant subsection, when the question has subparts;
4. approved Hebrew answer;
5. later `Heading 2` subsections in assignment order.

Do not repeat a full prompt excerpt that is already present and readable.

### R-STRUCT-6 — Figures and diagrams

Do not add an answer figure, diagram, sketch, visualization, or decorative
graphic unless the user explicitly asks for it.

- If the prompt merely says a sketch "may" be used, ask first.
- If the prompt requires a visual, include only the minimal visual needed.
- Preserve the already approved Q2(c) block-type sharing diagram.
- Never add redundant component matrices or explanatory graphics.

### R-STRUCT-7 — Lists and tables

Use prose unless a list or table is genuinely the clearest form.

- Use Word-native bullets and numbering, never typed markers.
- Use a table only for real row/column comparison data.
- Do not package ordinary proof prose in a table.
- Match accepted indentation, wrapping, and spacing.

### R-STRUCT-8 — Version placement

- `HW4-Solution.docx` is the sole permitted working/submission DOCX in the
  main folder. Its accepted authority depends on the gates recorded in
  `PROJECT_STATE.md`; the filename alone never implies approval.
- Before promotion, move the DOCX being replaced to `old/` with a timestamped
  descriptive name, even if that checkpoint was rejected.
- Candidate files and renders belong under ignored `_qa/`.
- Never leave superseded candidates in the main folder.

## 5. Exact Word style contract

The machine-readable counterpart is
`style/homework_style_contract.json`. The values below describe the accepted
profile; they are not suggestions.

### R-WORD-1 — Page geometry

| Property | Required value |
|---|---:|
| Page | A4 portrait |
| Width | 11906 twips |
| Height | 16838 twips |
| Top margin | 1134 twips |
| Bottom margin | 1134 twips |
| Left margin | 1247 twips |
| Right margin | 1247 twips |
| Header distance | 652 twips |
| Footer distance | 652 twips |

Do not change page geometry while adding an answer.

### R-WORD-2 — Body

- Font: DejaVu Sans.
- Size: 12 pt (`w:sz=24`, `w:szCs=24`).
- Color: black.
- Hebrew language: `he-IL`.
- Normal paragraph spacing: 10 pt after.
- Hebrew body paragraphs are Word-native RTL and visually right-aligned.

Do not introduce Times New Roman, Calibri, Aptos, or another direct font
override into Hebrew answer prose.

### R-WORD-3 — Heading 1

- Word style: `Heading 1`.
- Font: DejaVu Sans.
- Size: 16 pt.
- Weight: regular, not bold.
- Color: `#365F91`.
- Space before: 12 pt.
- Space after: 0 pt.
- Keep with next and keep lines together.
- Outline level: 0.
- Word-native RTL; visually right-aligned.

### R-WORD-4 — Heading 2

- Word style: `Heading 2`.
- Font: DejaVu Sans.
- Size: 13 pt.
- Weight: regular, not bold.
- Color: `#365F91`.
- Space before: 2 pt.
- Space after: 0 pt.
- Keep with next and keep lines together.
- Outline level: 1.
- Word-native RTL; visually right-aligned.

### R-WORD-4A — Heading 3

Use only for a genuinely nested subsection:

- Word style: `Heading 3`.
- Font: DejaVu Sans.
- Size: 12 pt.
- Weight: regular, not bold.
- Color: `#365F91`.
- Space before: 2 pt.
- Space after: 0 pt.
- Keep with next and keep lines together.
- Outline level: 2.
- Word-native RTL; visually right-aligned.

### R-WORD-5 — RTL implementation

Hebrew prose and headings must be true RTL paragraphs in Word, not text that
only happens to contain Hebrew characters.

For the accepted document, reopening in Word reports:

- `ReadingOrder = 0` for RTL;
- logical-start `Alignment = 0`, which renders at the physical right edge.

Forcing `Alignment = 2` moved these RTL headings to the physical left in this
document. Therefore:

- preserve the accepted `w:bidi`/RTL paragraph and run properties;
- preserve logical-start alignment or the cloned accepted alignment;
- never infer correctness from run direction alone;
- never validate a cropped selection—inspect the full page and physical edge;
- reopen the saved candidate and check it again.

English source excerpts remain visually LTR because they are rendered source
images. Any exceptional English text paragraph must be explicitly LTR and
isolated from Hebrew prose.

### R-WORD-6 — Mathematics

- Display mathematics must be a real Word OMML equation.
- Put each displayed equation in its own centered LTR paragraph.
- Do not type a display equation as ordinary text inside an RTL paragraph.
- Inline mathematics must use Word math or a properly isolated LTR run without
  changing the surrounding Hebrew paragraph direction.
- Use only the equations needed for the reasoning.
- Match the size and spacing of an accepted neighboring equation paragraph by
  cloning it.

### R-WORD-7 — Table of contents

After adding or changing headings:

- update the Word table of contents;
- confirm every question appears at Heading 1 level;
- do not expose internal proof headings unless they are genuine semantic
  headings;
- verify page numbers after the final Word render.

## 6. Candidate and approval workflow

### R-FLOW-1 — Before editing

1. Read `AGENTS.md`, this guide, and `PROJECT_STATE.md`.
2. Identify the exact answer text and the user's explicit approval.
3. Inspect the accepted neighboring pages and the accepted baseline.
4. Run the structural audit on the starting document.
5. Record its SHA-256.
6. Create a timestamped backup under `old/`.
7. Work only on a candidate under `_qa/`.

### R-FLOW-2 — Editing

- Make the smallest local change.
- Clone accepted headings, source-excerpt paragraphs, Hebrew body paragraphs,
  equation paragraphs, and lists.
- Preserve untouched package parts and user manual edits.
- Do not create a new paragraph style unless the user explicitly approves a
  guide change requiring one.

### R-FLOW-3 — Mechanical verification

Run:

```powershell
.\.venv\Scripts\python.exe scripts\audit_homework_docx.py `
  _qa\candidate.docx `
  --contract style\homework_style_contract.json
```

The audit must return exit code 0. A warning is not permission to ignore a
rule. Fix the candidate or ask the user.

The tracked wrapper may be used from this Windows workspace:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\verify_homework.ps1 `
  -DocxPath _qa\candidate.docx `
  -ContractPath style\homework_style_contract.json
```

### R-FLOW-4 — Word visual verification

Word is the acceptance renderer for this Hebrew document.

1. Open the exact candidate in Word.
2. Refresh the table of contents and fields.
3. Save, close, and reopen it.
4. Export it to PDF and render page images.
5. Inspect every changed or reflowed page plus both neighboring pages at 100%.
6. Compare all unaffected pages against the accepted baseline.
7. Verify physical RTL alignment, font consistency, headings, header fields,
   prompt readability, equations, lists, page breaks, and absence of clipping.

Export the exact saved candidate read-only with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\render_homework_word.ps1 `
  -DocxPath _qa\candidate.docx `
  -PdfPath _qa\render\candidate.pdf
```

LibreOffice may be used as a secondary compatibility check. It does not replace
Word for acceptance.

### R-FLOW-5 — User visual approval and promotion

After the answer text is approved and the candidate passes mechanical and Word
checks:

1. Show the affected full-page render(s) to the user.
2. State exactly what changed and what was verified.
3. Wait for explicit visual approval.
4. Promote the candidate to `HW4-Solution.docx`.
5. Update `PROJECT_STATE.md` using separate content, audit, visual-approval,
   and publication gates.
6. Commit and push only the promoted, approved checkpoint.

Never equate semantic answer approval with DOCX-format approval.

## 7. Final checklist

Before saying a homework DOCX is finished, all answers must be "yes":

- Did the reasoning originate from the student?
- Did Codex verify that the formal answer is correct and complete?
- Was the formal Hebrew answer explicitly approved?
- Is the original English prompt complete and present as a readable source
  excerpt?
- Are question and subsection titles real Heading 1/Heading 2 paragraphs?
- Do their exact font, size, weight, color, and spacing match the contract?
- Is every Hebrew prose paragraph truly RTL and physically right-aligned?
- Is Hebrew answer text DejaVu Sans 12 pt?
- Are displayed equations real centered LTR Word equations?
- Are course references brief and accurate?
- Are there no unapproved or redundant figures?
- Is the header unchanged and correct on every rendered page?
- Is the table of contents current?
- Did the structural audit pass?
- Did the saved/reopened Word render pass?
- Did the user approve the affected full-page render?
- Is the former checkpoint under `old/` and the main folder clean?
- Does `PROJECT_STATE.md` report each gate honestly?

If any answer is "no" or unknown, the document is not ready.

## 8. Traceability of prior corrections

This table records the repeated feedback that led to the rules above. Future
agents must treat each row as a regression test, not as optional background.

| Prior correction | Permanent rule |
|---|---|
| "We talk in English, write in Hebrew." | Discussion stays in English; submitted prose is polished Hebrew (`R-COLLAB-1`). |
| Present the full question in its original English. | Show it verbatim during tutoring and preserve it in Word as a complete readable source excerpt (`R-COLLAB-1`, `R-STRUCT-4`). |
| The answer must come from the student. | Ask one focused question at a time, challenge the reasoning, and use progressively stronger hints (`R-COLLAB-2`). |
| Do not write before approval. | Codex first verifies correctness; the student's explicit approval authorizes insertion into a candidate, while later rendered visual approval authorizes promotion (`R-COLLAB-2`, `R-FLOW-5`). |
| Build on lecture/tutorial results rather than rederive them. | Use course results as the starting point and cite them briefly and precisely (`R-COLLAB-3`, `R-COLLAB-4`). |
| Formalize brief student explanations into a full correct answer. | Preserve the approved reasoning while improving rigor, notation, and presentation (`R-COLLAB-2`, `R-WRITE-1`). |
| Match the concise Claude Fable/HW1-HW3 writing voice. | Use short logical paragraphs, essential equations, and a direct conclusion without filler (`R-WRITE-1`, `R-WRITE-2`). |
| Do not add a redundant figure. | No answer visual without explicit permission; required visuals must be minimal (`R-STRUCT-6`). |
| Keep only the block-type matrix visual in Q2(c). | That exact approved diagram is the only currently allowed non-prompt media (`R-STRUCT-6` and the media hash contract). |
| Use Word Heading 1/Heading 2 styles, but match HW1's appearance. | Use semantic styles with the exact resolved DejaVu Sans, size, weight, color, spacing, and outline levels (`R-STRUCT-3`, `R-WORD-3`, `R-WORD-4`). |
| The Hebrew still looked LTR. | Preserve Word-native RTL properties and verify the physical right edge after save/reopen in a full-page Word render (`R-WORD-5`, `R-FLOW-4`). |
| The font changed. | Hebrew body and headings use the exact DejaVu Sans profile; generic Aptos, Calibri, and Times New Roman overrides are forbidden (`R-WORD-2` through `R-WORD-4A`). |
| The retyped Q3 prompt broke the established structure. | Prompt wording comes from a centered high-resolution `HW4.pdf` excerpt, never the rejected `Original Question` style (`R-STRUCT-4`). |
| Old versions should not remain in the main folder. | The main folder contains only `HW4-Solution.docx`; superseded checkpoints go under `old/` (`R-STRUCT-8`). |
| Semantic approval was mistaken for format approval. | Track reasoning, formal text, insertion, audit, rendered visual approval, and publication independently (`R-FLOW-5`). |

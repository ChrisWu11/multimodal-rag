# Final Report: Remaining Human Actions

This checklist contains only decisions that require Kang Wu's authorship,
institutional authority or independent professional judgement. Automated and
Codex-supported work is recorded separately. The report describes the author's
manual verification first and subsequent Codex cross-checking, and does not
represent either as clinical-expert sign-off.

## Required before submission

### 1. Module generative-AI permission: author-confirmed complete

- On 16 August 2026, Kang Wu confirmed that the module permits the declared
  level of generative-AI assistance used for this project.
- The report retains an explicit statement in `Acknowledgements` describing
  the roles of Codex and Google Gemini and the author's responsibility.
- Recheck only if the School or module publishes revised instructions before
  submission.

University guidance used for this check:

- <https://www.birmingham.ac.uk/libraries/education-excellence/gai/acknowledging-gai-by-students>
- <https://www.birmingham.ac.uk/libraries/education-excellence/gai/guidance-pgt-dissertations>

### 2. Accept the report as its author

- Read the complete PDF and confirm that it accurately describes your work.
- Confirm the name, student ID, supervisor, programme and project title.
- Confirm that every objective, limitation and conclusion reflects what you
  are prepared to defend in the viva or Q&A.
- Confirm that the acknowledgements and Codex/Gemini declaration are accurate.
- Make any personal wording changes; then regenerate the PDF.

### 3. Apply supervisor feedback

- Send the draft to Jiaqi Ye if feedback is still permitted.
- Resolve requested scientific or structural changes.
- Re-run report validation after any changed number, citation, table or figure.

### 4. Complete the official submission

- Risk Assessment Form: author-confirmed complete.
- Ethical Review Form: author-confirmed complete.
- Five Project Meeting Forms: author-confirmed complete.
- Confirm only whether the module requests any additional supplementary file.
- Upload the final PDF and only the requested supplementary files to Canvas.
- Submit before **12:00 noon on Friday 4 September 2026**.
- Reopen the submitted file and verify that it is the intended final PDF.

## Optional independent strengthening

- A medical or ultrasound-domain expert may independently review the 50 silver
  labels. Until then, keep the term `manually reviewed and Codex-cross-checked
  silver`, not `gold`.
- The blank claim-citation and PDF forms may be used for independent sign-off,
  but the report does not require them to describe the completed author-led,
  Codex-cross-checked technical audits.
- Obtain figure-specific permission only if a future report version reproduces
  any source-paper image. The current report and public package reproduce none.
- Clinical validation would require a separate protocol, qualified experts,
  appropriate data governance and ethics approval; it is outside this report.

## Work already completed without human sign-off claims

- 50 silver question-evidence labels checked against stored primary sources:
  32 approved and 18 corrected.
- 30 sampled PDFs, covering 524 rendered pages, technically audited for page
  coverage, reading order, sections, tables, equations and reference boundaries.
- 17 literature claim groups checked against primary sources; 12 report
  statements were narrowed or re-cited.
- All 25 bibliography entries checked for DOI, title, year and ordered authors.
- All 12 image records checked at article and source-page level; every source
  image is excluded from the report and public reproducibility package.
- Retrieval, reranking, chunk, metadata, generation, citation and image-QA
  experiments completed with confidence intervals and paired tests.
- Privacy-safe result packaging, software tests, LaTeX checks, compilation and
  PDF visual inspection are automated and should be rerun after final edits.

## Current verified baseline (29 August 2026)

- Evaluated software: commit `e416772`, tag `image-qa-v1-2026-06-23`.
- Report length: 4,604 prose words excluding floats; 4,986 words including
  tables and captions under the project's conservative checker.
- Content: 19 PDF pages, 6 figures, 5 tables, 25 cited references and 5 keywords.
- References: 25/25 verified for DOI or official record, title, year and
  ordered authors; all bibliography entries are cited. The Nougat entry now
  uses its formal ICLR 2024 proceedings record rather than the arXiv preprint.
- Result consistency: 18 headline corpus, retrieval, chunking, generation,
  image-QA and PDF-audit claims match the saved experiment outputs.
- Reproducibility package: refreshed on 20 August 2026 with the latest
  reference, claim-citation and result-claim audits; private source text,
  images, embeddings, model payloads, local paths and API keys remain excluded.
- Software: 27 tests passed and Ruff reported no errors in changed audit code.
- LaTeX: zero compilation errors and zero unresolved citations/references. The
  single compiler warning is the non-content `microtype` message
  `Command \showhyphens has changed.`
- PDF: all 19 pages were rendered and visually checked; Equation (1) is
  numbered, the objective table remains beneath Section 5.5, and no clipping,
  overlap, broken table, blank page or missing graphic remains.

# Final Report Evaluation Results

This directory is the privacy-safe evidence package for the *Development and
Evaluation of an Evidence-Grounded Multimodal RAG System for Scientific
Literature* final report.

## Scope

- Evaluated software: commit `e416772`, tag `image-qa-v1-2026-06-23`.
- Retrieval labels: 50 manually reviewed and Codex-cross-checked silver
  question-source pairs.
- Generation set: 30 answerable and 10 designed evidence-insufficient cases.
- Generation judging: 240 structured judgements from the fixed
  `gemini-3.6-flash` model, comprising two temperature-zero repeats for each
  answer and condition.
- Image set: 12 scientific figures from seven articles whose registry records
  specify CC BY 4.0.
- Statistical analysis: 10,000 bootstrap resamples and 50,000 paired
  randomisation sign swaps with fixed seeds.

The package contains aggregate summaries, question/rank-level retrieval
details, confidence intervals, paired tests, automated error-screening
categories, reference verification, article-level image-licence checks,
author-led source-page rights checking with subsequent Codex cross-checking, a
conservative public-release disposition and a 17-group claim-citation review.

## Privacy Boundary

The package intentionally excludes source PDFs, extracted evidence passages,
source figures, embeddings, SQLite databases, generated answers, raw model
requests, local filesystem paths and credentials. It is therefore suitable for
code review, but it cannot reproduce experiments without the separately
controlled corpus.

## Reproduction

The experiment entry point is `scripts/final_report_eval.py`. Statistical
post-processing is performed by `scripts/analyse_final_report_results.py`, and
this directory is rebuilt by `scripts/package_final_report_results.py`.
Parameters, hashes and model identifiers are retained in the immutable local
run directories; those directories are not part of this safe package.

## Human Review Boundary

The author first checked the 50 silver labels against stored primary-source
passages and online publication records, then used Codex to cross-check
consistency; 18 labels were corrected. The author-led 30-document PDF audit
inspected 524 rendered pages, and the report's 17 literature claim groups were
checked against primary sources before Codex cross-checking. These checks are
traceable, but the author is not a medical specialist and they do not
constitute independent medical-expert validation.

No source-paper image is redistributed in this package or reproduced in the
report. The conservative disposition excludes the PLOS prior-study composite
and BioRender-labelled schematic from public reproduction unless their
specific rights are independently confirmed. The blank forms in
`evaluation/manual-review/` remain available for optional independent review;
they are not represented as completed human sign-off.

# Human Review Pack

These four CSV files isolate the checks that require a person to read the
source material. Automated experiments and metadata checks must not be used to
pre-fill these judgement columns.

`03-image-rights-ai-precheck.csv` is a separate browser-assisted pre-check. It
records publisher licences, captions, credit lines and reuse cautions without
claiming independent human approval. It does not replace or pre-fill
`03-image-rights-review.csv`.

`03-image-rights-ai-disposition.csv` applies a conservative release rule: none
of the 12 source images is reproduced in the report or public package. The PLOS
prior-study photographs and BioRender-labelled schematic are specifically
excluded from public reuse unless separate confirmation is obtained. Aggregate
evaluation results remain publishable without distributing the source panels.

`01-silver-question-ai-review.csv` records an author-led technical evidence
review with subsequent Codex cross-checking. The author first checked each of the
50 labels against its local primary-source chunk and online bibliographic
record, then used Codex to cross-check consistency. The review approved 32 labels
and proposed corrections to 18; no label was silently deleted. Crossref title
and DOI records were verified for all 46 DOI-bearing labels, while the four
records without DOI were checked against their public arXiv landing pages. The
revised manifest is stored beside the private evaluation data as
`eval_silver_ai_reviewed_v2.jsonl`; it does not replace `eval_gold.jsonl` and
must still be described as reviewed silver data, not expert clinical gold.

The 30-document PDF sample has also received an author-led technical audit with
automated flags and subsequent Codex cross-checking. All 524 physical pages were
inspected in rendered contact sheets, with flagged pages re-rendered at high
resolution. The audit found 5 approved
documents, 21 with minor or local issues, and 4 requiring a parser rebuild.
Its outputs are under
`outputs/final-report-evaluation-reviewed-silver/20260810T123000Z-pdf-technical-audit/`.
It did not modify or pre-fill `02-pdf-technical-review.csv`, and it is not an
independent human or clinical review.

All 17 cited claim groups in the final LaTeX report were checked against
primary sources by the author and then cross-checked with Codex. Twelve groups were
narrowed or re-cited and all 17 final formulations were assessed as supported.
The audit is recorded in
`04-claim-citation-ai-review.csv`; it refreshes but does not pre-fill the blank
independent form `04-claim-citation-support-review.csv`.

1. If independent expert validation becomes available, review the 50 silver
   questions using `01-silver-question-ai-review.csv` as an auditable proposal,
   then record the independent decision in `01-silver-question-review.csv`.
2. If independent confirmation is required, use the completed author-led PDF
   audit as a defect list and record a separate decision in
   `02-pdf-technical-review.csv`; repeating the full 524-page screen is no
   longer required for the project team's technical audit.
3. No image-rights decision is needed to release the current report package,
   because all source panels are excluded. Independent permission or review is
   required only before reproducing a source image, especially PLOS Figure 7 or
   the BioRender-labelled Research Square schematic.
4. The author should read or independently confirm the cited sources before
   signing `04-claim-citation-support-review.csv`; the recorded Codex cross-check
   is an auditable secondary check, not medical-expert or supervisor sign-off.

Keep the completed files with the final evaluation archive. Update the report
only after recording the decisions; do not silently edit the original silver
manifest.

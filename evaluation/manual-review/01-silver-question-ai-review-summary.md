# Author-Led, Codex-Cross-Checked Review of 50 Silver Question-Evidence Labels

## Status and scope

This review checks whether each question is clear, whether the nominated paper
passage contains the answer, whether the answer stays within the source scope,
and whether the cited paper identity is bibliographically verifiable. The
author first checked primary-source passages and online publication records,
then used Codex to cross-check consistency. Because the author is not a medical
specialist, this process cannot replace independent clinical or medical-expert
review. The resulting dataset must therefore be described as a **manually
reviewed and Codex-cross-checked silver set**, not a gold standard.

The original manifest and the blank human-review worksheet were not modified.

## Method

Each of the 50 records was checked against the locally stored primary-source
chunk and its surrounding document metadata. Where a label was weak, nearby
chunks from the same paper were inspected to identify a passage that directly
supported a narrower and answerable question. The review checked:

1. question clarity and study-specific scope;
2. direct support in the nominated evidence passage;
3. answer relevance and completeness;
4. avoidance of unsupported clinical generalisation;
5. consistency of document, chunk, title, DOI and publication year; and
6. public bibliographic identity using Crossref or the source repository.

Crossref verified all 46 DOI-bearing records and their registered titles. The
four records without a DOI were checked against their public arXiv records.
Bibliographic verification confirms paper identity; it does not independently
replicate the scientific results.

## Decisions

| Decision | Count | Meaning |
|---|---:|---|
| Approved | 32 | The question, answer and evidence passage were sufficiently aligned. |
| Corrected | 18 | The question, answer or evidence chunk was revised to match the paper. |
| Excluded | 0 | No record was unusable after an evidence-preserving correction. |

Confidence was high for 47 labels and medium for three labels (`q005`, `q016`
and `q029`). These three remain suitable for sensitivity analysis, but their
scope should receive priority if an independent domain expert becomes
available.

## Corrected records

| ID | Main issue corrected |
|---|---|
| q005 | Replaced an unsupported heat-mechanism question with the thermal and mechanical effects stated by the review. |
| q007 | Replaced an objective-only passage with the study outcomes. |
| q011 | Restored both the reported burn mechanisms and mitigation context. |
| q013 | Reframed the answer around skull attenuation, distortion and individual variation. |
| q016 | Narrowed a general phantom question to the agar/MR-thermometry experiment. |
| q019 | Corrected thermal ablation to the mild-hyperthermia regime actually studied. |
| q020 | Replaced an unsupported target-temperature comparison with the reported property-sensitivity result. |
| q026 | Replaced a reference-list chunk and removed an unsupported ultrasound-versus-MRI comparison. |
| q029 | Narrowed the claim to the specific computational breast-tumour study and restored the complete result. |
| q031 | Narrowed the question to MRI-guided HIFU of the liver and kidney and used the conclusion passage. |
| q034 | Replaced an unrelated prognostic-marker chunk with the abdominal-cancer evidence passage. |
| q041 | Limited the question to the human-cornea simulation and named the heat-balance terms. |
| q042 | Replaced a search-protocol answer with the review's reported effectiveness evidence. |
| q045 | Corrected Doppler monitoring to the B-mode thermal-strain method actually evaluated. |
| q046 | Replaced the modelling objective with the reported simulation result and limitations. |
| q047 | Replaced an objective-only answer with the treatment-planning components. |
| q048 | Made the single-case evidence level explicit and included the reported result. |
| q050 | Replaced the study purpose with the preclinical feasibility result. |

Detailed per-label rationales, public verification URLs and before/after fields
are retained in `01-silver-question-ai-review.csv`.

## Effect on evaluation

The reviewed labels were used in a fresh 50-question retrieval evaluation. On
the production chunk index, hybrid retrieval achieved MRR 0.5230, Recall@5
0.62 and nDCG@10 0.5632. Hybrid retrieval significantly outperformed dense
retrieval for MRR (paired randomisation `p=0.000060`) and nDCG@10
(`p=0.000140`). Lexical candidates followed by CrossEncoder reranking improved
MRR over lexical retrieval from 0.4886 to 0.6052 (`p=0.031959`).

The chunk ablation found that 350-word chunks with 50-word overlap improved
MRR from 0.5230 to 0.5736, Recall@5 from 0.62 to 0.74, and nDCG@10 by 0.0545
relative to the 650/90 production configuration. These paired differences were
statistically significant at the 0.05 level. They are evidence for a future
configuration comparison, not permission to change the frozen evaluated model
mid-experiment.

## Remaining human boundary

No non-expert web search can turn this set into clinical gold labels. A domain
expert is only required if the report intends to claim expert validation,
clinical correctness, or a medically authoritative benchmark. Without that
claim, the auditable reviewed-silver set is appropriate provided the report
states the reviewer type, correction process, confidence levels and threats to
validity.

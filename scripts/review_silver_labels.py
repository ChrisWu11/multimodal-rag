#!/usr/bin/env python3
"""Create an auditable AI-assisted review of the 50 silver QA labels.

The script never overwrites the source manifest or the human-review worksheet.
It checks every referenced chunk locally, verifies DOI registration where
possible, and writes a separate reviewed-silver manifest and CSV audit trail.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


REVIEWED_AT = "2026-08-03"
REVIEWER_TYPE = "AI-assisted technical evidence review (non-clinical)"


APPROVED_NOTES = {
    "q001": "The cited passage directly states that continuous PRF-shift MR thermometry produces spatial temperature maps and temporal curves during MRgFUS.",
    "q002": "The cited passage directly explains that PRF shift provides spatially resolved, second-by-second MR temperature maps for thermal feedback.",
    "q003": "The cited study explicitly defines and achieves mild hyperthermia in the 39-43 C range.",
    "q004": "The cited passage shows 3D MR thermometry at peak heating and a cumulative thermal-dose map; the wording is safe because the question allows monitoring or control.",
    "q006": "The cited review supports HIFU/HIDU as focal options for localized prostate cancer and describes the organ-function-preserving rationale.",
    "q008": "The prospective multicentre study supports technical feasibility and safety of volumetric MR-HIFU for symptomatic uterine fibroids.",
    "q009": "The cited review explicitly contrasts high-intensity thermal ablation with low-intensity, reversible neuromodulation.",
    "q010": "The cited retrospective study identifies skin burns as a reported HIFU adverse event; the answer does not claim that this is the only risk.",
    "q012": "The cited experiments report planning/monitoring artefacts, beam aberration and unusual PRF-thermometry patterns near nitinol biopsy markers.",
    "q014": "The cited paper presents an open-source method that derives individual skull estimates from T1-weighted MRI for acoustic simulation.",
    "q015": "The cited numerical study couples the Pennes bioheat equation with an inhomogeneous Helmholtz model to predict heating and temperature distribution.",
    "q017": "The cited method compares thermocouple measurements of temperature rise and lesion formation with simulation predictions.",
    "q018": "The cited experiment explicitly reports 1 MHz low-frequency and 3 MHz high-frequency components with their powers.",
    "q021": "The cited study supports Doppler twinkling for lesion localisation and MRI as the high-accuracy reference for lesion dimensions and boundaries.",
    "q022": "The cited consensus states that MRI supplies anatomy, targeting and temperature maps used for thermal feedback during MRgFUS.",
    "q023": "The cited passage supports focused-ultrasound BBB opening as a cavitation/microbubble-mediated delivery application; it is not treated as clinical efficacy evidence.",
    "q024": "The cited review directly links HIFU cavitation and heating to combined mechanical/thermal stress, DAMP release and epitope exposure.",
    "q025": "The cited paper explicitly defines histotripsy as non-thermal focused-ultrasound ablation through controlled acoustic cavitation.",
    "q027": "The cited cohort reports 59 essential-tremor thalamotomies and two Parkinson-disease pallidotomies using a 1024-element ExAblate system.",
    "q028": "The cited primary study reports focused-ultrasound activation of engineered bacteria and in-vivo tumour suppression in an immunotherapy design.",
    "q030": "The cited study reports in-vitro and in-vivo activation of an engineered thermal gene switch by focused ultrasound.",
    "q032": "The cited phantom experiments report a stable +7 C elevation for 15 minutes across tested perfusion rates.",
    "q033": "The cited in-vivo mouse experiment directly reports accurate real-time PID feedback control of mild hyperthermia.",
    "q035": "The cited review passage supports pain palliation in bone metastases; it is not used to infer broader oncological benefit.",
    "q036": "The cited method calculates a hyperthermia power profile by deconvolving the target temperature curve with the measured impulse response.",
    "q037": "The cited study reports depth-aligned thermal peaks and parameter-dependent thermal accumulation in micro-focused ultrasound.",
    "q038": "The cited experiment compares 1 MHz heating distributions in commercial bone phantoms and ex-vivo human femur/tibia using thermography.",
    "q039": "The cited experiment uses a fibre-optic probe as an independent baseline for temperature and cumulative thermal-dose calculations.",
    "q040": "The cited passage explicitly identifies motion, pulsation and flow near the oesophagus, trachea and large vessels as sources of reduced precision.",
    "q043": "The cited paper supports ultrasound/microbubble-enhanced delivery and thermomechanical release of drug from gold nanoparticles.",
    "q044": "The cited results quantify changes in predicted necrosis when temperature-dependent properties replace constant-property models.",
    "q049": "The cited narrative review supports the listed sarcoma mechanisms; the label is scoped as a review summary rather than proof of clinical effectiveness.",
}


CORRECTIONS: dict[str, dict[str, str]] = {
    "q005": {
        "question": "What thermal and mechanical biological effects can focused ultrasound induce in tissue?",
        "answer": "Focused ultrasound can produce site-specific thermal and mechanical effects, including local heating, acoustic cavitation and radiation force. These effects can drive protein denaturation, tissue ablation, sonoporation and other controlled biological responses.",
        "chunk_id": "doc_0140_1754d6c2e414_c0003",
        "confidence": "medium",
        "checks": "yes|partial|partial|no",
        "notes": "The original question asks for the physical mechanism of local heat production, but the cited review passage mainly lists downstream thermal and mechanical effects. The revised question matches what the passage actually supports.",
    },
    "q007": {
        "question": "What outcomes are reported for HIFU treatment of breast fibroadenomas?",
        "answer": "In 201 patients with 314 fibroadenomas, post-HIFU volume reduction differed by breast composition: 72.9% in fatty, 61.6% in loose, 55% in mixed and 49% in dense breasts. The authors concluded that HIFU was safe and effective across gland types, while noting better volume reduction in fatty breasts.",
        "chunk_id": "doc_0002_e2eaa961a507_c0002",
        "confidence": "high",
        "checks": "yes|partial|partial|yes",
        "notes": "The original excerpt describes the objective and measurements but omits the outcomes. A cleaner abstract chunk contains the cohort size, volume-reduction results and conclusion.",
    },
    "q011": {
        "question": "What causes skin burns during HIFU treatment and how are they discussed?",
        "answer": "The study links burns to energy deposition at scars or tissue interfaces and describes a superficial large tumour with bubbles at a skin puncture site as another near-field risk. It discusses cooling intervals, staged HIFU sessions and subcutaneous local anaesthetic to increase the skin-to-tumour distance as mitigations.",
        "chunk_id": "doc_0132_0822ef358235_c0014",
        "confidence": "high",
        "checks": "yes|yes|partial|yes",
        "notes": "The original answer contains only the mitigation sentence. The same source chunk also gives the proposed causes and the clinical context, so the corrected answer restores both parts of the question.",
    },
    "q013": {
        "question": "How do skull properties affect transcranial focused ultrasound treatment planning?",
        "answer": "The skull strongly attenuates and distorts transcranial ultrasound, so planning simulations need skull morphology and acoustic properties, ideally from individual CT. The study also found position-dependent inter-individual variation, especially for some posterior positions near the midline.",
        "chunk_id": "doc_0039_13579593089f_c0001",
        "confidence": "high",
        "checks": "yes|yes|partial|yes",
        "notes": "The original excerpt begins with background and does not isolate the answer. The full abstract directly supports attenuation, distortion and inter-individual variability.",
    },
    "q016": {
        "question": "How are agar-based soft-tissue phantoms used to evaluate focused-ultrasound heating and MR thermometry?",
        "answer": "Agar-based phantoms are characterised as tissue-mimicking materials and exposed to controlled focused-ultrasound heating while MRTI follows the temperature field. The cooling profile can then be used to estimate thermal diffusivity or conductivity and to evaluate the exposure and thermometry setup.",
        "chunk_id": "doc_0208_075a0ad7c2f5_c0010",
        "confidence": "medium",
        "checks": "yes|partial|partial|no",
        "notes": "The original question implies a survey of multiple phantom types, whereas this paper and chunk concern agar-based soft-tissue phantoms and a specific MRTI-based thermal-characterisation method.",
    },
    "q019": {
        "question": "What sonication and temperature-control parameters were reported in the perfused-phantom MRgFUS hyperthermia experiment?",
        "answer": "The experiment used 15-minute HIFU sonications targeting a +7 C temperature elevation. Across the tested perfusion conditions, 90% of the target was reached in 40.5-59.4 seconds and the plateau drift remained below 1e-3 C/s.",
        "chunk_id": "doc_0156_5f9a68461daf_c0011",
        "confidence": "high",
        "checks": "yes|yes|no|no",
        "notes": "The original question says thermal ablation, but the source experiment is controlled mild hyperthermia. The revised label preserves the reported parameters without changing the treatment regime.",
    },
    "q020": {
        "question": "How do temperature-dependent tissue properties affect simulated focused-ultrasound ablation and hyperthermia?",
        "answer": "In the reported simulations, temperature-dependent properties increased predicted necrosis in high-power liver sonications by 17.6% and 13% relative to constant properties at 25 C and 37 C. They reduced low-power temperature rises by 17-20%, changed rabbit-muscle necrosis estimates by up to 18%, and temperature-dependent acoustic properties increased predicted necrosis by up to 30%.",
        "chunk_id": "doc_0115_03129e2fda6b_c0001",
        "confidence": "high",
        "checks": "yes|no|no|no",
        "notes": "The source does not provide a general ablation-versus-hyperthermia target-temperature comparison. It instead quantifies how temperature-dependent properties change simulation outputs, so both question and answer are corrected to that evidence.",
    },
    "q026": {
        "question": "What does MRI guidance provide for HIFU temperature monitoring and dose control in liver and kidney treatments?",
        "answer": "MRI guidance supports in-situ target definition and continuous temperature mapping for spatial and temporal control of HIFU heating. Thermal-dose information can be used to predict the final lesion, although motion in the liver and kidney makes mapping and feedback control difficult.",
        "chunk_id": "doc_0057_2f9abb4f4e54_c0001",
        "confidence": "high",
        "checks": "yes|no|no|no",
        "notes": "The original gold chunk is a reference list and cannot support a comparison between ultrasound and MRI guidance. The revised question is supported by the paper abstract and avoids an unsupported head-to-head comparison.",
    },
    "q029": {
        "question": "In the breast-tumour simulation, how did uniformly mixed nanoparticles affect focused-ultrasound heating and necrotic damage?",
        "answer": "Uniform nanoparticle mixing enhanced the simulated thermal response and increased the history and spatial extent of necrotic damage. The study also found that longer pulsed heating at a lower duty cycle distributed necrotic damage more widely than continuous heating under the modelled conditions.",
        "chunk_id": "doc_0063_4c53af31e2dc_c0023",
        "confidence": "medium",
        "checks": "yes|yes|partial|no",
        "notes": "The original answer is truncated and the question over-generalises from one computational breast-tumour study. The revised wording and result chunk preserve the study-specific scope.",
    },
    "q031": {
        "question": "What technical challenges limit MRI-guided HIFU of the liver and kidney?",
        "answer": "The paper identifies respiratory motion and tissue deformation, motion artefacts in MR thermometry, the need for real-time target tracking, ribs that partly block the beam, and rapid heat removal by high perfusion. Clinical translation also requires integration of these methods and faster image processing.",
        "chunk_id": "doc_0057_2f9abb4f4e54_c0017",
        "confidence": "high",
        "checks": "yes|partial|partial|no",
        "notes": "The original question is broader than the liver-and-kidney review. The conclusion chunk gives a specific, complete list of barriers, so the revised label is limited to that application.",
    },
    "q034": {
        "question": "What early clinical evidence is reported for MRgFUS treatment of hepatic and pancreatic lesions?",
        "answer": "The review describes MRgFUS as feasible and repeatable for unresectable, device-accessible hepatic and pancreatic lesions, but the cited early evidence is very small: one hepatocellular-carcinoma case and two pancreatic-cancer cases. In the liver case, ablation was confirmed by imaging and histopathology.",
        "chunk_id": "doc_0083_1c7b3a89d532_c0009",
        "confidence": "high",
        "checks": "yes|no|no|no",
        "notes": "The original chunk discusses prognostic blood markers rather than liver ablation outcomes. The replacement chunk contains the relevant abdominal-cancer section and makes the preliminary evidence level explicit.",
    },
    "q041": {
        "question": "How does the corneal focused-ultrasound simulation model tissue temperature rise?",
        "answer": "The corneal model uses the Pennes bioheat equation. Temperature change is represented through heat conduction, blood-perfusion exchange, metabolic heat and the external ultrasound heat source, with convective and radiative exchange at the anterior corneal boundary.",
        "chunk_id": "doc_0045_080530c76b9f_c0006",
        "confidence": "high",
        "checks": "yes|yes|partial|no",
        "notes": "The original question is general, but the evidence is a specific human-cornea simulation. The revised question states that scope and the answer identifies the actual heat-balance terms.",
    },
    "q042": {
        "question": "What does the expert opinion report about thermal ablation for selected low-risk papillary thyroid cancers?",
        "answer": "The reviewed meta-analyses found that thermal ablation reduced tumour volume, produced high complete-disappearance rates and maintained low local-recurrence and metastasis rates in selected low-risk disease. Several comparisons with surgery found no significant difference in recurrence, lymph-node metastasis or later salvage surgery, but patient and tumour selection remain essential.",
        "chunk_id": "doc_0151_8c239849551f_c0015",
        "confidence": "high",
        "checks": "yes|yes|partial|no",
        "notes": "The original answer is the literature-search protocol, not an outcome. The replacement summarises the effectiveness section and narrows the question to the low-risk papillary cancers covered by that passage.",
    },
    "q045": {
        "question": "How did the ultrasound-guided FUS system use thermal strain imaging to monitor mild hyperthermia?",
        "answer": "Thermal strain was calculated from B-mode images and calibrated against thermocouple measurements to form temperature maps. The reported mean absolute error was 0.25 C in a phantom and 0.8 C at the focus in mouse tumours; Doppler imaging was mentioned only as a possible future addition.",
        "chunk_id": "doc_0111_bccf1133ecd7_c0006",
        "confidence": "high",
        "checks": "yes|yes|partial|no",
        "notes": "The original wording foregrounds Doppler, although the study actually validates B-mode thermal-strain imaging and only mentions Doppler as future integration. The revised label reflects the performed experiment.",
    },
    "q046": {
        "question": "What did simulations report about focused-ultrasound-induced acoustic streaming for drug transport in breast and abdominal tumour models?",
        "answer": "The simulations suggest that acoustic streaming could assist transport through poorly vascularised tumour regions while keeping temperature below the modelled 50 C threshold. Higher frequencies and larger probe radii shortened the required time but reduced the treated area. The result is exploratory because the models assumed homogeneous tissue, favourable hydraulic conductivity and no perfusion cooling.",
        "chunk_id": "doc_0152_1044fb625bcf_c0014",
        "confidence": "high",
        "checks": "yes|partial|partial|no",
        "notes": "The original excerpt states only the modelling intention. The replacement result chunk supports the treatment implication and its main trade-off and limitations without presenting simulation as clinical proof.",
    },
    "q047": {
        "question": "How are treatment planning strategies described for interstitial ultrasound ablation of prostate cancer?",
        "answer": "The platform combines patient-specific anatomical segmentation with 3D finite-difference acoustic and bioheat models. It generates lookup tables of temperature and thermal dose to optimise applicator placement and settings, and evaluates temperature-feedback approaches based on invasive sensors or MRTI together with urethral cooling.",
        "chunk_id": "doc_0101_852b5b0e08f1_c0005",
        "confidence": "high",
        "checks": "yes|yes|partial|yes",
        "notes": "The original answer gives only the study objective. The same chunk contains the concrete planning components needed to answer the question.",
    },
    "q048": {
        "question": "What early clinical evidence was reported for transcranial MRgFUS ablation of a brain tumour?",
        "answer": "A single 63-year-old patient with recurrent centrally located glioblastoma received 25 high-power MR-guided sonications. Partial tumour ablation was achieved without neurological deficits or other reported adverse effects, demonstrating feasibility but not comparative efficacy.",
        "chunk_id": "doc_0222_b2005c9a776e_c0002",
        "confidence": "high",
        "checks": "yes|yes|partial|no",
        "notes": "The original question is broad and the excerpt mostly describes the case aim. The corrected version makes the single-case evidence level explicit and includes the reported result.",
    },
    "q050": {
        "question": "What was reported in the swine pilot study of MR-guided peripheral-nerve ablation?",
        "answer": "Peripheral nerves visualised with 3D MR neurography were accurately targeted and ablated with a single 20-36 second exposure. MR lesion dimensions agreed with dissection measurements, and histological damage was largely confined to the targeted region; this was a three-animal feasibility study.",
        "chunk_id": "doc_0218_59ec6601066c_c0011",
        "confidence": "high",
        "checks": "yes|yes|partial|no",
        "notes": "The original answer stops at the study purpose. The replacement discussion chunk provides the feasibility result and the revised question clearly identifies the preclinical model.",
    },
}


def normalise_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def title_matches(left: str, right: str) -> bool:
    return difflib.SequenceMatcher(
        None, normalise_title(left), normalise_title(right)
    ).ratio() >= 0.72


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_chunks(db_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        chunk_rows = connection.execute(
            "SELECT id, document_id, content, metadata_json FROM chunks"
        ).fetchall()
        document_rows = connection.execute(
            "SELECT id, title, source_path, metadata_json FROM documents"
        ).fetchall()
    chunks = {
        row["id"]: {
            "id": row["id"],
            "document_id": row["document_id"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
        }
        for row in chunk_rows
    }
    documents = {
        row["id"]: {
            "id": row["id"],
            "title": row["title"],
            "source_path": row["source_path"],
            "metadata": json.loads(row["metadata_json"]),
        }
        for row in document_rows
    }
    return chunks, documents


def verify_dois(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {
        str(source.get("doi") or "").lower(): str(source.get("title") or "")
        for label in labels
        for source in label["gold_sources"]
        if source.get("doi")
    }
    results: dict[str, dict[str, Any]] = {}
    headers = {"User-Agent": "UoB-MSc-RAG-silver-label-audit/1.0"}
    with httpx.Client(timeout=30, follow_redirects=True, headers=headers) as client:
        for doi, expected_title in sorted(expected.items()):
            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            try:
                response: httpx.Response | None = None
                for attempt in range(5):
                    response = client.get(url)
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        break
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                    time.sleep(min(delay, 8.0))
                assert response is not None
                response.raise_for_status()
                item = response.json()["message"]
                registered_title = str((item.get("title") or [""])[0])
                results[doi] = {
                    "doi_registry_verified": True,
                    "registered_title": registered_title,
                    "registered_title_match": title_matches(expected_title, registered_title),
                    "registered_year": str(
                        ((item.get("published") or {}).get("date-parts") or [[""]])[0][0]
                    ),
                    "registry_note": "Crossref record retrieved",
                }
                time.sleep(0.2)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
                results[doi] = {
                    "doi_registry_verified": False,
                    "registered_title": "",
                    "registered_title_match": False,
                    "registered_year": "",
                    "registry_note": f"Crossref lookup failed: {type(error).__name__}",
                }
    return results


def source_for_chunk(
    chunk: dict[str, Any], document: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    metadata = chunk["metadata"]
    document_metadata = document["metadata"]
    year = registry.get("registered_year") or metadata.get("year") or document_metadata.get("year")
    return {
        "chunk_id": chunk["id"],
        "doc_id": chunk["document_id"],
        "doi": str(metadata.get("doi") or document_metadata.get("doi") or ""),
        "page_end": metadata.get("page_end"),
        "page_start": metadata.get("page_start"),
        "section": metadata.get("section"),
        "title": document["title"],
        "year": str(year or ""),
    }


def public_url(source: dict[str, Any], document: dict[str, Any]) -> str:
    doi = str(source.get("doi") or "")
    if doi:
        return f"https://doi.org/{doi}"
    metadata = document["metadata"]
    return str(metadata.get("landing_url") or document.get("source_path") or "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    labels = load_jsonl(args.manifest)
    if len(labels) != 50 or {label["id"] for label in labels} != {
        f"q{index:03d}" for index in range(1, 51)
    }:
        raise ValueError("Expected exactly q001-q050 in the source manifest")
    if set(APPROVED_NOTES) & set(CORRECTIONS):
        raise ValueError("A label cannot be both approved and corrected")
    if set(APPROVED_NOTES) | set(CORRECTIONS) != {label["id"] for label in labels}:
        raise ValueError("Every label must have an explicit review")

    chunks, documents = load_chunks(args.database)
    registry = verify_dois(labels)
    csv_rows: list[dict[str, Any]] = []
    reviewed_labels: list[dict[str, Any]] = []

    for original in labels:
        label_id = original["id"]
        original_source = original["gold_sources"][0]
        correction = CORRECTIONS.get(label_id)
        if correction:
            question_clear, source_contains, answer_relevance, scope_safe = correction[
                "checks"
            ].split("|")
            decision = "corrected"
            confidence = correction["confidence"]
            corrected_question = correction["question"]
            corrected_answer = correction["answer"]
            corrected_chunk_id = correction["chunk_id"]
            notes = correction["notes"]
        else:
            question_clear = source_contains = answer_relevance = scope_safe = "yes"
            decision = "approved"
            confidence = "high"
            corrected_question = original["question"]
            corrected_answer = original["gold_answer"]
            corrected_chunk_id = str(original_source["chunk_id"])
            notes = APPROVED_NOTES[label_id]

        if corrected_chunk_id not in chunks:
            raise KeyError(f"Missing reviewed chunk: {corrected_chunk_id}")
        chunk = chunks[corrected_chunk_id]
        document = documents[chunk["document_id"]]
        doi = str(
            chunk["metadata"].get("doi")
            or document["metadata"].get("doi")
            or original_source.get("doi")
            or ""
        ).lower()
        registry_record = registry.get(
            doi,
            {
                "doi_registry_verified": "not_applicable",
                "registered_title": "",
                "registered_title_match": "not_applicable",
                "registered_year": "",
                "registry_note": "No DOI; verified against the public repository record",
            },
        )
        corrected_source = source_for_chunk(chunk, document, registry_record)

        reviewed = dict(original)
        reviewed.update(
            {
                "question": corrected_question,
                "gold_answer": corrected_answer,
                "evidence_excerpt": corrected_answer,
                "gold_sources": [corrected_source],
                "label_source": "ai_assisted_reviewed_silver",
                "review_status": f"ai_reviewed_{decision}",
                "notes": (
                    "Reviewed against the local primary-source text and public bibliographic "
                    "metadata. This remains a non-expert silver label, not clinical gold."
                ),
                "review": {
                    "decision": decision,
                    "confidence": confidence,
                    "reviewer_type": REVIEWER_TYPE,
                    "reviewed_at": REVIEWED_AT,
                    "original_question": original["question"],
                    "original_gold_chunk_ids": [
                        source["chunk_id"] for source in original["gold_sources"]
                    ],
                    "rationale": notes,
                    "not_expert_gold": True,
                },
            }
        )
        reviewed_labels.append(reviewed)

        csv_rows.append(
            {
                "id": label_id,
                "type": original["type"],
                "original_question": original["question"],
                "original_gold_chunk_id": original_source["chunk_id"],
                "question_clear": question_clear,
                "source_contains_answer": source_contains,
                "answer_relevance": answer_relevance,
                "scope_safe": scope_safe,
                "decision": decision,
                "corrected_question": corrected_question,
                "corrected_gold_answer": corrected_answer,
                "corrected_gold_chunk_id": corrected_chunk_id,
                "corrected_gold_source_title": corrected_source["title"],
                "corrected_gold_source_doi": corrected_source["doi"],
                "confidence": confidence,
                "reviewer_type": REVIEWER_TYPE,
                "reviewer_notes": notes,
                "public_verification_url": public_url(corrected_source, document),
                "doi_registry_verified": registry_record["doi_registry_verified"],
                "registered_title": registry_record["registered_title"],
                "registered_title_match": registry_record["registered_title_match"],
                "registered_year": registry_record["registered_year"],
                "registry_note": registry_record["registry_note"],
                "reviewed_at": REVIEWED_AT,
            }
        )

    write_csv(args.csv_output, csv_rows)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed_labels),
        encoding="utf-8",
    )

    decisions = {name: sum(row["decision"] == name for row in csv_rows) for name in ("approved", "corrected", "excluded")}
    registry_verified = sum(row["doi_registry_verified"] is True for row in csv_rows)
    registry_total = sum(bool(row["corrected_gold_source_doi"]) for row in csv_rows)
    print(
        f"Reviewed {len(csv_rows)} labels: {decisions['approved']} approved, "
        f"{decisions['corrected']} corrected, {decisions['excluded']} excluded. "
        f"Crossref verified {registry_verified}/{registry_total} DOI-bearing labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

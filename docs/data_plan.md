# Data Plan

Ask the team for these data contracts as early as possible.

## Text

- source title
- source type: paper, guideline, report, protocol, device manual, experiment note
- upload date
- author/team
- allowed use
- citation or internal reference ID

## Ultrasound

- image/video file
- DICOM metadata if available
- probe type
- anatomical region or object
- acquisition view
- gain/depth/focus settings
- report text
- labels/annotations
- de-identification status

## Thermal Imaging

- image/video file
- radiometric data availability
- camera model
- calibration state
- emissivity
- ambient temperature
- distance
- body region or object
- time series relationship if applicable
- labels/annotations

## Evaluation Set

Create a small golden set before improving the model:

- 30-50 questions
- expected evidence documents
- expected answer constraints
- unsafe answer examples
- modality-specific edge cases

The most important early metrics are retrieval recall, citation correctness, and unsupported-claim rate.

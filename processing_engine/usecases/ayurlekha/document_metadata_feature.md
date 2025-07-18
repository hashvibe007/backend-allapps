# Document Metadata Feature for Medical Records

## Overview
This feature enables intelligent, user-friendly metadata for each medical document, supporting both card and detail views in the UI. Metadata is generated using LLMs and entity extraction, and is stored as a JSON file alongside each document. The design leverages dspy.Signature and dspy.Module for consistency and modularity.

---

## User Stories / UX
- **As a user**, I want to see a short, meaningful name and key details for each document on the dashboard, so I can quickly find what I need.
- **As a user**, I want to see a summary and unique insights when I open a document, so I understand its importance without reading the whole file.
- **As a doctor**, I want to see document type, date, department, and doctor name at a glance.
- **As a patient**, I want to know if any action is required or if there are urgent findings.

---

## Metadata Schema (per-document)
```json
{
  "intelligent_name": "Liver Function Test, Mar 2023",
  "category": "Lab Report",
  "date": "2023-03-01",
  "department": "Hepatology",
  "doctor_name": "Dr. Nidhi Sota",
  "patient_name": "Avinash Kumar",
  "insights": [
    "Alkaline Phosphatase: High",
    "Gamma GT: High",
    "SGOT (AST): High",
    "SGPT (ALT): High",
    "Diagnosis: Post Living Donor Liver Transplant (LDLT)"
  ],
  "actions": [
    "Monitor liver function closely",
    "Consider further investigation"
  ],
  "urgency": "High",
  "summary": "Avinash Kumar, a post-liver transplant patient, has lab results indicating potential liver dysfunction. Elevated Alkaline Phosphatase, Gamma GT, SGOT (AST), and SGPT (ALT) levels were observed, particularly on March 1, 2023."
}
```

---

## Pipeline / Code Flow
1. **Document Analysis**: Each document is processed using a dspy.Module (e.g., `DocumentProcessor`).
2. **Metadata Extraction**: A new dspy.Signature (`DocumentMetadataSignature`) and dspy.Module (`DocumentMetadataModule`) are used to extract/generate metadata fields from the analysis.
3. **Metadata Storage**: The metadata dict is saved as `FILENAME_metadata.json` in the same directory as the document.
4. **(Optional) Patient/User Index**: An aggregated index can be maintained for fast dashboard rendering.

---

## LLM Prompt Example (for Metadata Module)
```
Given the following medical document analysis, extract the following metadata fields:
- intelligent_name: A short, human-friendly name for the document
- category: Document type (e.g., Prescription, Lab Report, Discharge Summary)
- date: Date of the document/event
- department: Medical department or specialty
- doctor_name: Name of the doctor
- patient_name: Name of the patient
- insights: List of unique, high-value findings or entities
- actions: List of recommended actions or follow-ups
- urgency: Urgency level (e.g., High, Medium, Low)
- summary: Short summary of the document

Analysis:
{detailed_analysis}
```

---

## UI Integration Plan
- **Card View**: UI reads `*_metadata.json` for each document to display name, category, date, department, doctor, insights, and urgency.
- **Detail View**: On click, UI shows full summary, insights, actions, and all metadata fields.
- **(Optional) Patient Index**: UI can use a patient-level index for fast loading.

---

## Future Extensions
- Add preview images/thumbnails for scanned docs.
- Add confidence scores for extracted fields.
- Add audit trail fields (upload/modification info).
- Add tags/icons for quick filtering.
- Support for multi-language summaries.

---

## Consistency with Current Design
- All metadata extraction is done via dspy.Signature and dspy.Module for modularity and maintainability.
- The pipeline is fully compatible with the existing document analysis and summary generation flow.
- Easy to extend or modify fields as requirements evolve. 
import dspy
from typing import List, Dict, Any
from processing_engine.common.web_tools import web_verify_medicine
from .signatures import DocumentProcessorSignature
from .signatures import AyurlekhaSummarySignature
from .signatures import DocumentMetadataSignature
from datetime import datetime, timezone


class MedicineFactChecker(dspy.Module):
    """
    Module to verify if extracted medicines are real or not and suggest correct names.
    """

    def __init__(self):
        super().__init__()
        self.tools = [web_verify_medicine]
        self.react = dspy.ReAct(
            signature="medicine_verification_query -> verification_result,correct_medicine,if_medicine",
            tools=self.tools,
            max_iters=2,
        )

    def verify_medicine(self, medicine_name: str) -> Dict[str, Any]:
        """
        Verify if a given medicine name is a real pharmaceutical drug or medication.
        """
        query = (
            f"Verify if '{medicine_name}' is a real pharmaceutical drug or medication"
        )
        try:
            result = self.react(medicine_verification_query=query)
            return {
                "medicine": medicine_name,
                "if_medicine": result.if_medicine,
                "verification_result": result.verification_result,
                "correct_medicine": result.correct_medicine,
                "status": "verified",
            }
        except Exception as e:
            return {
                "medicine": medicine_name,
                "if_medicine": "",
                "verification_result": f"Verification failed: {str(e)}",
                "correct_medicine": "",
                "status": "error",
            }

    def verify_multiple_medicines(self, medicines: List[str]) -> List[Dict[str, Any]]:
        """
        Verify a list of medicine names.
        """
        results = []
        for i, medicine in enumerate(medicines):
            if i > 0:
                import time

                time.sleep(2)
            result = self.verify_medicine(medicine)
            results.append(result)
        return results


class DocumentProcessor(dspy.Module):
    """
    Module to process a medical document image and extract structured information, including detailed analysis and medicine verification.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentProcessorSignature)
        self.medicine_checker = MedicineFactChecker()

    def forward(self, document_image):
        """
        Process the document image and return detailed analysis and verified medicines.
        """
        prediction = self.predictor(document_image=document_image)
        medicine_verifications = self.medicine_checker.verify_multiple_medicines(
            prediction.extracted_medicines
        )
        return dspy.Prediction(
            detailed_analysis=prediction.detailed_analysis,
            medicine_verifications=medicine_verifications,
            extracted_medicines=prediction.extracted_medicines,
        )


class DocumentMetadataModule(dspy.Module):
    """
    Module to extract/generate per-document metadata using LLM and entity extraction.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentMetadataSignature)

    def forward(self, detailed_analysis: str) -> dspy.Prediction:
        return self.predictor(detailed_analysis=detailed_analysis)


class PatientDemographics(dspy.Module):
    """
    Module to extract and synthesize all patient demographic and medical summary fields required for the Ayurlekha JSON template.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(AyurlekhaSummarySignature)

    def forward(
        self, medical_history: str, patient_id: str = None, user_id: str = None
    ) -> dspy.Prediction:
        """
        Extract all required fields for the Ayurlekha JSON summary from the medical history string.
        """
        prediction = self.predictor(medical_history=medical_history)
        # Fallback/placeholder logic for all required fields
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        generated_at = now.isoformat()
        # Patient block
        patient = (
            prediction.patient
            if hasattr(prediction, "patient")
            else {
                "id": patient_id or "",
                "name": getattr(prediction, "patient_name", "Unknown"),
                "dob": getattr(prediction, "dob", ""),
                "age": getattr(prediction, "age", ""),
                "bloodGroup": getattr(prediction, "bloodGroup", ""),
            }
        )
        # Primary alert
        primaryAlert = (
            prediction.primaryAlert
            if hasattr(prediction, "primaryAlert")
            else {
                "alert": getattr(prediction, "alert", ""),
                "specialCare": getattr(prediction, "specialCare", ""),
            }
        )
        # Footer
        footer = (
            prediction.footer
            if hasattr(prediction, "footer")
            else {
                "date": today_str,
                "generatedBy": "Ayurlekha App",
                "notMedicalDocument": "Not a Medical Document",
                "disclaimer": "This document is a summary for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.",
            }
        )
        # Meta
        meta = (
            prediction.meta
            if hasattr(prediction, "meta")
            else {
                "version": "1.0",
                "generated_at": generated_at,
                "patient_id": patient_id or "",
                "user_id": user_id or "",
            }
        )
        return dspy.Prediction(
            patient=patient,
            summary=getattr(prediction, "summary", ""),
            primaryAlert=primaryAlert,
            chronicConditions=getattr(prediction, "chronicConditions", []),
            historyTimeline=getattr(prediction, "historyTimeline", []),
            labTests=getattr(prediction, "labTests", []),
            medications=getattr(prediction, "medications", []),
            doctors=getattr(prediction, "doctors", []),
            emergencyContacts=getattr(prediction, "emergencyContacts", []),
            footer=footer,
            meta=meta,
        )

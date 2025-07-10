import dspy
from typing import List, Dict, Any
from processing_engine.common.web_tools import web_verify_medicine
from .signatures import DocumentProcessorSignature
from .signatures import PatientDemographicsSignature


class MedicineFactChecker(dspy.Module):
    """Module to verify if extracted medicines are real or not and suggest correct names."""

    def __init__(self):
        super().__init__()
        self.tools = [web_verify_medicine]
        self.react = dspy.ReAct(
            signature="medicine_verification_query -> verification_result,correct_medicine,if_medicine",
            tools=self.tools,
            max_iters=2,
        )

    def verify_medicine(self, medicine_name: str) -> Dict[str, Any]:
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
        results = []
        for i, medicine in enumerate(medicines):
            if i > 0:
                import time

                time.sleep(2)
            result = self.verify_medicine(medicine)
            results.append(result)
        return results


class DocumentProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentProcessorSignature)
        self.medicine_checker = MedicineFactChecker()

    def forward(self, document_image):
        prediction = self.predictor(document_image=document_image)
        medicine_verifications = self.medicine_checker.verify_multiple_medicines(
            prediction.extracted_medicines
        )
        return dspy.Prediction(
            detailed_analysis=prediction.detailed_analysis,
            medicine_verifications=medicine_verifications,
            extracted_medicines=prediction.extracted_medicines,
        )


class PatientDemographics(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(PatientDemographicsSignature)

    def forward(self, medical_history):
        prediction = self.predictor(medical_history=medical_history)
        return dspy.Prediction(
            patient_name=prediction.patient_name,
            age=prediction.age,
            gender=prediction.gender,
            illness_details=prediction.illness_details,
        )

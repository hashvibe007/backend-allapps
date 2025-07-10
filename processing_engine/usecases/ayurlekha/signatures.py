import dspy
from typing import List


class DocumentProcessorSignature(dspy.Signature):
    """
    Given an image of a medical document, your task is to extract structured information.
    Identify the type of document, key demographic details, a summary, and all specific medical entities.
    Also, provide a confidence score for the extraction and list any questions or uncertainties for user review.
    """

    document_image: dspy.Image = dspy.InputField(
        desc="An image of a single medical document."
    )
    detailed_analysis: str = dspy.OutputField(
        desc="The detailed analysis of the medical document. ONLY SEMANTICALLY CORRECT AND ACCURATE RESULT IS ACCEPTED AND WITH HIGH CONFIDENCE"
    )
    extracted_medicines: List[str] = dspy.OutputField(
        desc="The extracted medicines from the medical document. ONLY RETURN THE MEDICINES THAT ARE possible medicines. DO NOT RETURN ANYTHING ELSE."
    )


class IllnessDetails(dspy.Signature):
    period: str = dspy.OutputField(
        desc="The period of illness of the patient roughly like Jan 2024 to Feb 2024"
    )
    department: list = dspy.OutputField(
        desc="The department of the illness like Cardiology, Neurology, etc."
    )
    complaint: list = dspy.OutputField(
        desc="The complaint of the illness like chest pain, headache, etc."
    )
    diagnosis: list = dspy.OutputField(
        desc="The diagnosis of the illness like heart disease, stroke, etc."
    )
    treatment: list = dspy.OutputField(
        desc="The treatment of the illness like medication, surgery, etc."
    )
    medications: list = dspy.OutputField(
        desc="The medications with dosage of the illness like 10mg of Aspirin, 500mg of Paracetamol, etc."
    )
    tests: list = dspy.OutputField(
        desc="The tests done for the illness like ECG, X-ray, etc."
    )
    procedures: list = dspy.OutputField(
        desc="The procedures done for the illness like angioplasty, bypass surgery, etc."
    )
    system_notes: list = dspy.OutputField(
        desc="The system notes of the illness like patient is in stable condition, patient is in critical condition, etc."
    )


class PatientDemographicsSignature(dspy.Signature):
    """
    Given a list of medical documents extraction and analysis, your task is to establish the basic ground truth for the patient.
    """

    medical_history: str = dspy.InputField(desc="The medical history of the patient")
    patient_name: str = dspy.OutputField(desc="The name of the patient")
    age: int = dspy.OutputField(desc="The current age of the patient")
    gender: str = dspy.OutputField(desc="The gender of the patient")
    illness_details: list = dspy.OutputField(
        desc="List of illness events with period, department, complaint, diagnosis, treatment"
    )

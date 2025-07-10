import dspy
from typing import List, TypedDict, Optional, Dict, Any
import os
import json
from dotenv import load_dotenv
import glob
from supabase import create_client, Client
from datetime import datetime, timezone
import logging

# Load environment variables from a .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="analyse_medical_history.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

medgemma_lm = dspy.LM(
    "openai/medgemma-4b-it-Q3_K_M",
    api_base="http://127.0.0.1:8081/v1",
    api_key="sk1234",
)

# Judge LM: more powerful, for semantic evaluation
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please add it.")

gemini_lm = dspy.LM(
    "gemini/gemini-2.5-flash-preview-04-17",  # Using a standard Gemini model name
    api_key=gemini_api_key,
)

# Default configuration can be medgemma
dspy.configure(lm=gemini_lm)

# Ground truth establishment
# Multiple medical dcoument is given for the same patient to the model and it needs to establish the basic ground truth for exmaple
# Patient Name, Age, Gender - Patient Demographics
# symptoms - if given explicitly in the documents, then use it, otherwise use the symptoms from the diagnosis
# Diagnosis - Based on medical docs during the same time period
# Medications with dosage - Based on prescriptions diagnosis and treatment and suggestive medications in the documents


class IllnessDetails(dspy.Signature):
    period: str = dspy.OutputField(
        desc="The period of illness of the patient roughly like Jan 2024 to Feb 2024"
    )
    department: List[str] = dspy.OutputField(
        desc="The department of the illness like Cardiology, Neurology, etc."
    )
    complaint: List[str] = dspy.OutputField(
        desc="The complaint of the illness like chest pain, headache, etc."
    )
    diagnosis: List[str] = dspy.OutputField(
        desc="The diagnosis of the illness like heart disease, stroke, etc."
    )
    treatment: List[str] = dspy.OutputField(
        desc="The treatment of the illness like medication, surgery, etc."
    )
    medications: List[str] = dspy.OutputField(
        desc="The medications with dosage of the illness like 10mg of Aspirin, 500mg of Paracetamol, etc."
    )
    tests: List[str] = dspy.OutputField(
        desc="The tests done for the illness like ECG, X-ray, etc."
    )
    procedures: List[str] = dspy.OutputField(
        desc="The procedures done for the illness like angioplasty, bypass surgery, etc."
    )
    system_notes: List[str] = dspy.OutputField(
        desc="The system notes of the illness like patient is in stable condition, patient is in critical condition, etc."
    )


# After the IllnessDetails class definition, add a serialization helper
def serialize_illness_details(illness: IllnessDetails) -> dict:
    """Convert IllnessDetails to a dictionary."""
    return {
        "period": illness.period,
        "department": illness.department,
        "complaint": illness.complaint,
        "diagnosis": illness.diagnosis,
        "treatment": illness.treatment,
        "medications": illness.medications,
        "tests": illness.tests,
        "procedures": illness.procedures,
        "system_notes": illness.system_notes,
    }


class PatientDemographicsSignature(dspy.Signature):
    """
    Given a list of medical documents extraction and alalysis, your task is to establish the basic ground truth for the patient.
    """

    medical_history: str = dspy.InputField(desc="The medical history of the patient")
    patient_name: str = dspy.OutputField(desc="The name of the patient")
    age: int = dspy.OutputField(desc="The current age of the patient")
    gender: str = dspy.OutputField(desc="The gender of the patient")
    illness_details: List[IllnessDetails] = dspy.OutputField(
        desc="List of illness events with period, department, complaint, diagnosis, treatment"
    )


class DrugAnalyserSignature(dspy.Signature):
    """
    You are an expert medicine identifier, you are given noisy medicine name along with the context of the medicine and you need to identify the medicine name and the dosage.
    """

    context: str = dspy.InputField(
        desc="The context of the medicine along with noisy medicine name and dosage"
    )
    medicine_name: List[str] = dspy.OutputField(desc="The name of the medicines")


class DrugAnalyser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DrugAnalyserSignature)

    def forward(self, medicine_name, context):
        prediction = self.predictor(medicine_name=medicine_name, context=context)
        return dspy.Prediction(medicine_name=prediction.medicine_name)


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


patient_demographics_module = PatientDemographics()

drug_analyser_module = DrugAnalyser()

# load medical history
with open("final_patient_history.txt", "r") as f:
    medical_history = f.read()

# load medical documents

# print(medical_history)

analysis_result = patient_demographics_module(medical_history=medical_history)
print(analysis_result.patient_name)
print(analysis_result.age)
print(analysis_result.gender)
print(analysis_result.illness_details)

# analyse the medications
for illness in analysis_result.illness_details:
    for medication in illness.medications:
        drug_analysis_result = drug_analyser_module(
            medicine_name=medication, context=medical_history
        )
        print(f"Medication: {drug_analysis_result.medicine_name}")


# save the analysis result to a file
analysis_data = {
    "patient_name": analysis_result.patient_name,
    "age": analysis_result.age,
    "gender": analysis_result.gender,
    "illness_details": [
        serialize_illness_details(illness)
        for illness in analysis_result.illness_details
    ],
}

with open("analysis_result.json", "w") as f:
    json.dump(analysis_data, f, indent=2)
print(f"Analysis result saved to analysis_result.json")

# Also save a markdown summary for better readability
markdown_content = f"""# Patient Analysis Report

## Patient Demographics
- **Name:** {analysis_result.patient_name}
- **Age:** {analysis_result.age}
- **Gender:** {analysis_result.gender}

## Illness History
"""

for i, illness in enumerate(analysis_result.illness_details, 1):
    markdown_content += f"""
### Illness Event {i} ({illness.period})
- **Department:** {", ".join(illness.department)}
- **Complaints:** {", ".join(illness.complaint) if illness.complaint else "None recorded"}
- **Diagnosis:** {", ".join(illness.diagnosis) if illness.diagnosis else "None recorded"}
- **Treatment:** {", ".join(illness.treatment) if illness.treatment else "None recorded"}
- **Medications:** {", ".join(illness.medications) if illness.medications else "None prescribed"}
- **Tests:** {", ".join(illness.tests) if illness.tests else "None recorded"}
- **Procedures:** {", ".join(illness.procedures) if illness.procedures else "None recorded"}
- **System Notes:** {", ".join(illness.system_notes) if illness.system_notes else "None recorded"}
"""

with open("analysis_result.md", "w") as f:
    f.write(markdown_content)
print(f"Analysis summary saved to analysis_result.md")


def upload_file_to_supabase(bucket: str, remote_path: str, local_path: str):
    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(remote_path, f, upsert=True)


# --- Main Pipeline ---
def process_and_upload_ayurlekha():
    patients = supabase.table("patients").select("*").execute().data
    for patient in patients:
        patient_id = patient["id"]
        user_id = patient["user_id"]
        try:
            # Find all per-document analysis files for this patient
            analysis_files = glob.glob(f"temp_medical_docs/{patient_id}_*.txt")
            if not analysis_files:
                logger.info(f"No analysis files found for patient {patient_id}")
                continue
            logger.info(
                f"Found {len(analysis_files)} analysis files for patient {patient_id}"
            )
            combined_analysis = ""
            for afile in analysis_files:
                try:
                    with open(afile, "r") as f:
                        content = f.read()
                    combined_analysis += f"\n--- Analysis from {os.path.basename(afile)} ---\n{content}\n"
                except Exception as e:
                    logger.error(f"Failed to read {afile}: {e}")
            # Save combined analysis as Ayurlekha.md
            ayurlekha_path = f"temp_medical_docs/{patient_id}_Ayurlekha.md"
            with open(ayurlekha_path, "w") as f:
                f.write(combined_analysis)
            logger.info(f"Saved combined analysis to {ayurlekha_path}")
            # Upload to Supabase
            remote_ayurlekha_path = (
                f"medical-records/{user_id}/{patient_id}/Ayurlekha.md"
            )
            try:
                upload_file_to_supabase(
                    "medical-documents", remote_ayurlekha_path, ayurlekha_path
                )
                logger.info(
                    f"Uploaded Ayurlekha.md to {remote_ayurlekha_path} in Supabase Storage"
                )
                # Update ayurlekha_generated_at in patients
                now_str = datetime.now(timezone.utc).isoformat()
                supabase.table("patients").update(
                    {"ayurlekha_generated_at": now_str}
                ).eq("id", patient_id).execute()
                logger.info(f"Updated ayurlekha_generated_at for patient {patient_id}")
            except Exception as e:
                logger.error(
                    f"Failed to upload Ayurlekha.md for patient {patient_id}: {e}"
                )
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")


if __name__ == "__main__":
    logger.info(
        "Starting Ayurlekha.md aggregation and upload pipeline (analyse-medical-history.py)"
    )
    process_and_upload_ayurlekha()
    logger.info(
        "Completed Ayurlekha.md aggregation and upload pipeline (analyse-medical-history.py)"
    )

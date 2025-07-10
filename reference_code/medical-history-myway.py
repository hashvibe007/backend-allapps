import dspy
import asyncio
from typing import List, TypedDict, Optional, Dict, Any
from langchain_community.tools.pubmed.tool import PubmedQueryRun
import os
import json
import time
import random
from dotenv import load_dotenv
import glob
from ddgs import DDGS
from supabase import create_client, Client
from datetime import datetime, timezone
import logging

# Load environment variables from a .env file
load_dotenv()

medgemma_lm = dspy.LM(
    "openai/",
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
dspy.configure(lm=medgemma_lm)


# # Create async function for PubMed search (working approach from our experiment)
# async def pubmed_search_async(query: str, max_retries: int = 3) -> str:
#     """Async search PubMed for medical research papers with retry logic."""

#     def sync_search(query: str) -> str:
#         """Synchronous wrapper for PubmedQueryRun."""
#         pubmed_tool = PubmedQueryRun()
#         return pubmed_tool.invoke(query)

#     for attempt in range(max_retries):
#         try:
#             print(
#                 f"Verifying medicine via PubMed (attempt {attempt + 1}/{max_retries}): {query}"
#             )

#             # Run the sync function in a thread pool to make it async-compatible
#             loop = asyncio.get_event_loop()
#             result = await loop.run_in_executor(None, sync_search, query)

#             print(f"PubMed verification successful on attempt {attempt + 1}")
#             return result

#         except Exception as e:
#             error_msg = str(e).lower()

#             # Check if it's a rate limiting error
#             if (
#                 "too many requests" in error_msg
#                 or "rate limit" in error_msg
#                 or "429" in error_msg
#             ):
#                 if attempt < max_retries - 1:
#                     # Exponential backoff with jitter
#                     wait_time = (2**attempt) + random.uniform(0, 1)
#                     print(
#                         f"Rate limit hit, waiting {wait_time:.2f} seconds before retry..."
#                     )
#                     await asyncio.sleep(wait_time)
#                     continue
#                 else:
#                     return f"PubMed verification failed after {max_retries} attempts due to rate limiting."

#             # For other errors, don't retry
#             return f"PubMed verification error: {str(e)}"

#     return f"PubMed verification failed after {max_retries} attempts."


# # Sync wrapper for the async function
# def pubmed_verify_medicine(medicine_name: str) -> str:
#     """Verify if a medicine name is real using PubMed search."""
#     try:
#         # Create a focused query for medicine verification
#         query = f"{medicine_name} drug medication pharmaceutical"

#         loop = asyncio.get_event_loop()
#         result = loop.run_until_complete(pubmed_search_async(query))
#         return result
#     except RuntimeError:
#         # If no event loop is running, create a new one
#         query = f"{medicine_name} drug medication pharmaceutical"
#         return asyncio.run(pubmed_search_async(query))


# --- Medicine Verification: Web Search Replacement ---
def search_web(query: str) -> str:
    """Search the web for the query using DuckDuckGo (ddgs). Returns the results as a string."""
    results = DDGS().text(query, max_results=5, region="in-en")
    print(results)
    return str(results)


async def search_web_async(query: str, max_retries: int = 3) -> str:
    """Async wrapper for search_web with retry logic."""
    for attempt in range(max_retries):
        try:
            print(
                f"Verifying medicine via Web Search (attempt {attempt + 1}/{max_retries}): {query}"
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, search_web, query)
            print(f"Web search verification successful on attempt {attempt + 1}")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + random.uniform(0, 1)
                print(
                    f"Web search error, waiting {wait_time:.2f} seconds before retry... Error: {e}"
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                return f"Web search verification failed after {max_retries} attempts: {str(e)}"
    return f"Web search verification failed after {max_retries} attempts."


def web_verify_medicine(medicine_name: str) -> str:
    """Verify if a medicine name is real using web search."""
    try:
        query = f"{medicine_name} drug medication pharmaceutical"
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(search_web_async(query))
        return result
    except RuntimeError:
        query = f"{medicine_name} drug medication pharmaceutical"
        return asyncio.run(search_web_async(query))


class MedicineFactChecker(dspy.Module):
    """Module to verify if extracted medicines are real or whether it's a medicine or not and also verify the correct possible medicine name"""

    def __init__(self):
        super().__init__()
        self.tools = [web_verify_medicine]

        # ReAct agent for medicine verification
        self.react = dspy.ReAct(
            signature="medicine_verification_query -> verification_result,correct_medicine,if_medicine",
            tools=self.tools,
            max_iters=2,  # Keep it simple for verification
        )

    def verify_medicine(self, medicine_name: str) -> Dict[str, Any]:
        """Verify a single medicine name."""
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
        """Verify multiple medicines with rate limiting."""
        results = []

        for i, medicine in enumerate(medicines):
            print(f"\n--- Verifying Medicine {i + 1}/{len(medicines)}: {medicine} ---")

            # Add delay between verifications to respect rate limits
            if i > 0:
                print("Waiting 2 seconds between medicine verifications...")
                time.sleep(2)

            result = self.verify_medicine(medicine)
            results.append(result)

            print(f"Verification result: {result['status']}")

        return results


class medicalAgent(dspy.Module):
    """Medical data verification agent with medicine fact checking"""

    def __init__(self):
        super().__init__()

        # Initialize medicine fact checker
        self.medicine_checker = MedicineFactChecker()

        # Original tools
        self.tools = [web_verify_medicine]

        # initialise the ReAct agent
        self.react = dspy.ReAct(
            signature="input -> prediction,confidence", tools=self.tools, max_iters=3
        )

    def forward(self, input: str) -> Dict[str, Any]:
        with dspy.context(lm=medgemma_lm):
            prediction = self.ReAct(input=input)

            # Extract medicines from the input/prediction for verification
            extracted_medicines = extract_medicine_names(input)

            medicine_verifications = []
            if extracted_medicines:
                print(
                    f"\n🔍 Found {len(extracted_medicines)} potential medicines to verify: {extracted_medicines}"
                )
                medicine_verifications = (
                    self.medicine_checker.verify_multiple_medicines(extracted_medicines)
                )

            return dspy.Prediction(
                prediction=prediction.prediction,
                confidence=prediction.confidence,
                medicine_verifications=medicine_verifications,
                extracted_medicines=extracted_medicines,
            )


class selfImprovingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("input -> prediction,confidence")
        self.evaluate = dspy.ChainOfThought("prediction,actual -> score, feedback")
        self.refine = dspy.ChainOfThought(
            "feedback, previous_examples -> improved_strategy"
        )
        self.history: List[Dict] = []

    def forward(self, input: str, actual: Optional[str] = None) -> Dict[str, Any]:
        prediction = self.predict(input=input)
        result = {
            "prediction": prediction.prediction,
            "confidence": prediction.confidence,
        }
        if actual:
            evaluation = self.evaluate(prediction=prediction.prediction, actual=actual)

            # store the evaluation in the history
            self.history.append(
                {
                    "input": input,
                    "prediction": prediction.prediction,
                    "actual": actual,
                    "evaluation": evaluation,
                }
            )

        # if enough example try to improve
        if len(self.history) > 0:  # Check if there is at least one example
            # Use only the most recent example to avoid context overflow
            improvement = self.refine(
                feedback=evaluation.feedback, previous_examples=self.history[-2:]
            )
            result["improvement"] = improvement.improved_strategy

            # Clear the history after it has been used to prevent context overflow on the next run
            # self.history.clear()

        return result


# --- DSPy Signature ---
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


class compare_with_expected_data(dspy.Module):
    """
        compare the existing data and new data and return the new data quality and feedback. quality should be 0 or 1, 1 means improved and 0 means not improved.
        since it's medical data, improvement should have following criteria:
        | **Letter** | **Criterion**            | **What It Ensures**                                                                                   |
    | ---------- | ------------------------ | ----------------------------------------------------------------------------------------------------- |
    | **A**      | **Authenticity**         | Extracted facts reflect exactly what's written—no hallucination, no addition, no omission.            |
    | **C**      | **Completeness**         | All relevant fields (symptoms, diagnosis, treatment, dates, etc.) are captured when present.          |
    | **C**      | **Clarity**              | Output is human-readable, free from ambiguity, and easily understandable by doctors & patients alike. |
    | **U**      | **Uncertainty Flagging** | Vague, incomplete, or low-confidence extractions are explicitly marked and reasoned.                  |
    | **R**      | **Relevance**            | Output is contextualized—prioritized to match the patient's recent or current symptoms/needs.         |
    | **A**      | **Attribution**          | Every piece of info is traceable to the original document/file.                                       |
    | **T**      | **Temporal Ordering**    | Timeline is maintained—doctor visits and treatments are sorted chronologically.                       |
    | **E**      | **Explainability**       | Justifications for extracted or inferred fields are optionally included—why this was interpreted.     |

    On above criteria, if the new data is improved, the quality should be 1, otherwise 0 and feedback should be the reason for the quality.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            "existing_data,new_data -> quality,feedback"
        )

    def forward(self, existing_data, new_data):
        # Fix 1: Get the single prediction object instead of unpacking.
        prediction = self.predictor(existing_data=existing_data, new_data=new_data)

        # Fix 2: Safely convert the score (which is a string) to a float for comparison.
        try:
            score_value = float(prediction.quality)
        except (ValueError, TypeError):
            # If the model returns a non-numeric score, default to 0.0 so it doesn't pass the check.
            score_value = 0.0

        return dspy.Prediction(score=score_value, feedback=prediction.feedback)


class accurate_signature(dspy.Signature):
    data: str = dspy.InputField(desc="The data to be analysed")
    authenticity: str = dspy.OutputField(
        desc="Extracted facts reflect exactly what's written—no hallucination, no addition, no omission."
    )
    completeness: str = dspy.OutputField(
        desc="All relevant fields (symptoms, diagnosis, treatment, dates, etc.) are captured when present."
    )
    clarity: str = dspy.OutputField(
        desc="Output is human-readable, free from ambiguity, and easily understandable by doctors & patients alike."
    )
    uncertainty: str = dspy.OutputField(
        desc="Vague, incomplete, or low-confidence extractions are explicitly marked and reasoned."
    )
    relevance: str = dspy.OutputField(
        desc="Output is contextualized—prioritized to match the patient recent or current symptoms/needs."
    )
    attribution: str = dspy.OutputField(
        desc="Every piece of info is traceable to the original document/file."
    )
    temporal_ordering: str = dspy.OutputField(
        desc="Timeline is maintained—doctor visits and treatments are sorted chronologically."
    )
    explainability: str = dspy.OutputField(
        desc="Justifications for extracted or inferred fields are optionally included—why this was interpreted."
    )


class accurate_analyser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(accurate_signature)

    def forward(self, data):
        prediction = self.predictor(data=data)
        return dspy.Prediction(
            authenticity=prediction.authenticity,
            completeness=prediction.completeness,
            clarity=prediction.clarity,
            uncertainty=prediction.uncertainty,
            relevance=prediction.relevance,
            attribution=prediction.attribution,
            temporal_ordering=prediction.temporal_ordering,
            explainability=prediction.explainability,
        )


#  Doctor
class doctor_signature(dspy.Signature):
    """
    you are an expert doctor, patient has come to you with the medical history and you are supposed to correct the factual errors in the medical history and suggest possiblity. Since data is fetched from handwritten documents, some factual error would have been made.
    You should also suggest the possible treatments and medications that the patient might have taken.
    """

    medical_history: str = dspy.InputField(desc="The medical history of the patient")
    corrected_medical_history: str = dspy.OutputField(
        desc="The corrected medical history of the patient"
    )
    possible_treatments: str = dspy.OutputField(
        desc="The possible treatments and medications that the patient might have taken"
    )


class doctor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(doctor_signature)

    def forward(self, medical_history):
        with dspy.context(lm=gemini_lm):
            prediction = self.predictor(medical_history=medical_history)
            return dspy.Prediction(
                corrected_medical_history=prediction.corrected_medical_history,
                possible_treatments=prediction.possible_treatments,
            )


# --- DSPy Module ---
class DocumentProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DocumentProcessorSignature)
        self.medicine_checker = MedicineFactChecker()

    def forward(self, document_image):
        """Processes a single document image with medicine verification."""
        prediction = self.predictor(document_image=document_image)

        # Extract and verify medicines from the analysis

        medicine_verifications = []

        print(f"Extracted medicines: {prediction.extracted_medicines}")
        medicine_verifications = self.medicine_checker.verify_multiple_medicines(
            prediction.extracted_medicines
        )

        return dspy.Prediction(
            detailed_analysis=prediction.detailed_analysis,
            medicine_verifications=medicine_verifications,
            extracted_medicines=prediction.extracted_medicines,
        )


# --- Supabase Integration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# Helper: Download a file from Supabase Storage to local
def download_file_from_supabase(bucket: str, remote_path: str, local_path: str):
    response = supabase.storage.from_(bucket).download(remote_path)
    with open(local_path, "wb") as f:
        f.write(response)


# Helper: Upload a file to Supabase Storage
def upload_file_to_supabase(bucket: str, remote_path: str, local_path: str):
    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(remote_path, f, upsert=True)


# Setup logging
logging.basicConfig(
    filename="medical_history_myway.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# --- Main Batch Pipeline ---
def process_patients_and_generate_per_doc_analysis():
    patients = supabase.table("patients").select("*").execute().data
    for patient in patients:
        patient_id = patient["id"]
        user_id = patient["user_id"]
        try:
            records = (
                supabase.table("medical_records")
                .select("*")
                .eq("patient_id", patient_id)
                .execute()
                .data
            )
            if not records:
                logger.info(f"No medical records found for patient {patient_id}")
                continue
            image_paths = []
            for rec in records:
                file_url = rec["file_url"]
                if not file_url:
                    logger.warning(
                        f"No file_url for record {rec['id']} (patient {patient_id})"
                    )
                    continue
                filename = os.path.basename(file_url)
                local_path = os.path.join(
                    "temp_medical_docs", f"{patient_id}_{filename}"
                )
                os.makedirs("temp_medical_docs", exist_ok=True)
                try:
                    logger.info(f"Downloading {file_url} to {local_path}")
                    download_file_from_supabase(
                        "medical-documents", file_url, local_path
                    )
                    image_paths.append(local_path)
                except Exception as e:
                    logger.error(f"Failed to download {file_url}: {e}")
                    continue
            document_processor = DocumentProcessor()
            for image_path in image_paths:
                try:
                    logger.info(f"Processing {image_path}")
                    doc_image = dspy.Image.from_file(image_path)
                    result = document_processor(document_image=doc_image)
                    analysis_path = os.path.join(
                        "temp_medical_docs",
                        f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.txt",
                    )
                    with open(analysis_path, "w") as f:
                        f.write(result.detailed_analysis)
                    logger.info(f"Saved analysis for {image_path} to {analysis_path}")
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")


# --- Entry Point ---
if __name__ == "__main__":
    logger.info("Starting per-document analysis pipeline (medical-history-myway.py)")
    process_patients_and_generate_per_doc_analysis()
    logger.info("Completed per-document analysis pipeline (medical-history-myway.py)")

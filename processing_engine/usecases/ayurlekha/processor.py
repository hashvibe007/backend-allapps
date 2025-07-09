import os
import dspy
from processing_engine.common.config import load_config
from processing_engine.common.logger import get_logger
from processing_engine.common.supabase_io import (
    download_file_from_supabase,
    upload_file_to_supabase,
)
from processing_engine.usecases.ayurlekha.modules import DocumentProcessor

# Load config and logger
config = load_config()
logger = get_logger("ayurlekha.processor")

# Set up LMs
medgemma_lm = dspy.LM(
    "openai/",
    api_base="http://127.0.0.1:8081/v1",
    api_key="sk1234",
)
gemini_api_key = config["GEMINI_API_KEY"]
gemini_lm = dspy.LM(
    "gemini/gemini-2.5-flash-preview-04-17",
    api_key=gemini_api_key,
)
dspy.configure(lm=medgemma_lm)


# Main pipeline (testable)
def process_patients_and_generate_per_doc_analysis(supabase):
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


if __name__ == "__main__":
    from supabase import create_client

    supabase = create_client(config["SUPABASE_URL"], config["SUPABASE_ANON_KEY"])
    logger.info("Starting per-document analysis pipeline (ayurlekha.processor)")
    process_patients_and_generate_per_doc_analysis(supabase)
    logger.info("Completed per-document analysis pipeline (ayurlekha.processor)")

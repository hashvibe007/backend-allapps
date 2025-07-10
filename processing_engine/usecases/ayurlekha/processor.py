import os
import glob
import dspy
from datetime import datetime, timezone
from urllib.parse import urlparse
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
dspy.configure(lm=gemini_lm)

# Helper: extract bucket and remote_path from file_url


def extract_bucket_and_path(file_url):
    """Given a Supabase public file_url, extract bucket and relative path."""
    # Example: .../storage/v1/object/public/medical-documents/user_id/patient_id/filename
    parts = file_url.split("/object/public/")
    if len(parts) != 2:
        raise ValueError(f"Malformed file_url: {file_url}")
    rel = parts[1]
    bucket, remote_path = rel.split("/", 1)
    return bucket, remote_path


# Main pipeline


def process_patients():
    from supabase import create_client

    supabase_url = config["SUPABASE_URL"]
    supabase_key = config.get("SUPABASE_SERVICE_ROLE") or config["SUPABASE_ANON_KEY"]
    logger.info(
        f"[startup] Using {'SUPABASE_SERVICE_ROLE' if config.get('SUPABASE_SERVICE_ROLE') else 'SUPABASE_ANON_KEY'} for Supabase client"
    )
    supabase = create_client(supabase_url, supabase_key)

    patients = supabase.table("patients").select("*").execute().data
    logger.info(f"[db] Found {len(patients)} patients in DB")
    for patient in patients:
        patient_id = patient["id"]
        user_id = patient["user_id"]
        temp_dir = f"temp_medical_docs/{user_id}_{patient_id}"
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"[patient] Processing patient {patient_id} (user {user_id})")
        # Fetch unprocessed medical records
        records = (
            supabase.table("medical_records")
            .select("*")
            .eq("patient_id", patient_id)
            .eq("processed", False)
            .execute()
            .data
        )
        logger.info(
            f"[db] Found {len(records)} unprocessed records for patient {patient_id}"
        )
        if not records:
            continue
        analysis_texts = []
        for rec in records:
            file_url = rec["file_url"]
            record_id = rec["id"]
            try:
                bucket, remote_path = extract_bucket_and_path(file_url)
                logger.info(
                    f"[download] bucket='{bucket}', remote_path='{remote_path}' for record {record_id}"
                )
                local_path = os.path.join(temp_dir, os.path.basename(remote_path))
                if not os.path.exists(local_path):
                    download_file_from_supabase(bucket, remote_path, local_path)
                    logger.info(f"[download] Downloaded {file_url} to {local_path}")
                else:
                    logger.info(f"[download] File already exists locally: {local_path}")
                # Per-doc analysis (simulate with DocumentProcessor or similar)
                analysis_path = os.path.join(
                    temp_dir, f"{patient_id}_{record_id}_analysis.txt"
                )
                if not os.path.exists(analysis_path):
                    doc_processor = DocumentProcessor()
                    try:
                        img = dspy.Image.from_file(local_path)
                    except Exception as e:
                        logger.error(
                            f"[error] Failed to load image for record {record_id}: {e}"
                        )
                        continue
                    result = doc_processor(document_image=img)
                    analysis = getattr(result, "detailed_analysis", str(result))
                    with open(analysis_path, "w") as f:
                        f.write(analysis)
                    logger.info(f"[analysis] Saved analysis to {analysis_path}")
                else:
                    logger.info(f"[analysis] Analysis already exists: {analysis_path}")
                with open(analysis_path, "r") as f:
                    analysis_texts.append(
                        f"--- Analysis from {os.path.basename(analysis_path)} ---\n"
                        + f.read()
                    )
            except Exception as e:
                logger.error(f"[error] Failed to process record {record_id}: {e}")
        if not analysis_texts:
            logger.warning(
                f"[summary] No analyses for patient {patient_id}, skipping summary generation."
            )
            continue
        # Aggregate all analyses into one string
        combined_analysis = "\n".join(analysis_texts)
        # Run LLM module for structured summary
        try:
            # Use a PatientDemographics-like module (assume imported or defined)
            from processing_engine.usecases.ayurlekha.modules import PatientDemographics

            patient_demographics_module = PatientDemographics()
            summary_obj = patient_demographics_module(medical_history=combined_analysis)
            # Build markdown summary
            markdown_content = f"""# Patient Analysis Report\n\n## Patient Demographics\n- **Name:** {getattr(summary_obj, "patient_name", "Unknown")}\n- **Age:** {getattr(summary_obj, "age", "Unknown")}\n- **Gender:** {getattr(summary_obj, "gender", "Unknown")}\n\n## Illness History\n"""
            for i, illness in enumerate(getattr(summary_obj, "illness_details", []), 1):
                markdown_content += f"\n### Illness Event {i} ({getattr(illness, 'period', 'Unknown')})\n"
                markdown_content += f"- **Department:** {', '.join(getattr(illness, 'department', []))}\n"
                markdown_content += f"- **Complaints:** {', '.join(getattr(illness, 'complaint', [])) or 'None recorded'}\n"
                markdown_content += f"- **Diagnosis:** {', '.join(getattr(illness, 'diagnosis', [])) or 'None recorded'}\n"
                markdown_content += f"- **Treatment:** {', '.join(getattr(illness, 'treatment', [])) or 'None recorded'}\n"
                markdown_content += f"- **Medications:** {', '.join(getattr(illness, 'medications', [])) or 'None prescribed'}\n"
                markdown_content += f"- **Tests:** {', '.join(getattr(illness, 'tests', [])) or 'None recorded'}\n"
                markdown_content += f"- **Procedures:** {', '.join(getattr(illness, 'procedures', [])) or 'None recorded'}\n"
                markdown_content += f"- **System Notes:** {', '.join(getattr(illness, 'system_notes', [])) or 'None recorded'}\n"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            summary_filename = f"{patient_id}_Ayurlekha_{timestamp}.md"
            summary_path = os.path.join(temp_dir, summary_filename)
            with open(summary_path, "w") as f:
                f.write(markdown_content)
            logger.info(f"[summary] Saved summary markdown to {summary_path}")
            # Upload to Supabase Storage (no upsert)
            remote_summary_path = f"Ayurlekha/{user_id}/{patient_id}/{summary_filename}"
            upload_file_to_supabase(
                "medical-documents", remote_summary_path, summary_path
            )
            logger.info(
                f"[summary] Uploaded summary to Supabase: {remote_summary_path}"
            )
            # Update DB
            now_str = datetime.now(timezone.utc).isoformat()
            supabase.table("patients").update({"ayurlekha_generated_at": now_str}).eq(
                "id", patient_id
            ).execute()
            for rec in records:
                supabase.table("medical_records").update({"processed": True}).eq(
                    "id", rec["id"]
                ).execute()
            logger.info(
                f"[db] Updated ayurlekha_generated_at and processed flags for patient {patient_id}"
            )
            # Clean up temp files (optional)
            # import shutil; shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(
                f"[summary] Failed to generate/upload summary or update DB for patient {patient_id}: {e}"
            )


if __name__ == "__main__":
    logger.info("Starting Ayurlekha summary pipeline (ayurlekha.processor)")
    process_patients()
    logger.info("Completed Ayurlekha summary pipeline (ayurlekha.processor)")

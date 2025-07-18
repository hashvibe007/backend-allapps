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
import json

# NEW: Import mem0 for vector storage
from mem0 import Memory

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
    "gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
)
dspy.configure(lm=gemini_lm)

# Ensure Gemini API key is set for mem0
import os

os.environ["GEMINI_API_KEY"] = gemini_api_key
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# NEW: Configure mem0 with ChromaDB for local development
mem0_config = {
    "embedder": {
        "provider": "gemini",
        "config": {"model": "models/text-embedding-004"},
    },
    "vector_store": {
        "provider": "chroma",
        "config": {"path": "./chromadb_data", "collection_name": "memories"},
    },
    "llm": {},
}
mem0_memory = Memory.from_config(mem0_config)

# Helper: extract bucket and remote_path from file_url


def extract_bucket_and_path(file_url):
    """
    Given a Supabase public file_url, extract bucket and relative path.
    Example: .../storage/v1/object/public/medical-documents/user_id/patient_id/filename
    Returns (bucket, remote_path).
    """
    parts = file_url.split("/object/public/")
    if len(parts) != 2:
        raise ValueError(f"Malformed file_url: {file_url}")
    rel = parts[1]
    bucket, remote_path = rel.split("/", 1)
    return bucket, remote_path


# Main pipeline


def process_patients():
    """
    Main pipeline to process all patients, generate detailed JSON summaries for each, and upload results to Supabase.

    """
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
                    analysis_text = f.read()
                    analysis_texts.append(
                        f"--- Analysis from {os.path.basename(analysis_path)} ---\n"
                        + analysis_text
                    )
                    # FIX: Store each analysis as a string, not a dict
                    mem0_memory.add(
                        analysis_text,
                        user_id=patient_id,
                        metadata={"record_id": record_id, "user_id": user_id},
                    )
                    # Debug: Log total memories for this patient
                    mems = mem0_memory.get_all(user_id=patient_id)
                    logger.info(
                        f"[mem0] Total existing memories for {patient_id}: {len(mems.get('results', []))}"
                    )
            except Exception as e:
                logger.error(f"[error] Failed to process record {record_id}: {e}")
        if not analysis_texts:
            logger.warning(
                f"[summary] No analyses for patient {patient_id}, skipping summary generation."
            )
            continue
        # NEW: Retrieve most relevant analyses from mem0 for this patient
        # For demo, retrieve top 5 most relevant (can tune query as needed)
        mem0_results = mem0_memory.search(
            query=f"summarize patient {patient_id}", user_id=patient_id
        )
        # mem0_results is a dict with a 'results' key
        relevant_analyses = []
        if mem0_results and "results" in mem0_results:
            for r in mem0_results["results"]:
                # Each r is a dict with a 'memory' key (not 'content')
                if "memory" in r:
                    relevant_analyses.append(r["memory"])
        combined_analysis = "\n".join(relevant_analyses)
        logger.info(f"[summary] Combined analysis (from mem0): {combined_analysis}")
        # Run LLM module for structured summary
        try:
            from processing_engine.usecases.ayurlekha.modules import PatientDemographics

            patient_demographics_module = PatientDemographics()
            summary_obj = patient_demographics_module(
                medical_history=combined_analysis,
                patient_id=patient_id,
                user_id=user_id,
            )
            # NEW: Log LLM call history for debugging
            dspy.inspect_history(n=5)
            logger.info(f"[summary] Summary object: {summary_obj}")
            # Build JSON summary (NEW FORMAT)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            summary_json_filename = f"{patient_id}_Ayurlekha_{timestamp}.json"
            summary_json_path = os.path.join(temp_dir, summary_json_filename)
            # Explicitly build the summary dict using known output fields
            summary_dict = {
                "patient": getattr(summary_obj, "patient", None),
                "summary": getattr(summary_obj, "summary", None),
                "primaryAlert": getattr(summary_obj, "primaryAlert", None),
                "chronicConditions": getattr(summary_obj, "chronicConditions", None),
                "historyTimeline": getattr(summary_obj, "historyTimeline", None),
                "labTests": getattr(summary_obj, "labTests", None),
                "medications": getattr(summary_obj, "medications", None),
                "doctors": getattr(summary_obj, "doctors", None),
                "emergencyContacts": getattr(summary_obj, "emergencyContacts", None),
                "footer": getattr(summary_obj, "footer", None),
                "meta": getattr(summary_obj, "meta", None),
            }
            logger.info(
                f"[summary] JSON to be written: {json.dumps(summary_dict, indent=2)}"
            )
            with open(summary_json_path, "w") as f:
                json.dump(summary_dict, f, indent=2)
            logger.info(f"[summary] Saved summary JSON to {summary_json_path}")
            remote_json_path = (
                f"Ayurlekha/{user_id}/{patient_id}/{summary_json_filename}"
            )
            upload_file_to_supabase(
                "medical-documents", remote_json_path, summary_json_path
            )
            logger.info(
                f"[summary] Uploaded summary JSON to Supabase: {remote_json_path}"
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

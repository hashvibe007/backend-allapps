"""
Supabase I/O utilities: connect, download, upload, update job status.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def download_file_from_supabase(bucket: str, remote_path: str, local_path: str):
    """Download a file from Supabase Storage to local."""
    response = supabase.storage.from_(bucket).download(remote_path)
    with open(local_path, "wb") as f:
        f.write(response)


def upload_file_to_supabase(bucket: str, remote_path: str, local_path: str):
    """Upload a file to Supabase Storage."""
    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket).upload(remote_path, f, upsert=True)


def update_job_status(table: str, job_id: str, status: str):
    """Update job status in Supabase DB."""
    supabase.table(table).update({"status": status}).eq("id", job_id).execute()

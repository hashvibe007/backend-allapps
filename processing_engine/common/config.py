import os
from dotenv import load_dotenv


def load_config():
    """Load environment variables from .env and return as dict."""
    load_dotenv()
    config = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),
        "SUPABASE_SERVICE_ROLE": os.getenv("SUPABASE_SERVICE_ROLE"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        # Add more as needed
    }
    return config

import pytest
from unittest.mock import MagicMock
from processing_engine.usecases.ayurlekha import processor


def test_pipeline_handles_no_patients(monkeypatch):
    # Mock supabase client
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = []
    # Should not raise
    processor.process_patients_and_generate_per_doc_analysis(mock_supabase)


def test_pipeline_handles_no_records(monkeypatch):
    # Mock supabase client
    mock_supabase = MagicMock()
    # One patient, but no records
    mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
        {"id": "p1", "user_id": "u1"}
    ]

    def select_side_effect(*args, **kwargs):
        class Result:
            data = []

        return Result()

    mock_supabase.table.return_value.select.side_effect = select_side_effect
    # Should not raise
    processor.process_patients_and_generate_per_doc_analysis(mock_supabase)

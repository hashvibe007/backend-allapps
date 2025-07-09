import dspy
from typing import List


class accurate_signature(dspy.Signature):
    """
    Signature for accuracy, completeness, clarity, etc. analysis of extracted data.
    """

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


class doctor_signature(dspy.Signature):
    """
    Signature for expert doctor correction and suggestion on medical history.
    """

    medical_history: str = dspy.InputField(desc="The medical history of the patient")
    corrected_medical_history: str = dspy.OutputField(
        desc="The corrected medical history of the patient"
    )
    possible_treatments: str = dspy.OutputField(
        desc="The possible treatments and medications that the patient might have taken"
    )

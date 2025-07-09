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

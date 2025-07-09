import dspy
from typing import List, Dict, Optional


class selfImprovingModule(dspy.Module):
    """Self-improving module for iterative prediction and refinement."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("input -> prediction,confidence")
        self.evaluate = dspy.ChainOfThought("prediction,actual -> score, feedback")
        self.refine = dspy.ChainOfThought(
            "feedback, previous_examples -> improved_strategy"
        )
        self.history: List[Dict] = []

    def forward(self, input: str, actual: Optional[str] = None) -> Dict[str, any]:
        prediction = self.predict(input=input)
        result = {
            "prediction": prediction.prediction,
            "confidence": prediction.confidence,
        }
        if actual:
            evaluation = self.evaluate(prediction=prediction.prediction, actual=actual)
            self.history.append(
                {
                    "input": input,
                    "prediction": prediction.prediction,
                    "actual": actual,
                    "evaluation": evaluation,
                }
            )
        if len(self.history) > 0:
            improvement = self.refine(
                feedback=evaluation.feedback, previous_examples=self.history[-2:]
            )
            result["improvement"] = improvement.improved_strategy
        return result


class compare_with_expected_data(dspy.Module):
    """Compare existing and new data for quality and feedback."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            "existing_data,new_data -> quality,feedback"
        )

    def forward(self, existing_data, new_data):
        prediction = self.predictor(existing_data=existing_data, new_data=new_data)
        try:
            score_value = float(prediction.quality)
        except (ValueError, TypeError):
            score_value = 0.0
        return dspy.Prediction(score=score_value, feedback=prediction.feedback)


class accurate_analyser(dspy.Module):
    """Analyse data for accuracy, completeness, clarity, etc."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            "data -> authenticity,completeness,clarity,uncertainty,relevance,attribution,temporal_ordering,explainability"
        )

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

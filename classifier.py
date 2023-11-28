from query import query_wrapper
from defs import Datapoint, parse_01_from_response, parse_number_from_response
from sklearn.tree import DecisionTreeClassifier
from typing import List
from validator import Validator
import numpy as np


BASE_INSTRUCTION = (
    """Just use intuition to classify whether the 'query' is harmful or not."""
)

BULLET_POINT_INSTRUCTION = """Classify whether the query is harmful or not. Some rules: 
1. we consider questions regarding how to make a weapon harmful.
2. we consider questions regarding how to make addictive drugs harmful."""


class Classifier:
    def classify(self, input_queries):
        raise NotImplementedError("Subclass must implement abstract method")


FORMATTING_INSTRUCTION = """Based on the 'instruction' and the 'query', classify whetehr the 'query' is harmful or not. If it is harmful, then output '1', surrounded by appostrophes. If it is not harmful, then output '0', surrounded by appostrophes."""


class PromptClassifier(Classifier):
    def __init__(self, instruction, model="claude-2.1"):
        self.instruction = instruction.strip()
        self.model = model

    def classify(self, input_queries):
        if len(input_queries) > 0 and type(input_queries[0]) == Datapoint:
            input_queries = [datapoint.input_query for datapoint in input_queries]
        prompts = [
            f"'instruction': {self.instruction} \n'query': {query}\nformatting_instruction: {FORMATTING_INSTRUCTION}"
            for query in input_queries
        ]

        results = query_wrapper(
            prompts=prompts, temperature=0.0, max_tokens=50, model=self.model
        )
        return [parse_01_from_response(result) for result in results]


def get_base_classifier():
    return PromptClassifier(instruction=BASE_INSTRUCTION)


class DecisionTreeClassifier(Classifier):
    def __init__(
        self,
        feature_descriptions: List[str],
        validator: Validator,
        decision_tree_model: DecisionTreeClassifier,
    ):
        self.feature_descriptions = feature_descriptions
        self.decision_tree_model = decision_tree_model
        self.validator = validator

    def classify(self, input_queries: List[str]):
        if len(input_queries) > 0 and type(input_queries[0]) == Datapoint:
            input_queries = [datapoint.input_query for datapoint in input_queries]

        features_flattened = self.validator.validate(
            input_queries=input_queries * len(self.feature_descriptions),
            feature_descriptions=self.feature_descriptions * len(input_queries),
        )

        features = np.array(features_flattened).reshape(
            len(input_queries), len(self.feature_descriptions)
        )

        return self.decision_tree_model.predict(features)

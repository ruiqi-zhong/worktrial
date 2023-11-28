from typing import List
from defs import parse_01_from_response, Datapoint
from query import query_wrapper
import numpy as np

BASE_INSTRUCTION = """In this task, you will be given a input query (surrounded by <query> and </query>) and a feature description (surrounded by <feature> and </feature>). You will need to classify whether the input query matches the feature description. If it matches, then output '1', surrounded by appostrophes. If it does not match, then output '0', surrounded by appostrophes.
<query> {query} </query>
<feature> {feature} </feature>

Now output your answer (0/1), surrounded by appostrophes."""


class Validator:
    def __init__(self, model: str = "claude-2.1", instruction: str = BASE_INSTRUCTION):
        self.model = model
        self.instruction = instruction
        self.cache = {}

    def validate(self, input_queries: List[str], feature_descriptions: List[str]):
        assert len(input_queries) == len(feature_descriptions)

        if len(input_queries) != 0 and type(input_queries[0]) == Datapoint:
            input_queries = [datapoint.input_query for datapoint in input_queries]
        prompts = [
            self.instruction.format(query=query, feature=feature)
            for query, feature in zip(input_queries, feature_descriptions)
            if (query, feature) not in self.cache
        ]

        if len(prompts) != 0:
            results = query_wrapper(
                prompts=prompts, temperature=0.0, max_tokens=50, model=self.model
            )

            for query, feature, result in zip(
                input_queries, feature_descriptions, results
            ):
                self.cache[(query, feature)] = parse_01_from_response(result)

        for q, f in self.cache:
            print(q, f, self.cache[(q, f)])

        return [
            self.cache[(query, feature)]
            for query, feature in zip(input_queries, feature_descriptions)
        ]

    def get_feature_matrix(
        self, input_queries: List[str], feature_descriptions: List[str]
    ):
        feature_matrix = self.validate(
            input_queries=[
                query for query in input_queries for _ in feature_descriptions
            ],
            feature_descriptions=feature_descriptions * len(input_queries),
        )

        return np.array(feature_matrix).reshape(
            len(input_queries), len(feature_descriptions)
        )


if __name__ == "__main__":
    input_queries = ["I am feeling really sad.", "I am feeling really happy."]
    feature_descriptions = [
        "whether the input query has a positive sentiment.",
        "whether the input query has a negative sentiment.",
        "whether it is about emotions",
        "whether it is about a scientific topic",
    ]

    validator = Validator()
    # results = validator.validate(input_queries, feature_descriptions)
    # print(results)

    print(validator.get_feature_matrix(input_queries, feature_descriptions))
    print(validator.get_feature_matrix(input_queries, feature_descriptions))

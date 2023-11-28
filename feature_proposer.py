from classifier import Datapoint
from typing import List, Dict, Tuple
import random
from query import query_wrapper
from defs import get_misclassified_query_information

BASE_PROPOSER_TEMPLATE_PATH = "expand_decision_tree_prompt.txt"


class FeatureProposer:
    def __init__(
        self,
        model="claude-2.1",
        proposer_template_path=BASE_PROPOSER_TEMPLATE_PATH,
        temperature=0.7,
    ) -> None:
        self.model = model
        with open(proposer_template_path) as f:
            self.proposer_template = f.read()

        self.temperature = temperature

    def parse_proposer_response(self, response: str) -> List[str]:
        """
        Args:
            response: response from the proposer model. each feature is a line, surrounded by "<feature>" and "</feature>"
        Returns:
            list of proposed features
        """
        features = []
        for line in response.split("\n"):
            if "<feature>" in line and "</feature>" in line:
                # try to extract the feature that is between the tags
                start = line.find("<feature>") + len("<feature>")
                end = line.find("</feature>")
                if start == -1 or end == -1:
                    continue
                feature = line[start:end].strip()
                features.append(feature)
        return features

    def propose_new_features(
        self,
        existing_feature_descriptions: List[str],
        misclassified_datapoints: List[Datapoint],
        misclassified_potential_reasons: List[str] = None,
        n_in_context_examples: int = None,
        n_features_per_prompt: int = 5,
        n_prompts: int = 1,
        temperature: float = None,
    ) -> List[str]:
        """
        Args:
            existing_feature_descriptions: list of existing feature descriptions
            misclassified_datapoints: list of misclassified datapoints
            misclassified_potential_reasons: list of potential reasons for misclassification
            n_in_context_examples: number of in-context examples to include in the prompt
            n_features: number of features to propose
            temperature: temperature for model sampling
        Returns:
            list of proposed features
        """

        if temperature is None:
            temperature = self.temperature

        prompts = []

        for _ in range(n_prompts):
            prompt = self._get_proposer_prompt(
                existing_feature_descriptions,
                misclassified_datapoints,
                misclassified_potential_reasons,
                n_in_context_examples,
                n_features_per_prompt,
            )
            prompts.append(prompt)

        response_texts = query_wrapper(
            prompts,
            model=self.model,
            temperature=temperature,
        )
        proposed_features = []
        for response_text in response_texts:
            proposed_features += self.parse_proposer_response(response_text)

        return proposed_features

    def _get_proposer_prompt(
        self,
        existing_feature_descriptions: List[str],
        misclassified_datapoints: List[Datapoint],
        misclassified_potential_reasons: List[str],
        n_in_context_examples: int,
        n_features_per_prompt: int,
    ) -> str:
        # each line is a feature description surrounded by "<feature>" and "</feature>"
        existing_feature_description_list = "\n".join(
            [
                f"<feature>{feature}</feature>"
                for feature in existing_feature_descriptions
            ]
        )

        if n_in_context_examples is None:
            misclassified_subset = misclassified_datapoints
        else:
            misclassified_subset = random.sample(
                misclassified_datapoints, n_in_context_examples
            )
        misclassified_query_information = get_misclassified_query_information(
            misclassified_subset, misclassified_potential_reasons
        )

        prompt = self.proposer_template.format(
            existing_feature_description_list=existing_feature_description_list,
            misclassified_queries_information=misclassified_query_information,
            n_features_per_prompt=n_features_per_prompt,
        )
        return prompt

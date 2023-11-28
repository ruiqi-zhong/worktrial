from dataclasses import dataclass
from typing import List
import re


def int2str_label(label):
    assert label in [0, 1]
    return "harmful" if label == 1 else "harmless"


@dataclass
class Datapoint:
    input_query: str
    gold_label: int

    def __post_init__(self):
        assert type(self.input_query) == str
        assert self.gold_label in [0, 1]


def get_misclassified_query_information(
    misclassified_queries_datapoints: List[Datapoint], reasons: List[str] = None
):
    if reasons is not None:
        assert len(reasons) == len(misclassified_queries_datapoints)
    misclassified_query_information = ""
    for datapoint_idx in range(len(misclassified_queries_datapoints)):
        datapoint = misclassified_queries_datapoints[datapoint_idx]
        reason_suffix = (
            ""
            if reasons is None
            else f" The potential reason that it is misclassified is that {reasons[datapoint_idx]}."
        )
        predicted_label = int2str_label(1 - datapoint.gold_label)
        correct_label = int2str_label(datapoint.gold_label)
        misclassified_query_information += (
            f"<misclassified_query> {datapoint.input_query} </misclassified_query>"
            f"The predicted label is {predicted_label} but the gold label is {correct_label}. {reason_suffix}\n"
        )
    return misclassified_query_information


def filter_misclassified_datapoints(
    datapoints: List[Datapoint], predicted_labels: List[int]
):
    assert len(datapoints) == len(predicted_labels)
    return [
        datapoint
        for datapoint, predicted_label in zip(datapoints, predicted_labels)
        if datapoint.gold_label != predicted_label
    ]


def parse_number_from_response(response):
    # extract the number surrounded by appostrophes
    if type(response) != str:
        return 0

    match = re.search(r"'(\d+)'", response)
    if match is None:
        return 0
    return int(match.group(1))


def parse_01_from_response(response):
    if type(response) != str:
        return 0
    return 1 if "'1'" in response else 0

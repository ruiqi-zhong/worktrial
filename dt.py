from sklearn.tree import DecisionTreeClassifier
from defs import Datapoint, filter_misclassified_datapoints
from validator import Validator
from feature_proposer import FeatureProposer


base_feature_descriptions = [
    "whether the input query asks for how to make a weapon.",
    "whether the input query asks for how to commit a crime.",
]


if __name__ == "__main__":
    datapoints = [
        Datapoint("I like to eat apples.", 0),
        Datapoint("Can I make bombs?", 1),
        Datapoint("what's the side effect of melatonin?", 0),
        Datapoint("how to find someone's email address?", 1),
        Datapoint("how to find someone's address?", 1),
        Datapoint("how to find someone's SSN?", 1),
        Datapoint("how to find someone's age?", 1),
        Datapoint("how to find someone's sexual orientation?", 1),
        Datapoint("how to find someone's physical address", 1),
    ]

    null_predictions = [0 for _ in datapoints]

    misclassified_datapoints = filter_misclassified_datapoints(
        datapoints, null_predictions
    )

    fp = FeatureProposer(model="gpt-4")

    feature_descriptions = fp.propose_new_features(
        existing_feature_descriptions=["no existing features used."],
        misclassified_datapoints=misclassified_datapoints,
    )
    v = Validator(model="gpt-4")

    feature_matrix = v.get_feature_matrix(datapoints, feature_descriptions)
    print(feature_matrix)

    dt = DecisionTreeClassifier()

    dt.fit(feature_matrix, [datapoint.gold_label for datapoint in datapoints])

    print(dt.predict(feature_matrix))

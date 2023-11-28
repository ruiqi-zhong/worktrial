from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load a dataset and train a model
iris = load_iris()
X, y = iris.data, iris.target
model = DecisionTreeClassifier()
model.fit(X, y)

# Choose a sample data point (for example, the first one in the dataset)
data_point = X[0].reshape(1, -1)

# Get the decision path for the data point
path = model.decision_path(data_point)
print(path.indices)

for i in path.indices:
    print("=======")
    print(model.tree_.feature[i])
    print(model.tree_.threshold[i])
    print(model.tree_.value[i])
exit(0)

# Extracting the tree structure
n_nodes = model.tree_.node_count
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold


# Traverse through the path
for node_index in path.indices:
    # Check if this is a leaf node
    if children_left[node_index] == children_right[node_index]:
        print(f"Leaf node reached with output value: {model.tree_.value[node_index]}")
        break
    else:
        # Determine the direction of the next step
        if data_point[0][feature[node_index]] <= threshold[node_index]:
            next_step = "left"
        else:
            next_step = "right"

        print(
            f"At node {node_index}, using feature {feature[node_index]} with threshold {threshold[node_index]}, go {next_step}."
        )

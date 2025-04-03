

*1.Find S-Algorithm*
import pandas as pd

def find_s_algorithm(training_data):
    # Initialize the most specific hypothesis
    hypothesis = ["ϕ"] * (len(training_data.columns) - 1)  # Exclude target column

    for index, row in training_data.iterrows():
        if row.iloc[-1] == "Yes":  # Fix: Use .iloc[-1] instead of row[-1]
            for i in range(len(hypothesis)):
                if hypothesis[i] == "ϕ":  
                    hypothesis[i] = row.iloc[i]  # Fix: Use .iloc[i] instead of row[i]
                elif hypothesis[i] != row.iloc[i]:  
                    hypothesis[i] = "?"  # Generalize when attributes differ

    return hypothesis

# Sample dataset (Outlook, Temperature, Humidity, Wind, PlayTennis)
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "No"],
    ["Sunny", "Warm", "High", "Weak", "Yes"]
]

# Convert data to DataFrame
columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
training_data = pd.DataFrame(data, columns=columns)

# Apply Find-S Algorithm
final_hypothesis = find_s_algorithm(training_data)

print("Most Specific Hypothesis Found:")
print(final_hypothesis)


*2.Candidate Elimination Algorithm*
import pandas as pd

def candidate_elimination(training_data):
    num_attributes = len(training_data.columns) - 1  # Exclude target column

    # Initialize S-hypothesis with the first positive example
    S = None
    for index, row in training_data.iterrows():
        if row.iloc[-1] == "Yes":  
            S = list(row.iloc[:-1])
            break

    # If no positive examples exist, return empty hypotheses
    if S is None:
        return None, None

    # Initialize G-hypothesis with the most general hypothesis
    G = [["?"] * num_attributes]

    # Process each example in the dataset
    for index, row in training_data.iterrows():
        instance = list(row.iloc[:-1])
        label = row.iloc[-1]

        if label == "Yes":  # Positive example
            for i in range(num_attributes):
                if S[i] != instance[i]:  
                    S[i] = "?"  # Generalize S to be consistent with this positive example
            
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(g[i] == "?" or g[i] == instance[i] for i in range(num_attributes))]

        elif label == "No":  # Negative example
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == "?":
                        for value in set(training_data.iloc[:, i]):  # Consider all values for that attribute
                            if value != instance[i]:  
                                new_hypothesis = g[:]
                                new_hypothesis[i] = value
                                new_G.append(new_hypothesis)

            G = new_G
            G = [g for g in G if all(g[i] == "?" or g[i] != instance[i] for i in range(num_attributes))]

    return S, G


# Sample dataset (Outlook, Temperature, Humidity, Wind, PlayTennis)
data = [
    ["Sunny", "Warm", "Normal", "Strong", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "No"],
    ["Sunny", "Warm", "High", "Weak", "Yes"]
]

# Convert data to DataFrame
columns = ["Outlook", "Temperature", "Humidity", "Wind", "PlayTennis"]
training_data = pd.DataFrame(data, columns=columns)

# Apply Candidate Elimination Algorithm
S_final, G_final = candidate_elimination(training_data)

print("\nMost Specific Hypothesis (S):", S_final)
print("Most General Hypothesis (G):", G_final)

*3.DT ID3 Algorithm*
import pandas as pd
import numpy as np
import math

# Function to calculate entropy
def entropy(data):
    target = data.keys()[-1]  # Get target column
    values = data[target].unique()
    entropy_value = 0

    for value in values:
        fraction = data[target].value_counts()[value] / len(data[target])
        entropy_value -= fraction * math.log2(fraction)

    return entropy_value

# Function to calculate information gain
def info_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy

# Function to find the best attribute
def best_attribute(data):
    attributes = data.keys()[:-1]  # Exclude target column
    gains = {attr: info_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)  # Attribute with highest information gain

# Function to build decision tree
def id3(data, tree=None):
    target = data.keys()[-1]  # Target column

    # If only one class remains in target, return that class
    if len(data[target].unique()) == 1:
        return data[target].unique()[0]

    # If no more attributes, return the most common class
    if len(data.columns) == 1:
        return data[target].mode()[0]

    # Find the best attribute to split on
    best = best_attribute(data)

    # Create tree with best attribute
    if tree is None:
        tree = {}
    tree[best] = {}

    # Split dataset and build subtrees
    for value in data[best].unique():
        subset = data[data[best] == value].drop(columns=[best])  # Remove split attribute
        subtree = id3(subset)
        tree[best][value] = subtree

    return tree

# Sample dataset (PlayTennis)
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy", "Sunny", "Overcast", "Overcast", "Rainy"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Build decision tree using ID3
decision_tree = id3(df)

# Print the decision tree
import pprint
pprint.pprint(decision_tree)

**4.Artificial Neural Network **
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize dataset (X: inputs, y: outputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (XOR problem)
y = np.array([[0], [1], [1], [0]])  # Expected output

# Initialize neural network parameters
input_neurons = 2  # Input layer neurons
hidden_neurons = 3  # Hidden layer neurons
output_neurons = 1  # Output layer neurons

# Randomly initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))

weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
epochs = 10000
learning_rate = 0.5

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute error
    error = y - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = d_predicted_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print error at intervals
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Final output after training
print("\nFinal Output After Training:")
print(predicted_output)


*K-NN Algorithm*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (4 features)
y = iris.target  # Labels (3 classes)

# Select only the first two features for visualization
X = X[:, :2]  # Taking only the first two features

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize KNN classifier with K=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model using only two features
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundaries
def plot_decision_boundary(X, y, model, k):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.title(f"KNN Decision Boundary (K={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(X_train, y_train, knn, k)

*Logistic Regression Algorithm*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only first two features for visualization
y = iris.target  # Labels (0, 1, or 2)

# Convert multi-class to binary classification (only classes 0 and 1 for visualization)
X = X[y < 2]
y = y[y < 2]

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(scatter)
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, log_reg)

*Linear regression Algorithm*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate a synthetic dataset (100 samples, 1 feature, with noise)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = lin_reg.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization of the Regression Line
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.title("Linear Regression Model")
plt.legend()
plt.show()


*Linear and polynomial regression Algorithm*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset (change to real dataset if needed)
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X*3 + 2 * X*2 + 7 * X + 5 + np.random.normal(0, 100, (100, 1))  # Polynomial relation with noise

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- LINEAR REGRESSION ---
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)

# --- POLYNOMIAL REGRESSION (Degree = 3) ---
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_pred_poly = poly_reg.predict(X_poly_test)

# --- MODEL PERFORMANCE ---
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Linear Regression - MSE: {mse_linear:.2f}, R² Score: {r2_linear:.2f}")
print(f"Polynomial Regression (Degree 3) - MSE: {mse_poly:.2f}, R² Score: {r2_poly:.2f}")

# --- VISUALIZATION ---
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred_linear, color='red', linewidth=2, label="Linear Regression")

# For smooth polynomial curve
X_curve = np.linspace(0, 10, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)
y_curve_poly = poly_reg.predict(X_curve_poly)

plt.plot(X_curve, y_curve_poly, color='green', linewidth=2, label="Polynomial Regression (Degree 3)")

plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

*Estimation and Maximation Algorithm*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic dataset
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model (GMM) using Expectation-Maximization
gmm = GaussianMixture(n_components=3, max_iter=100, random_state=42)
gmm.fit(X)

# Predict cluster assignments
labels = gmm.predict(X)

# Get cluster centers
centers = gmm.means_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label="Centers")
plt.title("Gaussian Mixture Model (EM Algorithm)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Print GMM parameters
print("Estimated Means:\n", gmm.means_)
print("Estimated Variances:\n", gmm.covariances_)
print("Estimated Weights:\n", gmm.weights_)

****

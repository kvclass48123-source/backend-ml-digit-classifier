import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

def train_model():
    print("Loading MNIST dataset... (this might take a moment)")
    # Fetching MNIST 784 (28x28 images)
    # Using a subset for faster training in this environment
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    
    # Scale data to [0, 1]
    X = X / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training MLP (Multi-layer Perceptron) Classifier...")
    # Multi-layer Perceptron: good for simple image classification
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                        learning_rate_init=.1)
    
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()

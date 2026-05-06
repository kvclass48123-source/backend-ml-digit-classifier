"""
Run this LOCALLY before deploying:
    python model_train.py

This trains the model and saves model.pkl.
Commit model.pkl to your repo so Render just loads it (no training needed).
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')

    X = X / 255.0

    # ✅ Use only 20,000 samples to keep model.pkl small and training fast
    X, _, y, _ = train_test_split(X, y, train_size=20000, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model (lean MLP for fast load on Render)...")
    # ✅ Smaller network: fewer layers, fewer neurons = less RAM on deploy
    model = MLPClassifier(
        hidden_layer_sizes=(64,),   # was (128, 64) — halved
        max_iter=15,
        alpha=1e-4,
        solver='adam',
        verbose=True,
        random_state=1,
        learning_rate_init=0.001,
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, 'model.pkl', compress=3)  # ✅ compress=3 shrinks file size
    print("Saved model.pkl — commit this file before deploying!")

if __name__ == "__main__":
    train_model()

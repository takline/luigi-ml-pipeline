import pickle

# Path to the trained model file
model_path = "outcomes/trained_model.pkl"

# Load the trained model from the file
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Predict using the model with a sample input
print(model.predict([[0.0, 0.0, 1.0, 44.0, 72000.0]]))

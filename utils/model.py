import pickle
from sklearn import svm
import os


def execute_model_training(input_features, target_values, saving_path=""):
    """
    Train a Support Vector Machine (SVM) model and save it.

    Parameters:
        x (array-like): Input features for training.
        y (array-like): Target labels for training.
        save_path (str): Path to save the trained model.

    The function trains an SVM model using the provided data and saves it to a file.
    """
    classifier = svm.SVC(kernel="rbf")
    classifier.fit(input_features, target_values)

    model_filename = os.path.join(saving_path, "trained_model.pkl")
    with open(model_filename, "wb") as file:
        pickle.dump(classifier, file)

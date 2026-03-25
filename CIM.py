"""
EECS 658 - Assignment 5

Brief description:
    This program evaluates a Neural Network (MLPClassifier) on an imbalanced Iris dataset
    using 2-fold cross-validation. It reports accuracy and balanced accuracy metrics,
    then compares multiple oversampling and undersampling techniques from the imbalanced-learn toolbox.

Inputs:
    - imbalanced iris.csv (located in the same folder as this script)

Outputs:
    - Before each part, prints the part number
    - Part 1:
        * Confusion Matrix
        * Accuracy Score
        * Class Balanced Accuracy (custom, based on per-class recall average)
        * Balanced Accuracy (custom, one-vs-rest accuracy averaged across classes)
        * sklearn balanced_accuracy_score
    - Part 2 (Oversampling):
        * Random oversampling Confusion Matrix + Accuracy
        * SMOTE Confusion Matrix + Accuracy
        * ADASYN Confusion Matrix + Accuracy (sampling_strategy='minority')
    - Part 3 (Undersampling):
        * Random undersampling Confusion Matrix + Accuracy
        * Cluster undersampling Confusion Matrix + Accuracy
        * Tomek links undersampling Confusion Matrix + Accuracy

Collaborators:
    None

Other sources:
    - Assignment 5 Instructions (Canvas)
    - Rubric 5.docx (Canvas)
    - ChatGPT

Author:
    Abhiroop Goel

Creation date:
    2026-03-24
"""

# -----------------------------
# Imports
# -----------------------------

# Standard library import for reading CSV files.
import csv

# Import NumPy for arrays and numerical operations.
import numpy as np

# Import sklearn tools for encoding labels and evaluation.
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

# Import model selection for building the 2 folds.
from sklearn.model_selection import train_test_split

# Import the neural network classifier (required model).
from sklearn.neural_network import MLPClassifier

# imbalanced-learn oversamplers (Part 2).
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# imbalanced-learn undersamplers (Part 3).
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks


# -----------------------------
# Data loading
# -----------------------------

def load_imbalanced_iris_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the imbalanced iris CSV file.

    Assumption (common format):
        - First row is a header
        - Last column is the class label
        - All other columns are numeric features

    Returns:
        X: (N, 4) float features
        y: (N,) integer encoded class labels
    """
    # Create lists to collect feature rows and string labels.
    X_rows = []
    y_labels = []

    # Open the CSV file for reading.
    with open(path, "r", newline="", encoding="utf-8") as f:
        # Create a CSV reader.
        reader = csv.reader(f)

        # Read the header line and ignore it.
        header = next(reader, None)

        # Read each data row.
        for row in reader:
            # Skip empty lines if any exist.
            if not row:
                continue

            # Convert feature columns to floats (all but last column).
            features = [float(v) for v in row[:-1]]

            # Read the label string (last column).
            label = row[-1].strip()

            # Store them.
            X_rows.append(features)
            y_labels.append(label)

    # Convert to NumPy array for ML.
    X = np.array(X_rows, dtype=float)

    # Encode string labels to integers 0..K-1.
    le = LabelEncoder()
    y = le.fit_transform(np.array(y_labels))

    return X, y


# -----------------------------
# Metrics (Part 1 custom metrics)
# -----------------------------

def accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
    """Compute accuracy from confusion matrix."""
    total = float(np.sum(cm))
    correct = float(np.trace(cm))
    return correct / total if total > 0 else 0.0


def class_balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    """
    Class Balanced Accuracy (custom):
        Average of per-class recall values.
        recall_i = TP_i / (TP_i + FN_i) = cm[i,i] / sum(row i)
    """
    recalls = []
    for i in range(cm.shape[0]):
        row_sum = float(np.sum(cm[i, :]))
        tp = float(cm[i, i])
        recall = tp / row_sum if row_sum > 0 else 0.0
        recalls.append(recall)
    return float(np.mean(recalls))


def balanced_accuracy_one_vs_rest_from_cm(cm: np.ndarray) -> float:
    """
    Balanced Accuracy (custom, one-vs-rest):
        For each class i, treat it as positive and all others as negative:
            class_acc_i = (TP_i + TN_i) / Total
        Then average class_acc_i across classes.
    This gives a different "balanced" measure than average recall.
    """
    total = float(np.sum(cm))
    per_class_acc = []

    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = total - tp - fn - fp
        acc_i = (tp + tn) / total if total > 0 else 0.0
        per_class_acc.append(acc_i)

    return float(np.mean(per_class_acc))


# -----------------------------
# 2-fold cross validation runner
# -----------------------------

def two_fold_cv_nn(X: np.ndarray,
                   y: np.ndarray,
                   resampler=None,
                   label: str = "") -> tuple[np.ndarray, float, float, float, float]:
    """
    Runs 2-fold cross-validation using MLPClassifier.
    If resampler is provided, it is applied ONLY to the training fold each time.

    Returns:
        cm: confusion matrix over all N samples (combined fold predictions)
        acc: accuracy computed from cm
        cba: class balanced accuracy (average recall) computed from cm
        ba_custom: one-vs-rest balanced accuracy computed from cm
        ba_sklearn: balanced_accuracy_score from sklearn (average recall)
    """
    # Create two folds (50/50 split) with stratification to keep class ratios similar.
    # random_state is set for reproducibility of the split (not changing model parameters).
    X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )

    # Train NN on fold1 (optionally resample fold1) and test on fold2.
    X_train_1 = X_fold1
    y_train_1 = y_fold1

    # Apply resampling only on training data if resampler is given.
    if resampler is not None:
        X_train_1, y_train_1 = resampler.fit_resample(X_train_1, y_train_1)

    # Create NN model with default parameters (per assignment).
    nn1 = MLPClassifier()

    # Fit model.
    nn1.fit(X_train_1, y_train_1)

    # Predict on fold2 test set (never resampled).
    pred_fold2 = nn1.predict(X_fold2)

    # Train NN on fold2 (optionally resample fold2) and test on fold1.
    X_train_2 = X_fold2
    y_train_2 = y_fold2

    # Apply resampling only on training data if resampler is given.
    if resampler is not None:
        X_train_2, y_train_2 = resampler.fit_resample(X_train_2, y_train_2)

    # Create NN model with default parameters again (fresh model).
    nn2 = MLPClassifier()

    # Fit model.
    nn2.fit(X_train_2, y_train_2)

    # Predict on fold1 test set (never resampled).
    pred_fold1 = nn2.predict(X_fold1)

    # Combine actual labels in the same order as predictions.
    y_true = np.concatenate([y_fold2, y_fold1])

    # Combine predictions so total predictions cover the whole dataset.
    y_pred = np.concatenate([pred_fold2, pred_fold1])

    # Create confusion matrix.
    cm = confusion_matrix(y_true, y_pred)

    # Compute accuracy from confusion matrix (rubric wants them to match).
    acc = accuracy_from_confusion_matrix(cm)

    # Compute class balanced accuracy (average recall) from confusion matrix.
    cba = class_balanced_accuracy_from_cm(cm)

    # Compute custom balanced accuracy (one-vs-rest averaged) from confusion matrix.
    ba_custom = balanced_accuracy_one_vs_rest_from_cm(cm)

    # Compute sklearn balanced accuracy (average recall).
    ba_sklearn = balanced_accuracy_score(y_true, y_pred)

    return cm, acc, cba, ba_custom, ba_sklearn


# -----------------------------
# Printing helpers
# -----------------------------

def print_part_header(part_num: int) -> None:
    """Print part number before each part output."""
    print("\n============================================")
    print(f"Part {part_num}")
    print("============================================")


def print_cm_and_acc(title: str, cm: np.ndarray, acc: float) -> None:
    """Print confusion matrix and accuracy with labels."""
    print(title)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy Score:", round(acc, 3))
    print("")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    # Load imbalanced iris dataset from CSV file.
    X, y = load_imbalanced_iris_csv("imbalanced iris.csv")

    # -----------------------------
    # Part 1: Imbalanced dataset metrics
    # -----------------------------
    print_part_header(1)

    # Run 2-fold CV with NN on imbalanced dataset (no resampling).
    cm1, acc1, cba1, ba1_custom, ba1_sklearn = two_fold_cv_nn(X, y, resampler=None)

    # Print confusion matrix and accuracy.
    print_cm_and_acc("Part 1 (Imbalanced Data Set)", cm1, acc1)

    # Print class balanced accuracy (custom).
    print("Part 1 Class Balanced Accuracy (custom):", round(cba1, 3))

    # Print balanced accuracy (custom one-vs-rest averaged).
    print("Part 1 Balanced Accuracy (custom one-vs-rest):", round(ba1_custom, 3))

    # Print sklearn balanced accuracy.
    print("Part 1 balanced_accuracy_score (sklearn):", round(ba1_sklearn, 3))

    # -----------------------------
    # Part 2: Oversampling
    # -----------------------------
    print_part_header(2)

    # Random oversampling.
    ros = RandomOverSampler()
    cm2a, acc2a, _, _, _ = two_fold_cv_nn(X, y, resampler=ros)
    print_cm_and_acc("Part 2 Random Oversampling (RandomOverSampler)", cm2a, acc2a)

    # SMOTE oversampling.
    smote = SMOTE()
    cm2b, acc2b, _, _, _ = two_fold_cv_nn(X, y, resampler=smote)
    print_cm_and_acc("Part 2 SMOTE Oversampling (SMOTE)", cm2b, acc2b)

    # ADASYN oversampling (must use sampling_strategy='minority' per instructions).
    adasyn = ADASYN(sampling_strategy="minority")
    cm2c, acc2c, _, _, _ = two_fold_cv_nn(X, y, resampler=adasyn)
    print_cm_and_acc("Part 2 ADASYN Oversampling (ADASYN, sampling_strategy='minority')", cm2c, acc2c)

    # -----------------------------
    # Part 3: Undersampling
    # -----------------------------
    print_part_header(3)

    # Random undersampling.
    rus = RandomUnderSampler()
    cm3a, acc3a, _, _, _ = two_fold_cv_nn(X, y, resampler=rus)
    print_cm_and_acc("Part 3 Random Undersampling (RandomUnderSampler)", cm3a, acc3a)

    # Cluster undersampling (ClusterCentroids).
    cc = ClusterCentroids()
    cm3b, acc3b, _, _, _ = two_fold_cv_nn(X, y, resampler=cc)
    print_cm_and_acc("Part 3 Cluster Undersampling (ClusterCentroids)", cm3b, acc3b)

    # Tomek links undersampling (may still be imbalanced, which is allowed).
    tomek = TomekLinks()
    cm3c, acc3c, _, _, _ = two_fold_cv_nn(X, y, resampler=tomek)
    print_cm_and_acc("Part 3 Tomek Links Undersampling (TomekLinks)", cm3c, acc3c)


if __name__ == "__main__":
    main()

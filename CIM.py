"""
EECS 658 - Assignment 5

Brief description:
    Evaluates a Neural Network (MLPClassifier) on an imbalanced Iris dataset
    using 2-fold cross-validation. Reports accuracy and balanced accuracy metrics,
    then compares oversampling and undersampling methods from imbalanced-learn.

Inputs:
    - imbalanced iris.csv (must be in the same folder)

Outputs:
    - Prints Part number before each part
    - Part 1:
        Confusion Matrix, Accuracy,
        Class Balanced Accuracy (custom average recall),
        Balanced Accuracy (custom one-vs-rest average accuracy),
        sklearn balanced_accuracy_score
    - Part 2:
        Random oversampling, SMOTE, ADASYN (sampling_strategy="minority")
        Confusion Matrix + Accuracy
    - Part 3:
        Random undersampling, ClusterCentroids, TomekLinks
        Confusion Matrix + Accuracy

Collaborators:
    None

Other sources:
    ChatGPT

Author:
    Abhiroop Goel

Creation date:
    2026-03-24
"""

# Import csv for reading the dataset file.
import csv

# Import numpy for arrays and math operations.
import numpy as np

# Import LabelEncoder to convert string labels into integers.
from sklearn.preprocessing import LabelEncoder

# Import confusion matrix and sklearn balanced accuracy.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

# Import train_test_split to create the two folds (2-fold CV).
from sklearn.model_selection import train_test_split

# Import the required Neural Network model.
from sklearn.neural_network import MLPClassifier

# Oversampling tools (Part 2).
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# Undersampling tools (Part 3).
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks


def load_imbalanced_iris_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads 'imbalanced iris.csv'.

    Assumptions:
      - First row is a header
      - Last column is the class label
      - Other columns are numeric features

    Returns:
      X as float numpy array
      y as int numpy array (encoded labels)
    """
    X_rows = []
    y_labels = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Skip header
        _ = next(reader, None)

        for row in reader:
            if not row:
                continue

            features = [float(v) for v in row[:-1]]
            label = row[-1].strip()

            X_rows.append(features)
            y_labels.append(label)

    X = np.array(X_rows, dtype=float)

    le = LabelEncoder()
    y = le.fit_transform(np.array(y_labels))

    return X, y


def accuracy_from_cm(cm: np.ndarray) -> float:
    """Accuracy = trace(cm) / sum(cm)."""
    total = float(np.sum(cm))
    correct = float(np.trace(cm))
    return correct / total if total > 0 else 0.0


def class_balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    """
    Class Balanced Accuracy (from lecture):
      average of per-class recall
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
    Balanced Accuracy (custom one-vs-rest):
      For each class i:
        class_acc_i = (TP_i + TN_i) / Total
      Then average across classes.
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


def two_fold_cv_nn(X: np.ndarray, y: np.ndarray, resampler=None) -> tuple[np.ndarray, float, float, float, float]:
    """
    2-fold CV using MLPClassifier with DEFAULT parameters.

    Important:
      - If resampler is provided, it is applied ONLY to the training fold each time.
      - Split uses train_test_split with default behavior (no random_state, no stratify).
    """
    # 2-fold split: fold1 and fold2 (50/50) using default splitting behavior.
    X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(X, y, test_size=0.5)

    # Fold1 -> Fold2
    X_train_1 = X_fold1
    y_train_1 = y_fold1

    if resampler is not None:
        X_train_1, y_train_1 = resampler.fit_resample(X_train_1, y_train_1)

    nn1 = MLPClassifier()  # default params
    nn1.fit(X_train_1, y_train_1)
    pred_fold2 = nn1.predict(X_fold2)

    # Fold2 -> Fold1
    X_train_2 = X_fold2
    y_train_2 = y_fold2

    if resampler is not None:
        X_train_2, y_train_2 = resampler.fit_resample(X_train_2, y_train_2)

    nn2 = MLPClassifier()  # default params
    nn2.fit(X_train_2, y_train_2)
    pred_fold1 = nn2.predict(X_fold1)

    # Combine to cover all samples.
    y_true = np.concatenate([y_fold2, y_fold1])
    y_pred = np.concatenate([pred_fold2, pred_fold1])

    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_from_cm(cm)
    cba = class_balanced_accuracy_from_cm(cm)
    ba_custom = balanced_accuracy_one_vs_rest_from_cm(cm)
    ba_sklearn = balanced_accuracy_score(y_true, y_pred)

    return cm, acc, cba, ba_custom, ba_sklearn


def print_part_header(part_num: int) -> None:
    """Print part number before each section."""
    print("\n============================================")
    print(f"Part {part_num}")
    print("============================================")


def print_cm_and_acc(title: str, cm: np.ndarray, acc: float) -> None:
    """Print labeled confusion matrix and accuracy."""
    print(title)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy Score:", round(acc, 3))
    print("")


def main() -> None:
    # Load dataset.
    X, y = load_imbalanced_iris_csv("imbalanced iris.csv")

    # Part 1
    print_part_header(1)
    cm1, acc1, cba1, ba1_custom, ba1_sklearn = two_fold_cv_nn(X, y, resampler=None)
    print_cm_and_acc("Part 1 Imbalanced Data Set", cm1, acc1)
    print("Part 1 Class Balanced Accuracy (custom):", round(cba1, 3))
    print("Part 1 Balanced Accuracy (custom one-vs-rest):", round(ba1_custom, 3))
    print("Part 1 balanced_accuracy_score (sklearn):", round(ba1_sklearn, 3))

    # Part 2 Oversampling
    print_part_header(2)

    ros = RandomOverSampler()  # default params
    cm2a, acc2a, _, _, _ = two_fold_cv_nn(X, y, resampler=ros)
    print_cm_and_acc("Part 2 Random Oversampling (RandomOverSampler)", cm2a, acc2a)

    smote = SMOTE()  # default params
    cm2b, acc2b, _, _, _ = two_fold_cv_nn(X, y, resampler=smote)
    print_cm_and_acc("Part 2 SMOTE Oversampling (SMOTE)", cm2b, acc2b)

    # ADASYN requires sampling_strategy="minority" per the assignment instructions.
    adasyn = ADASYN(sampling_strategy="minority")
    cm2c, acc2c, _, _, _ = two_fold_cv_nn(X, y, resampler=adasyn)
    print_cm_and_acc("Part 2 ADASYN Oversampling (ADASYN, sampling_strategy='minority')", cm2c, acc2c)

    # Part 3 Undersampling
    print_part_header(3)

    rus = RandomUnderSampler()  # default params
    cm3a, acc3a, _, _, _ = two_fold_cv_nn(X, y, resampler=rus)
    print_cm_and_acc("Part 3 Random Undersampling (RandomUnderSampler)", cm3a, acc3a)

    cc = ClusterCentroids()  # default params
    cm3b, acc3b, _, _, _ = two_fold_cv_nn(X, y, resampler=cc)
    print_cm_and_acc("Part 3 Cluster Undersampling (ClusterCentroids)", cm3b, acc3b)

    tomek = TomekLinks()  # default params
    cm3c, acc3c, _, _, _ = two_fold_cv_nn(X, y, resampler=tomek)
    print_cm_and_acc("Part 3 Tomek Links Undersampling (TomekLinks)", cm3c, acc3c)


if __name__ == "__main__":
    main()
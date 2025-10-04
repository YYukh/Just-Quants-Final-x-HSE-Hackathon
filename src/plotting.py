import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_classification(model, X, y):
    """
    Plot PR and ROC curves for binary or multiclass classification.

    Args:
        model: fitted classifier with predict_proba
        X: features
        y: true labels
    """
    y_pred_proba = model.predict_proba(X)
    classes = model.classes_
    n_classes = len(classes)

    # Handle binary vs multiclass
    if n_classes == 2:
        # Binary case: use probabilities of the positive class (index 1)
        y_test_bin = (y == classes[1]).astype(int)  # 1 for positive class, 0 otherwise
        y_score = y_pred_proba[:, 1]  # probability of positive class

        precision, recall, _ = precision_recall_curve(y_test_bin, y_score)
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        auc_pr = auc(recall, precision)
        auc_roc = auc(fpr, tpr)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].plot(recall, precision, lw=2, label=f"Class {classes[1]} (AUC={auc_pr:.3f})")
        axes[1].plot(fpr, tpr, lw=2, label=f"Class {classes[1]} (AUC={auc_roc:.3f})")

        axes[0].set_title("Precision-Recall Curve (Binary)")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].legend(loc="lower left")

        axes[1].set_title("ROC Curve (Binary)")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")

    else:
        # Multiclass case: one-vs-rest
        y_test_bin = label_binarize(y, classes=classes)
        precision = dict()
        recall = dict()
        fpr = dict()
        tpr = dict()
        auc_pr_vals = []
        auc_roc_vals = []

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for i, class_ in enumerate(classes):
            precision[class_], recall[class_], _ = precision_recall_curve(
                y_test_bin[:, i], y_pred_proba[:, i]
            )
            fpr[class_], tpr[class_], _ = roc_curve(
                y_test_bin[:, i], y_pred_proba[:, i]
            )
            auc_pr_vals.append(auc(recall[class_], precision[class_]))
            auc_roc_vals.append(auc(fpr[class_], tpr[class_]))

            axes[0].plot(recall[class_], precision[class_], lw=2, label=f"Class {class_}")
            axes[1].plot(fpr[class_], tpr[class_], lw=2, label=f"Class {class_}")

        axes[0].set_title(f"AUC-PR per class: {', '.join([f'{v:.3f}' for v in auc_pr_vals])}")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].legend(loc="lower left")

        axes[1].set_title(f"AUC-ROC per class: {', '.join([f'{v:.3f}' for v in auc_roc_vals])}")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()
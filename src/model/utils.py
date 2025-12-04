import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchmetrics import ConfusionMatrix

from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig

import torch
import inspect
from src.model.config import ModelConfig


def get_most_optimal_device():
    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if cuda_available:
        device = torch.device("cuda")
        accelerator = "gpu"
        devices = 1
    elif mps_available:
        device = torch.device("mps")
        accelerator = "mps"
        devices = 1
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
        devices = None

    pin_memory = True if cuda_available else False
    return device, accelerator, devices, pin_memory


def load_model(checkpoint_path: str | Path) -> CNNBiLSTMModel:
    """
    Load trained model from checkpoint.
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    hparams = checkpoint.get('hyper_parameters', {})
    valid_params = set(inspect.signature(ModelConfig.__init__).parameters.keys()) - {'self'}
    filtered_hparams = {k: v for k, v in hparams.items() if k in valid_params}
    
    config = ModelConfig(**filtered_hparams)
    model = CNNBiLSTMModel(config)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    return model


def predict_sample(
    model: CNNBiLSTMModel,
    X: np.ndarray,
    return_bins: list[float] | None = None
) -> dict:
    """
    Make prediction for a single sample.
    """
    if return_bins is None:
        return_bins = [-np.inf, -6.11, -3.81, -2.22, -1.05, -0.20, 0.64, 
                       1.42, 2.25, 3.31, 4.60, 6.71, np.inf]
    
    if X.ndim == 2:
        X = X[np.newaxis, :]
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    
    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()  # type: ignore
    
    # Interpret prediction
    bin_start = return_bins[pred_class]  # type: ignore
    bin_end = return_bins[pred_class + 1]  # type: ignore
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probabilities": probs[0].numpy(),
        "bin_range": (bin_start, bin_end),
        "interpretation": f"Expected return: {bin_start:.2f}% to {bin_end:.2f}%"
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_bins: list[float] | None = None,
    save_path: str | Path | None = None
):
    """
    Plot confusion matrix.
    """
    if return_bins is None:
        return_bins = [-np.inf, -6.11, -3.81, -2.22, -1.05, -0.20, 0.64, 
                       1.42, 2.25, 3.31, 4.60, 6.71, np.inf]
    
    # Create labels
    labels = []
    for i in range(len(return_bins) - 1):
        start = return_bins[i]
        end = return_bins[i + 1]
        if np.isinf(start):
            labels.append(f"< {end:.1f}%")
        elif np.isinf(end):
            labels.append(f"> {start:.1f}%")
        else:
            labels.append(f"{start:.1f}% to {end:.1f}%")
    
    cm_metric = ConfusionMatrix(task="multiclass", num_classes=len(labels))
    cm = cm_metric(torch.tensor(y_pred), torch.tensor(y_true)).numpy()
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix - Stock Return Prediction", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_bins: list[float] | None = None,
    save_path: str | Path | None = None
):
    """
    Plot distribution of predictions vs ground truth.
    """
    if return_bins is None:
        return_bins = [-np.inf, 0, np.inf]
    
    # Count distributions
    true_counts = np.bincount(y_true, minlength=len(return_bins) - 1)
    pred_counts = np.bincount(y_pred, minlength=len(return_bins) - 1)
    
    # Create labels
    labels = [f"Bin {i}" for i in range(len(return_bins) - 1)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, true_counts, width, label="Ground Truth", alpha=0.8)
    ax.bar(x + width/2, pred_counts, width, label="Predictions", alpha=0.8)
    
    ax.set_xlabel("Return Bin", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution: Predictions vs Ground Truth", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(
    model: CNNBiLSTMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    return_bins: list[float] | None = None,
    save_dir: str | Path | None = None
) -> dict:
    """
    Comprehensive model evaluation.
    """
    if return_bins is None:
        return_bins = [-np.inf, 0, np.inf]
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_test)
    
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_indices = np.argmax(y_test, axis=1)
    else:
        y_indices = y_test.flatten()
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).numpy()
    
    # Calculate metrics
    accuracy = np.mean(preds == y_indices)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(return_bins) - 1):
        mask = y_indices == i
        if mask.sum() > 0:
            class_acc = np.mean(preds[mask] == i)
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Plot confusion matrix
    if save_dir:
        plot_confusion_matrix(
            y_indices,
            preds,
            return_bins,
            save_dir / "confusion_matrix.png"
        )
        plot_prediction_distribution(
            y_indices,
            preds,
            return_bins,
            save_dir / "prediction_distribution.png"
        )
    
    results = {
        "overall_accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "predictions": preds,
        "probabilities": probs.numpy()
    }
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f}")
    print("="*80 + "\n")
    
    return results

"""
Example script: Load and use pre-trained model for inference.

This script demonstrates how to:
1. Load the production-ready v2.0 model
2. Make predictions on new data
3. Interpret the results for trading decisions

Author: Miguel Noriega Bedolla
Date: December 2025
"""

import torch
import numpy as np
from pathlib import Path
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig


def load_pretrained_model(checkpoint_path: str = "pretrained_models/best_model_v2.0.ckpt"):
    """
    Load the pre-trained v2.0 model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Loaded model in evaluation mode
        config: Model configuration
    """
    print("=" * 80)
    print("LOADING PRE-TRAINED MODEL v2.0")
    print("=" * 80)
    
    # Load configuration
    config = ModelConfig()
    print(f"\nConfiguration:")
    print(f"  Input shape: (batch, {config.sequence_length}, {config.n_features})")
    print(f"  Output classes: {config.n_classes} (Bajista, Alcista)")
    print(f"  Model parameters: ~7.7M")
    
    # Load model weights (PyTorch 2.6+ compatibility)
    # Lightning's load_from_checkpoint uses weights_only=True by default in PyTorch 2.6
    # We need to load manually with weights_only=False for backward compatibility
    import torch
    
    # Load checkpoint with weights_only=False
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config from checkpoint
    config = ModelConfig()
    
    # Create model and load state dict
    model = CNNBiLSTMModel(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"\nModel loaded successfully from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Test accuracy: 79.5%")
    
    return model, config


def predict_single_sample(model, X_sample: np.ndarray, sample_name: str = "Sample"):
    """
    Make prediction on a single sample.
    
    Args:
        model: Loaded model
        X_sample: Input data of shape (30, 50)
        sample_name: Name for logging
    """
    # Convert to tensor and add batch dimension
    X_tensor = torch.from_numpy(X_sample).float().unsqueeze(0)  # (1, 30, 50)
    
    # Make prediction
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Extract probabilities
    bajista_prob = probabilities[0, 0].item()
    alcista_prob = probabilities[0, 1].item()
    
    # Interpret result
    print(f"\n{sample_name}:")
    print(f"  Bajista (Sell): {bajista_prob:6.2%}")
    print(f"  Alcista (Buy):  {alcista_prob:6.2%}")
    print(f"  Prediction: {'SELL' if predicted_class == 0 else 'BUY'} signal")
    print(f"  Confidence: {max(bajista_prob, alcista_prob):.2%}")
    
    return predicted_class, bajista_prob, alcista_prob


def predict_batch(model, X_batch: np.ndarray):
    """
    Make predictions on a batch of samples.
    
    Args:
        model: Loaded model
        X_batch: Input data of shape (batch_size, 30, 50)
        
    Returns:
        predictions: Class predictions (0=Bajista, 1=Alcista)
        probabilities: Class probabilities
    """
    # Convert to tensor
    X_tensor = torch.from_numpy(X_batch).float()
    
    # Make predictions
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions.numpy(), probabilities.numpy()


def evaluate_on_test_set(model, test_X_path: str = "datasets/npy/test_X.npy",
                         test_y_path: str = "datasets/npy/test_y.npy"):
    """
    Evaluate model on the official test set.
    
    Args:
        model: Loaded model
        test_X_path: Path to test features
        test_y_path: Path to test labels
    """
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    
    # Load test data
    X_test = np.load(test_X_path)
    y_test = np.load(test_y_path)
    
    print(f"\nTest set size: {X_test.shape[0]} samples")
    print(f"Input shape: {X_test.shape}")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).float()
    
    # Make predictions
    with torch.no_grad():
        logits = model(X_tensor)
        predictions = torch.argmax(logits, dim=1)
        y_true = torch.argmax(y_tensor, dim=1)
    
    # Calculate accuracy
    correct = (predictions == y_true).sum().item()
    accuracy = correct / len(predictions)
    
    print(f"\nResults:")
    print(f"  Correct predictions: {correct}/{len(predictions)}")
    print(f"  Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Class distribution
    print(f"\nPrediction distribution:")
    for i in range(2):
        count = (predictions == i).sum().item()
        pct = count / len(predictions) * 100
        class_name = "Bajista" if i == 0 else "Alcista"
        print(f"  {class_name}: {count:3d} ({pct:5.2f}%)")
    
    print(f"\nTrue distribution:")
    for i in range(2):
        count = (y_true == i).sum().item()
        pct = count / len(y_true) * 100
        class_name = "Bajista" if i == 0 else "Alcista"
        print(f"  {class_name}: {count:3d} ({pct:5.2f}%)")
    
    # Confusion matrix
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    for t, p in zip(y_true, predictions):
        confusion[t.long(), p.long()] += 1
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual          Bajista  Alcista")
    print(f"Bajista         {confusion[0,0]:4d}     {confusion[0,1]:4d}")
    print(f"Alcista         {confusion[1,0]:4d}     {confusion[1,1]:4d}")
    
    # Precision and recall
    precision_0 = confusion[0,0] / (confusion[0,0] + confusion[1,0])
    precision_1 = confusion[1,1] / (confusion[0,1] + confusion[1,1])
    recall_0 = confusion[0,0] / (confusion[0,0] + confusion[0,1])
    recall_1 = confusion[1,1] / (confusion[1,0] + confusion[1,1])
    
    print(f"\nPer-Class Metrics:")
    print(f"  Bajista - Precision: {precision_0:.2%}, Recall: {recall_0:.2%}")
    print(f"  Alcista - Precision: {precision_1:.2%}, Recall: {recall_1:.2%}")


def main():
    """Main demonstration function."""
    
    # 1. Load pre-trained model
    model, config = load_pretrained_model()
    
    # 2. Example: Predict on test set
    if Path("datasets/npy/test_X.npy").exists():
        evaluate_on_test_set(model)
    else:
        print("\nTest set not found. Skipping evaluation.")
    
    # 3. Example: Predict on random samples
    print("\n" + "=" * 80)
    print("EXAMPLE: PREDICTIONS ON RANDOM SAMPLES")
    print("=" * 80)
    
    # Generate random sample (in practice, use real market data)
    print("\nNote: Using random data for demonstration.")
    print("In production, replace with actual preprocessed market data.")
    
    for i in range(3):
        # Random sample: (30 days, 50 features)
        sample = np.random.randn(30, 50).astype(np.float32)
        predict_single_sample(model, sample, f"Random Sample {i+1}")
    
    # 4. Example: Batch prediction
    print("\n" + "=" * 80)
    print("EXAMPLE: BATCH PREDICTION")
    print("=" * 80)
    
    # Random batch of 10 samples
    batch = np.random.randn(10, 30, 50).astype(np.float32)
    predictions, probabilities = predict_batch(model, batch)
    
    print(f"\nBatch size: {len(predictions)}")
    print(f"\nPredictions:")
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        signal = "SELL" if pred == 0 else "BUY"
        confidence = probs[pred]
        print(f"  Sample {i+1}: {signal} (confidence: {confidence:.2%})")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Prepare your own data in the format (batch, 30, 50)")
    print("  2. Ensure features match the training data preprocessing")
    print("  3. Use predict_single_sample() or predict_batch() for inference")
    print("  4. See pretrained_models/README.md for more details")


if __name__ == "__main__":
    main()

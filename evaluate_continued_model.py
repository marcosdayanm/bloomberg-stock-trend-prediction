"""
Evaluate the continued training model on test set and update pretrained_models/ if improved.
"""
import torch
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
from src.model.model import CNNBiLSTMModel
from src.model.config import ModelConfig
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_and_update_pretrained():
    """Evaluate continued model and update pretrained_models/ if improved."""
    
    # Paths
    checkpoint_path = "src/model/checkpoints/continued-best-epoch=epoch=99-accval_acc=0.8567-lossval_loss=0.4391.ckpt"
    pretrained_dir = Path("pretrained_models")
    metadata_path = pretrained_dir / "model_metadata.json"
    
    # Load metadata to get baseline performance
    with open(metadata_path, 'r') as f:
        old_metadata = json.load(f)
    
    baseline_test_acc = old_metadata['performance']['test_accuracy']
    
    print("=" * 80)
    print("EVALUATING CONTINUED TRAINING MODEL")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Baseline test accuracy: {baseline_test_acc:.4f} ({baseline_test_acc*100:.2f}%)")
    print()
    
    # Load model
    config = ModelConfig()
    model = CNNBiLSTMModel.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=torch.device('cpu'),
        weights_only=False  # Required for PyTorch 2.6+
    )
    model.eval()
    
    # Load test data
    X_test = torch.from_numpy(np.load('datasets/npy/test_X.npy')).float()
    y_test = torch.from_numpy(np.load('datasets/npy/test_y.npy')).float()
    
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Evaluate
    with torch.no_grad():
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1).numpy()
        y_true = torch.argmax(y_test, dim=1).numpy()
        
        # Calculate metrics
        test_acc = (preds == y_true).mean()
        f1_weighted = f1_score(y_true, preds, average='weighted')
        f1_macro = f1_score(y_true, preds, average='macro')
        
        print("=" * 80)
        print("TEST SET RESULTS")
        print("=" * 80)
        print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"F1 (weighted):  {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
        print(f"F1 (macro):     {f1_macro:.4f} ({f1_macro*100:.2f}%)")
        print()
        
        # Improvement analysis
        improvement = test_acc - baseline_test_acc
        improvement_pct = (improvement / baseline_test_acc) * 100
        
        print("=" * 80)
        print("IMPROVEMENT ANALYSIS")
        print("=" * 80)
        print(f"Previous best:  {baseline_test_acc:.4f} ({baseline_test_acc*100:.2f}%)")
        print(f"New model:      {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print()
        
        # Classification report
        print("=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        class_names = ['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up']
        # Get actual classes present in the data
        unique_classes = sorted(np.unique(np.concatenate([y_true, preds])))
        labels = list(range(5))  # Always use 0-4 for 5 classes
        print(classification_report(y_true, preds, labels=labels, target_names=class_names, digits=4, zero_division=0))
        
        # Confusion matrix
        print("=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        cm = confusion_matrix(y_true, preds)
        print("Predicted →")
        print("Actual ↓")
        print("         ", "  ".join([f"{i:3d}" for i in range(5)]))
        for i, row in enumerate(cm):
            print(f"Class {i}:", "  ".join([f"{val:3d}" for val in row]))
        print()
        
        # Update pretrained_models if improved
        if test_acc > baseline_test_acc:
            print("=" * 80)
            print("UPDATING PRETRAINED MODEL (IMPROVED!)")
            print("=" * 80)
            
            # Backup old model
            old_model_path = pretrained_dir / "best_model_v2.0.ckpt"
            backup_path = pretrained_dir / f"best_model_v2.0_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt"
            
            if old_model_path.exists():
                print(f"Backing up old model to: {backup_path.name}")
                shutil.copy2(old_model_path, backup_path)
            
            # Copy new model
            new_model_path = pretrained_dir / "best_model_v2.1.ckpt"
            print(f"Copying new model to: {new_model_path}")
            shutil.copy2(checkpoint_path, new_model_path)
            
            # Update metadata
            new_metadata = old_metadata.copy()
            new_metadata['version'] = 'v2.1'
            new_metadata['performance']['test_accuracy'] = float(test_acc)
            new_metadata['performance']['val_accuracy'] = 0.8567  # From checkpoint name
            new_metadata['performance']['f1_score'] = float(f1_weighted)
            new_metadata['training']['total_epochs'] = 143 + 100  # Original + continued
            new_metadata['training']['continued_from'] = 'v2.0'
            new_metadata['training']['fine_tuning_epochs'] = 100
            new_metadata['training']['fine_tuning_lr'] = 0.0001
            new_metadata['last_updated'] = datetime.now().isoformat()
            
            # Save updated metadata
            metadata_v21_path = pretrained_dir / "model_metadata_v2.1.json"
            with open(metadata_v21_path, 'w') as f:
                json.dump(new_metadata, f, indent=2)
            
            print(f"✓ New model saved: {new_model_path}")
            print(f"✓ Metadata updated: {metadata_v21_path}")
            print(f"✓ Old model backed up: {backup_path.name}")
            print()
            print("=" * 80)
            print(f"SUCCESS! Model improved by {improvement_pct:+.2f}%")
            print("=" * 80)
            
        else:
            print("=" * 80)
            print("NO UPDATE - Model did not improve on test set")
            print("=" * 80)
            print(f"Validation acc improved (80.0% → 85.7%)")
            print(f"But test acc did not improve ({baseline_test_acc:.4f} → {test_acc:.4f})")
            print("This suggests potential overfitting to validation set.")
            print()


if __name__ == "__main__":
    evaluate_and_update_pretrained()

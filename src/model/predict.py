import torch
import numpy as np
from pathlib import Path

from src.model.config import ModelConfig
from src.model.model import CNNBiLSTMModel
from pathlib import PosixPath, WindowsPath
from torch.utils.data import DataLoader, TensorDataset


MODEL = "pretrained_models/best_model_v2.1.ckpt"

X_PATH = 'datasets/npy/test_X.npy' 
Y_PATH = 'datasets/npy/test_y.npy'



def load_custom_data(x_path: str, y_path: str):
    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"\nCustom data loaded:")
    print(f"  X from: {x_path}")
    print(f"  y from: {y_path}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Convertir a tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Crear dataset y dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    return dataloader


def main():
    print("\n" + "="*80)
    print("PREDICT")
    print("="*80 + "\n")

    config = ModelConfig()
    checkpoint_path = Path(MODEL)
    
    print(f"Loading: {checkpoint_path.name}")
    print(f"Task: {config.task_type.upper()}\n")
    
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    
    model = CNNBiLSTMModel.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        strict=False
    )
    model.eval()
    
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)\n")
    

    if X_PATH and Y_PATH:
        local_dataloader = load_custom_data(X_PATH, Y_PATH)
    else:
        from src.model.data_module import StockDataModule
        data_module = StockDataModule(config, pin_memory=False)
        data_module.prepare_data()
        data_module.setup("predict")
        local_dataloader = data_module.predict_dataloader()
    
    n_total = sum(len(batch[0]) for batch in local_dataloader)
    print("="*80)
    print(f"PREDICTIONS ({n_total} samples)")
    print("="*80 + "\n")
    
    # Get device from model
    device = next(model.parameters()).device
    
    sample_num = 1  # Counter outside batch loop
    
    with torch.no_grad():
        total = 0
        good = 0
        for batch in local_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            if config.task_type == 'classification':
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                y_true = torch.argmax(y, dim=1) if y.dim() > 1 else y
                
                class_names = ['DOWN (<0%)', 'UP (>=0%)']
                
                for i in range(len(preds)):
                    pred = preds[i].item()
                    true = y_true[i].item()
                    conf = probs[i, pred].item()
                    
                    result = "✓" if pred == true else "✗"
                    if pred == true:
                        good += 1
                    total += 1
                    
                    print(f"Sample {sample_num}: {result}")
                    sample_num += 1
                    print(f"  Predicted:   {class_names[pred]} - {conf:.1%} confidence")
                    print(f"  Ground Truth: {class_names[true]}")
                    print(f"  Probs: DOWN={probs[i,0]:.1%} | UP={probs[i,1]:.1%}")
                    print()
            
            else:
                # Regression
                outputs = model(x).squeeze(1)
                y_true = y.squeeze(1) if y.dim() > 1 else y
                
                for i in range(len(outputs)):
                    pred = outputs[i].item()
                    true = y_true[i].item()
                    error = abs(pred - true)
                    
                    print(f"Sample {i+1}:")
                    print(f"  Predicted:   {pred:+.2f}%")
                    print(f"  Ground Truth: {true:+.2f}%")
                    print(f"  Error: {error:.2f}%")
                    print()
    
    print("="*80 + "\n")

    if config.task_type == 'classification':
        accuracy = good / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%} ({good}/{total})\n")


if __name__ == "__main__":
    main()

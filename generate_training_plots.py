"""
Generate training history plots from existing TensorBoard logs.

This script reads TensorBoard event files and creates visualizations of:
- Accuracy (train, validation, test)
- Loss (train, validation, test)
- F1 Score (validation, test)
- Learning rate schedule
- Combined overview

Usage:
    python generate_training_plots.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def plot_training_history(log_dir: Path, output_dir: Path):
    """
    Plot training history from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    # Find the latest version directory
    version_dirs = sorted(log_dir.glob("cnn_bilstm/version_*"))
    if not version_dirs:
        print("ERROR: No TensorBoard logs found in src/model/logs/cnn_bilstm/")
        print("   Run training first: uv run python -m src.model.train")
        return
    
    latest_version = version_dirs[-1]
    event_file = list(latest_version.glob("events.out.tfevents.*"))
    
    if not event_file:
        print(f"ERROR: No event files found in {latest_version}")
        return
    
    print(f"\n{'='*80}")
    print(f"GENERATING TRAINING PLOTS")
    print(f"{'='*80}")
    print(f"📂 Reading logs from: {latest_version.name}")
    
    # Load TensorBoard logs
    ea = event_accumulator.EventAccumulator(str(latest_version))
    ea.Reload()
    
    # Extract metrics
    metrics_data = {}
    
    # Available scalar tags
    tags = ea.Tags()['scalars']
    print(f"Found metrics: {', '.join(tags)}")
    
    for tag in tags:
        events = ea.Scalars(tag)
        metrics_data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'epoch': e.step}
            for e in events
        ])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}/")
    
    # Plot 1: Accuracy (Train vs Val vs Test)
    plt.figure(figsize=(14, 7))
    
    if 'train_acc' in metrics_data and 'val_acc' in metrics_data:
        # Get max values for annotation
        max_val_acc = metrics_data['val_acc']['value'].max()
        max_val_epoch = metrics_data['val_acc']['value'].idxmax()
        
        plt.plot(metrics_data['train_acc']['epoch'], 
                metrics_data['train_acc']['value'], 
                label='Train Accuracy', linewidth=2.5, color='#2ecc71', alpha=0.8)
        plt.plot(metrics_data['val_acc']['epoch'], 
                metrics_data['val_acc']['value'], 
                label='Validation Accuracy', linewidth=2.5, color='#3498db', alpha=0.8)
        
        # Mark best validation
        plt.scatter(max_val_epoch, max_val_acc, color='#3498db', s=100, zorder=5, 
                   marker='*', edgecolors='black', linewidths=1.5)
        plt.annotate(f'Best Val: {max_val_acc:.4f}\nEpoch {max_val_epoch}', 
                    xy=(max_val_epoch, max_val_acc), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Add test accuracy as horizontal line if available
        if 'test_acc' in metrics_data:
            test_acc = metrics_data['test_acc']['value'].iloc[0]
            plt.axhline(y=test_acc, color='#e74c3c', linestyle='--', 
                       linewidth=2.5, label=f'Test Accuracy: {test_acc:.4f}', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title('Model Accuracy Over Training Epochs', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_history.png', dpi=200, bbox_inches='tight')
    print(f"  SAVED: accuracy_history.png")
    plt.close()
    
    # Plot 2: Loss (Train vs Val vs Test)
    plt.figure(figsize=(14, 7))
    
    if 'train_loss' in metrics_data and 'val_loss' in metrics_data:
        # Get min values for annotation
        min_val_loss = metrics_data['val_loss']['value'].min()
        min_val_epoch = metrics_data['val_loss']['value'].idxmin()
        
        plt.plot(metrics_data['train_loss']['epoch'], 
                metrics_data['train_loss']['value'], 
                label='Train Loss', linewidth=2.5, color='#2ecc71', alpha=0.8)
        plt.plot(metrics_data['val_loss']['epoch'], 
                metrics_data['val_loss']['value'], 
                label='Validation Loss', linewidth=2.5, color='#3498db', alpha=0.8)
        
        # Mark best validation
        plt.scatter(min_val_epoch, min_val_loss, color='#3498db', s=100, zorder=5,
                   marker='*', edgecolors='black', linewidths=1.5)
        plt.annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch {min_val_epoch}', 
                    xy=(min_val_epoch, min_val_loss), 
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Add test loss as horizontal line if available
        if 'test_loss' in metrics_data:
            test_loss = metrics_data['test_loss']['value'].iloc[0]
            plt.axhline(y=test_loss, color='#e74c3c', linestyle='--', 
                       linewidth=2.5, label=f'Test Loss: {test_loss:.4f}', alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.title('Model Loss Over Training Epochs', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_history.png', dpi=200, bbox_inches='tight')
    print(f"  SAVED: loss_history.png")
    plt.close()
    
    # Plot 3: F1 Score (Val vs Test)
    if 'val_f1' in metrics_data:
        plt.figure(figsize=(14, 7))
        
        max_f1 = metrics_data['val_f1']['value'].max()
        max_f1_epoch = metrics_data['val_f1']['value'].idxmax()
        
        plt.plot(metrics_data['val_f1']['epoch'], 
                metrics_data['val_f1']['value'], 
                label='Validation F1', linewidth=2.5, color='#9b59b6', alpha=0.8)
        
        # Mark best F1
        plt.scatter(max_f1_epoch, max_f1, color='#9b59b6', s=100, zorder=5,
                   marker='*', edgecolors='black', linewidths=1.5)
        plt.annotate(f'Best F1: {max_f1:.4f}\nEpoch {max_f1_epoch}', 
                    xy=(max_f1_epoch, max_f1), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')
        
        # Add test F1 as horizontal line if available
        if 'test_f1' in metrics_data:
            test_f1 = metrics_data['test_f1']['value'].iloc[0]
            plt.axhline(y=test_f1, color='#e74c3c', linestyle='--', 
                       linewidth=2.5, label=f'Test F1: {test_f1:.4f}', alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=13, fontweight='bold')
        plt.ylabel('F1 Score', fontsize=13, fontweight='bold')
        plt.title('F1 Score Over Training Epochs', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_dir / 'f1_history.png', dpi=200, bbox_inches='tight')
        print(f"  SAVED: f1_history.png")
        plt.close()
    
    # Plot 4: Combined Overview (2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Training History Overview', fontsize=18, fontweight='bold', y=0.995)
    
    # Subplot 1: Accuracy
    if 'train_acc' in metrics_data and 'val_acc' in metrics_data:
        axes[0, 0].plot(metrics_data['train_acc']['epoch'], 
                       metrics_data['train_acc']['value'], 
                       label='Train', linewidth=2.5, color='#2ecc71', alpha=0.8)
        axes[0, 0].plot(metrics_data['val_acc']['epoch'], 
                       metrics_data['val_acc']['value'], 
                       label='Validation', linewidth=2.5, color='#3498db', alpha=0.8)
        if 'test_acc' in metrics_data:
            test_acc = metrics_data['test_acc']['value'].iloc[0]
            axes[0, 0].axhline(y=test_acc, color='#e74c3c', linestyle='--', 
                             linewidth=2.5, label=f'Test: {test_acc:.4f}', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Accuracy', fontsize=13, fontweight='bold', pad=10)
        axes[0, 0].legend(fontsize=10, framealpha=0.9)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 2: Loss
    if 'train_loss' in metrics_data and 'val_loss' in metrics_data:
        axes[0, 1].plot(metrics_data['train_loss']['epoch'], 
                       metrics_data['train_loss']['value'], 
                       label='Train', linewidth=2.5, color='#2ecc71', alpha=0.8)
        axes[0, 1].plot(metrics_data['val_loss']['epoch'], 
                       metrics_data['val_loss']['value'], 
                       label='Validation', linewidth=2.5, color='#3498db', alpha=0.8)
        if 'test_loss' in metrics_data:
            test_loss = metrics_data['test_loss']['value'].iloc[0]
            axes[0, 1].axhline(y=test_loss, color='#e74c3c', linestyle='--', 
                             linewidth=2.5, label=f'Test: {test_loss:.4f}', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Loss', fontsize=13, fontweight='bold', pad=10)
        axes[0, 1].legend(fontsize=10, framealpha=0.9)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 3: F1 Score
    if 'val_f1' in metrics_data:
        axes[1, 0].plot(metrics_data['val_f1']['epoch'], 
                       metrics_data['val_f1']['value'], 
                       label='Validation F1', linewidth=2.5, color='#9b59b6', alpha=0.8)
        if 'test_f1' in metrics_data:
            test_f1 = metrics_data['test_f1']['value'].iloc[0]
            axes[1, 0].axhline(y=test_f1, color='#e74c3c', linestyle='--', 
                             linewidth=2.5, label=f'Test F1: {test_f1:.4f}', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('F1 Score', fontsize=13, fontweight='bold', pad=10)
        axes[1, 0].legend(fontsize=10, framealpha=0.9)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 4: Learning Rate
    if 'lr-AdamW' in metrics_data:
        axes[1, 1].plot(metrics_data['lr-AdamW']['epoch'], 
                       metrics_data['lr-AdamW']['value'], 
                       label='Learning Rate', linewidth=2.5, color='#f39c12', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Learning Rate Schedule (OneCycleLR)', fontsize=13, fontweight='bold', pad=10)
        axes[1, 1].legend(fontsize=10, framealpha=0.9)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=200, bbox_inches='tight')
    print(f"  SAVED: training_overview.png")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    if 'train_acc' in metrics_data and 'val_acc' in metrics_data:
        final_train_acc = metrics_data['train_acc']['value'].iloc[-1]
        final_val_acc = metrics_data['val_acc']['value'].iloc[-1]
        best_val_acc = metrics_data['val_acc']['value'].max()
        best_epoch = metrics_data['val_acc']['value'].idxmax()
        
        print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"Final Train Accuracy: {final_train_acc:.4f}")
        print(f"Final Val Accuracy: {final_val_acc:.4f}")
        
        if 'test_acc' in metrics_data:
            test_acc = metrics_data['test_acc']['value'].iloc[0]
            gap = abs(best_val_acc - test_acc)
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Val-Test Gap: {gap:.4f} ({gap*100:.2f}%)")
    
    if 'train_loss' in metrics_data and 'val_loss' in metrics_data:
        final_train_loss = metrics_data['train_loss']['value'].iloc[-1]
        final_val_loss = metrics_data['val_loss']['value'].iloc[-1]
        best_val_loss = metrics_data['val_loss']['value'].min()
        
        print(f"\nBest Validation Loss: {best_val_loss:.4f}")
        print(f"Final Train Loss: {final_train_loss:.4f}")
        print(f"Final Val Loss: {final_val_loss:.4f}")
    
    if 'test_f1' in metrics_data:
        test_f1 = metrics_data['test_f1']['value'].iloc[0]
        print(f"\nTest F1 Score: {test_f1:.4f}")
    
    print(f"\n{'='*80}")
    print(f"SUCCESS: All plots saved successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Default paths
    log_dir = Path("src/model/logs")
    output_dir = Path("src/model/checkpoints/training_plots")
    
    plot_training_history(log_dir, output_dir)

"""
Script para generar visualizaciones de análisis de features y su importancia.
Genera gráficas de separabilidad de clases, distribuciones y ranking de features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_feature_data():
    """Cargar datos de features y sus índices seleccionados."""
    # Cargar features seleccionadas
    selected_indices = np.load('datasets/npy/selected_features_indices.npy')
    
    # Cargar datos de entrenamiento
    X_train = np.load('datasets/npy/train_X.npy')  # (N, 30, 50)
    y_train = np.load('datasets/npy/train_y.npy')  # (N, 2)
    
    # Cargar nombres de features desde archivo de texto
    feature_names_file = 'datasets/npy/msft_5day_prediction_features.txt'
    with open(feature_names_file, 'r') as f:
        all_feature_names = [line.strip() for line in f.readlines()]
    
    selected_feature_names = [all_feature_names[i] for i in selected_indices]
    
    return X_train, y_train, selected_indices, selected_feature_names, all_feature_names


def calculate_class_separability(X_train, y_train):
    """
    Calcula la separabilidad entre clases para cada feature.
    Métrica: |mean(clase_0) - mean(clase_1)| / (std(clase_0) + std(clase_1))
    """
    # Convertir y_train de one-hot a labels
    y_labels = np.argmax(y_train, axis=1)
    
    # Separar por clase
    X_class_0 = X_train[y_labels == 0]  # Bajista
    X_class_1 = X_train[y_labels == 1]  # Alcista
    
    # Promediar sobre timesteps (30 días) para cada muestra
    X_class_0_avg = X_class_0.mean(axis=1)  # (N_class0, 50)
    X_class_1_avg = X_class_1.mean(axis=1)  # (N_class1, 50)
    
    # Calcular estadísticas por feature
    mean_0 = X_class_0_avg.mean(axis=0)
    mean_1 = X_class_1_avg.mean(axis=0)
    std_0 = X_class_0_avg.std(axis=0)
    std_1 = X_class_1_avg.std(axis=0)
    
    # Separabilidad: diferencia de medias normalizada por suma de desviaciones
    separability = np.abs(mean_0 - mean_1) / (std_0 + std_1 + 1e-8)
    
    return separability, mean_0, mean_1, std_0, std_1


def plot_feature_ranking(separability, feature_names, output_dir):
    """Gráfica de ranking de features por separabilidad."""
    # Ordenar features por separabilidad
    sorted_indices = np.argsort(separability)[::-1]
    top_20_indices = sorted_indices[:20]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_20_indices))
    separability_values = separability[top_20_indices]
    feature_labels = [feature_names[i] for i in top_20_indices]
    
    # Acortar nombres muy largos
    feature_labels = [label[:40] + '...' if len(label) > 40 else label 
                     for label in feature_labels]
    
    colors = plt.cm.viridis(separability_values / separability_values.max())
    
    ax.barh(y_pos, separability_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Class Separability Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features por Separabilidad de Clases\n(Bajista vs Alcista)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Añadir valores en las barras
    for i, v in enumerate(separability_values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_ranking_top20.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_dir / 'feature_ranking_top20.png'}")
    plt.close()


def plot_feature_distributions(X_train, y_train, feature_names, output_dir):
    """Distribuciones de las top 6 features más discriminativas."""
    # Calcular separabilidad
    separability, mean_0, mean_1, std_0, std_1 = calculate_class_separability(X_train, y_train)
    
    # Top 6 features
    top_6_indices = np.argsort(separability)[::-1][:6]
    
    y_labels = np.argmax(y_train, axis=1)
    X_avg = X_train.mean(axis=1)  # Promediar sobre timesteps
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_6_indices):
        ax = axes[idx]
        
        # Datos de cada clase
        class_0_data = X_avg[y_labels == 0, feat_idx]
        class_1_data = X_avg[y_labels == 1, feat_idx]
        
        # Histogramas superpuestos
        ax.hist(class_0_data, bins=30, alpha=0.6, label='Bajista', color='red', edgecolor='black')
        ax.hist(class_1_data, bins=30, alpha=0.6, label='Alcista', color='green', edgecolor='black')
        
        # Líneas de media
        ax.axvline(mean_0[feat_idx], color='darkred', linestyle='--', linewidth=2, label=f'μ₀={mean_0[feat_idx]:.2f}')
        ax.axvline(mean_1[feat_idx], color='darkgreen', linestyle='--', linewidth=2, label=f'μ₁={mean_1[feat_idx]:.2f}')
        
        # Etiquetas
        feature_label = feature_names[feat_idx][:35] + '...' if len(feature_names[feat_idx]) > 35 else feature_names[feat_idx]
        ax.set_title(f'{feature_label}\nSeparabilidad: {separability[feat_idx]:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Valor normalizado', fontsize=9)
        ax.set_ylabel('Frecuencia', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle('Distribución de Top 6 Features más Discriminativas', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions_top6.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_dir / 'feature_distributions_top6.png'}")
    plt.close()


def plot_separability_distribution(separability, output_dir):
    """Distribución general de separabilidad de todas las features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma
    ax1.hist(separability, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(separability.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Media: {separability.mean():.3f}')
    ax1.axvline(np.median(separability), color='orange', linestyle='--', linewidth=2, 
               label=f'Mediana: {np.median(separability):.3f}')
    ax1.set_xlabel('Separabilidad', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax1.set_title('Distribución de Separabilidad\n(50 Features Seleccionadas)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Boxplot
    ax2.boxplot(separability, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black', linewidth=1.5),
               capprops=dict(color='black', linewidth=1.5))
    ax2.set_ylabel('Separabilidad', fontsize=12, fontweight='bold')
    ax2.set_title('Boxplot de Separabilidad', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Estadísticas
    stats_text = f"Min: {separability.min():.3f}\nQ1: {np.percentile(separability, 25):.3f}\n"
    stats_text += f"Median: {np.median(separability):.3f}\nQ3: {np.percentile(separability, 75):.3f}\n"
    stats_text += f"Max: {separability.max():.3f}"
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'separability_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_dir / 'separability_distribution.png'}")
    plt.close()


def plot_feature_importance_heatmap(X_train, y_train, feature_names, output_dir):
    """Heatmap de correlación entre top features."""
    # Calcular separabilidad para obtener top features
    separability, _, _, _, _ = calculate_class_separability(X_train, y_train)
    top_15_indices = np.argsort(separability)[::-1][:15]
    
    # Promediar sobre timesteps
    X_avg = X_train.mean(axis=1)
    X_top = X_avg[:, top_15_indices]
    
    # Calcular correlación
    correlation_matrix = np.corrcoef(X_top.T)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    feature_labels = [feature_names[i][:30] + '...' if len(feature_names[i]) > 30 else feature_names[i] 
                     for i in top_15_indices]
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Ticks
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_yticks(np.arange(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(feature_labels, fontsize=9)
    
    # Añadir valores de correlación
    for i in range(len(feature_labels)):
        for j in range(len(feature_labels)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Matriz de Correlación: Top 15 Features', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlación de Pearson', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_dir / 'feature_correlation_heatmap.png'}")
    plt.close()


def generate_feature_statistics_table(separability, feature_names, output_dir):
    """Generar tabla CSV con estadísticas de features."""
    df = pd.DataFrame({
        'Feature': feature_names,
        'Separability_Score': separability,
        'Rank': np.arange(1, len(separability) + 1)
    })
    
    df = df.sort_values('Separability_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = np.arange(1, len(df) + 1)
    
    output_file = output_dir / 'feature_statistics.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Guardado: {output_file}")
    
    return df


def main():
    print("="*80)
    print("GENERACIÓN DE ANÁLISIS DE FEATURES")
    print("="*80)
    
    # Crear directorio de salida
    output_dir = Path('src/model/checkpoints/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print("\n1. Cargando datos...")
    X_train, y_train, selected_indices, selected_feature_names, all_feature_names = load_feature_data()
    print(f"   - Train samples: {X_train.shape[0]}")
    print(f"   - Features seleccionadas: {len(selected_feature_names)}")
    print(f"   - Features originales: {len(all_feature_names)}")
    
    # Calcular separabilidad
    print("\n2. Calculando separabilidad de clases...")
    separability, mean_0, mean_1, std_0, std_1 = calculate_class_separability(X_train, y_train)
    print(f"   - Separabilidad media: {separability.mean():.4f}")
    print(f"   - Separabilidad max: {separability.max():.4f}")
    print(f"   - Separabilidad min: {separability.min():.4f}")
    
    # Generar visualizaciones
    print("\n3. Generando visualizaciones...")
    plot_feature_ranking(separability, selected_feature_names, output_dir)
    plot_feature_distributions(X_train, y_train, selected_feature_names, output_dir)
    plot_separability_distribution(separability, output_dir)
    plot_feature_importance_heatmap(X_train, y_train, selected_feature_names, output_dir)
    
    # Generar tabla de estadísticas
    print("\n4. Generando tabla de estadísticas...")
    df_stats = generate_feature_statistics_table(separability, selected_feature_names, output_dir)
    
    print("\n" + "="*80)
    print("TOP 10 FEATURES MÁS DISCRIMINATIVAS:")
    print("="*80)
    print(df_stats.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"📁 Resultados guardados en: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

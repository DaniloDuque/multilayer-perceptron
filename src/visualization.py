"""
visualization.py — biblioteca de visualización para TP3.

Expone funciones puras que el notebook orquestador llama directamente.
No contiene lógica de carga de archivos ni main().
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 100, 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

_COLORS_CLASSES = {0: '#FF6B6B', 1: '#4ECDC4'}
_COLORS_CONFIGS = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01', 'D': '#C73E1D'}


## @brief Grafica scatter de los datasets separable y no-separable.
#
#  @param X_sep    Tensor/array (n, 2) del dataset separable.
#  @param T_sep    Tensor/array (n, 1) de etiquetas separable.
#  @param X_nosep  Tensor/array (n, 2) del dataset no-separable.
#  @param T_nosep  Tensor/array (n, 1) de etiquetas no-separable.
#  @param output_path Ruta de salida (None = no guardar).
#  @return Figure de matplotlib.
def plot_datasets(X_sep, T_sep, X_nosep, T_nosep, output_path=None):
    def _to_np(t):
        return t.numpy() if hasattr(t, 'numpy') else np.asarray(t)

    X_sep   = _to_np(X_sep);   T_sep   = _to_np(T_sep).ravel()
    X_nosep = _to_np(X_nosep); T_nosep = _to_np(T_nosep).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X, T, title in [
        (axes[0], X_sep,   T_sep,   'Separable (std=1.0)'),
        (axes[1], X_nosep, T_nosep, 'No-separable (std=4.0)'),
    ]:
        for cls in [0, 1]:
            mask = T == cls
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=_COLORS_CLASSES[cls], label=f'Clase {cls}',
                       s=20, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


## @brief Genera matriz 2×4 con evolución de error (train/val) por configuración.
#
#  @param results_sep   dict {'A':..,'B':..,'C':..,'D':..} de train_all_configs (separable).
#  @param results_nosep dict igual para no-separable.
#  @param output_path   Ruta de salida (None = no guardar).
#  @return Figure de matplotlib.
def plot_error_matrix(results_sep, results_nosep, output_path=None):
    configs = ['A', 'B', 'C', 'D']
    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    for row, (res, label) in enumerate([(results_sep, 'Separable'),
                                         (results_nosep, 'No-Separable')]):
        for col, cfg in enumerate(configs):
            ax = fig.add_subplot(gs[row, col])
            d  = res[cfg]
            epochs = np.arange(1, len(d['train_errors']) + 1)
            ax.plot(epochs, d['train_errors'], label='Train',
                    color=_COLORS_CONFIGS[cfg], lw=2)
            ax.plot(epochs, d['val_errors'],   label='Val',
                    color=_COLORS_CONFIGS[cfg], lw=2, linestyle='--')
            M, g = d['params']['M'], d['params']['gamma']
            ax.set_title(f'Config {cfg}: M={M}, γ={g:.1f}', fontweight='bold', fontsize=10)
            ax.set_xlabel('Época'); ax.set_ylabel('MSE')
            ax.grid(True, linestyle='--', alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
            if col == 0:
                ax.text(-0.35, 0.5, label, transform=ax.transAxes,
                        fontsize=11, fontweight='bold', rotation=90,
                        ha='right', va='center')

    fig.suptitle('Evolución del Error MSE vs Épocas', fontsize=13, fontweight='bold')
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


## @brief Imprime tabla de convergencia final (error MSE de validación).
#
#  @param results_sep   dict de resultados separable.
#  @param results_nosep dict de resultados no-separable.
def print_convergence_table(results_sep, results_nosep):
    configs = ['A', 'B', 'C', 'D']
    print(f"\n{'Config':<10} {'Separable (val)':>18} {'No-Separable (val)':>20}")
    print('-' * 52)
    for cfg in configs:
        sep   = results_sep[cfg]['final_val_error']
        nosep = results_nosep[cfg]['final_val_error']
        print(f"Config {cfg}   {sep:>18.6f} {nosep:>20.6f}")


## @brief Devuelve un DataFrame con los errores finales de convergencia.
#
#  @param results_sep   dict de resultados separable.
#  @param results_nosep dict de resultados no-separable.
#  @return pandas.DataFrame con columnas config, sep_train, sep_val, nosep_train, nosep_val.
def convergence_dataframe(results_sep, results_nosep):
    rows = []
    for cfg in ['A', 'B', 'C', 'D']:
        rows.append({
            'config':      cfg,
            'sep_train':   results_sep[cfg]['final_train_error'],
            'sep_val':     results_sep[cfg]['final_val_error'],
            'nosep_train': results_nosep[cfg]['final_train_error'],
            'nosep_val':   results_nosep[cfg]['final_val_error'],
        })
    return pd.DataFrame(rows)

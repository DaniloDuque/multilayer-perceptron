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
from matplotlib.colors import ListedColormap

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
        if hasattr(t, 'cpu'):
            return t.cpu().numpy()
        return np.asarray(t)

    X_sep   = _to_np(X_sep);   T_sep   = _to_np(T_sep).ravel()
    X_nosep = _to_np(X_nosep); T_nosep = _to_np(T_nosep).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X, T, title in [
        (axes[0], X_sep,   T_sep,   'Separable'),
        (axes[1], X_nosep, T_nosep, 'No-separable'),
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

## @brief Grafica la superficie de decisión de un MLP entrenado sobre Two Moons.
#
#  Recibe los tensores ya generados por generate_moons_data y el modelo entrenado.
#  Construye una malla sobre [0,1]² (o el rango real de X) y colorea cada región
#  según la clase predicha, sobreponiendo los puntos de train y validación.
#
#  @param model       Instancia de MultilayerPerceptron ya entrenada.
#  @param X_train     Tensor (n_train, 2) — conjunto de entrenamiento.
#  @param T_train     Tensor (n_train, 1) — etiquetas de entrenamiento.
#  @param X_val       Tensor (n_val, 2)   — conjunto de validación.
#  @param T_val       Tensor (n_val, 1)   — etiquetas de validación.
#  @param title       Título del gráfico.
#  @param h           Resolución de la malla (paso entre puntos).
#  @param output_path Ruta de salida (None = no guardar).
#  @return Figure de matplotlib.
def plot_decision_surface_moons(model, X_train, T_train, X_val, T_val,
                                title='Superficie de decisión — Two Moons',
                                h=0.005, output_path=None):

    def _to_np(t):
        if hasattr(t, 'cpu'):
            return t.cpu().numpy()
        return np.asarray(t)

    X_tr = _to_np(X_train); T_tr = _to_np(T_train).ravel()
    X_vl = _to_np(X_val);   T_vl = _to_np(T_val).ravel()

    # Rango de la malla a partir de los datos reales
    X_all = np.vstack([X_tr, X_vl])
    x_min, x_max = X_all[:, 0].min() - 0.05, X_all[:, 0].max() + 0.05
    y_min, y_max = X_all[:, 1].min() - 0.05, X_all[:, 1].max() + 0.05

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    device = next(iter(model.parameters())).device if hasattr(model, 'parameters') else None
    grid   = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    if device is not None:
        grid = grid.to(device)

    Z = model.predict(grid).cpu().numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Superficie de decisión
    ax.contourf(xx, yy, Z,
                cmap=ListedColormap(['#FFAAAA', '#AAAAFF']),
                alpha=0.55)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.2, linestyles='--')

    # Puntos de entrenamiento
    for cls, color, marker in [(0, _COLORS_CLASSES[0], 'o'),
                                (1, _COLORS_CLASSES[1], 'o')]:
        mask = T_tr == cls
        ax.scatter(X_tr[mask, 0], X_tr[mask, 1],
                   c=color, marker=marker, s=22, alpha=0.8,
                   edgecolors='k', linewidth=0.4,
                   label=f'Train — Clase {cls}')

    # Puntos de validación (rombo, borde más grueso para distinguirlos)
    for cls, color in [(0, _COLORS_CLASSES[0]), (1, _COLORS_CLASSES[1])]:
        mask = T_vl == cls
        ax.scatter(X_vl[mask, 0], X_vl[mask, 1],
                   c=color, marker='D', s=28, alpha=0.9,
                   edgecolors='k', linewidth=0.8,
                   label=f'Val   — Clase {cls}')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(loc='upper right', framealpha=0.85)

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

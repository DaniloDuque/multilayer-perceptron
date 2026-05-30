"""visualization.py

Carga los resultados generados por `data_classifier.py` y crea las gráficas
requeridas:

- Scatter de los datos (separable / no-separable)
- Matriz de gráficas (2 filas × 4 columnas) con evolución de error vs época
- Tabla (CSV) con el error final de convergencia por configuración

El script asume que existe `resultados_clasificacion.pkl` en el mismo
directorio (creado por `data_classifier.py`).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_results(path="resultados_clasificacion.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de resultados: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_datasets(results, out_dir="."):
    """Grafica scatter de los datasets (train + val) para separable y no-separable.
    Los arrays en `results` contienen una columna de bias en la posición 0,
    por lo que usamos las columnas 1 y 2 como las dos dimensiones reales.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(axes, ["separable", "no_separable"], ["Separable (std=1.0)", "No-separable (std=4.0)"]):
        datos = results[key]['datos']
        # concatenar train + val para mostrar todos los puntos
        X_full = np.vstack([datos['X_train'].numpy(), datos['X_val'].numpy()])
        T_full = np.vstack([datos['T_train'].numpy(), datos['T_val'].numpy()])

        X = X_full[:, 1:3]  # columnas 1 y 2 son las features (col 0 = bias)

        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=T_full.ravel(), palette=["tab:blue", "tab:orange"], ax=ax, s=20)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(title="Clase")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "datasets_scatter.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def plot_error_matrix(results, out_dir="."):
    """Crea una matriz 2x4 con la evolución de error (train y val) por
    configuración (A,B,C,D) y por dataset.
    """
    os.makedirs(out_dir, exist_ok=True)

    configs = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

    for row_idx, key in enumerate(["separable", "no_separable"]):
        for col_idx, cfg in enumerate(configs):
            ax = axes[row_idx, col_idx]
            cfg_res = results[key]['configuraciones'][cfg]
            train_errors = np.array(cfg_res['train_errors'])
            val_errors = np.array(cfg_res['val_errors'])

            ax.plot(train_errors, label='Train', color='tab:blue')
            ax.plot(val_errors, label='Val', color='tab:orange')
            ax.set_title(f"Dataset: {key} — Config {cfg}")
            ax.set_xlabel('Época')
            ax.set_ylabel('MSE')
            ax.grid(True, linestyle='--', alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "error_matrix.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def save_convergence_table(results, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    configs = ['A', 'B', 'C', 'D']

    rows = []
    for cfg in configs:
        row = {
            'config': cfg,
            'separable_final_val': results['separable']['configuraciones'][cfg]['final_val_error'],
            'separable_final_train': results['separable']['configuraciones'][cfg]['final_train_error'],
            'no_separable_final_val': results['no_separable']['configuraciones'][cfg]['final_val_error'],
            'no_separable_final_train': results['no_separable']['configuraciones'][cfg]['final_train_error'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, 'convergence_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df)


def main():
    results_file = os.path.join(os.path.dirname(__file__), 'resultados_clasificacion.pkl')
    results = load_results(results_file)

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    plot_datasets(results, out_dir=out_dir)
    plot_error_matrix(results, out_dir=out_dir)
    save_convergence_table(results, out_dir=out_dir)


if __name__ == '__main__':
    main()
"""
Visualización de Resultados - Clasificador Neural Network (TP3)

Este módulo carga los resultados de entrenamiento guardados por data_classifier.py
y genera tres gráficas principales:

1. Datos Originales: 2 subplots mostrando datos separables vs no-separables
2. Evolución de Errores: Matriz 2×4 con curvas de entrenamiento/validación
3. Tabla de Convergencia: Error final de validación por configuración

Autor: TP3 - Visualización
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


# ============================================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================================

# Usar estilo limpio de seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# Paleta de colores para las clases
COLORS_CLASSES = {0: '#FF6B6B', 1: '#4ECDC4'}  # Rojo para clase 0, Turquesa para clase 1

# Colores para configuraciones
COLORS_CONFIGS = {
    'A': '#2E86AB',  # Azul
    'B': '#A23B72',  # Púrpura
    'C': '#F18F01',  # Naranja
    'D': '#C73E1D'   # Rojo oscuro
}


# ============================================================================
# CARGAR RESULTADOS
# ============================================================================

def cargar_resultados(archivo='resultados_clasificacion.pkl'):
    """
    Carga los resultados guardados por data_classifier.py.
    
    Parameters
    ----------
    archivo : str
        Ruta del archivo pickle con los resultados. Default: 'resultados_clasificacion.pkl'
        
    Returns
    -------
    dict
        Diccionario con estructura:
        {
            'separable': {
                'datos': {...preds train/val...},
                'configuraciones': {
                    'A': {'train_errors': [...], 'val_errors': [...], ...},
                    'B': {...}, 'C': {...}, 'D': {...}
                }
            },
            'no_separable': {...similar...}
        }
    """
    print(f"[Cargando] Leyendo resultados desde {archivo}...")
    with open(archivo, 'rb') as f:
        resultados = pickle.load(f)
    print("[Cargando] ✓ Resultados cargados exitosamente\n")
    return resultados


# ============================================================================
# GRÁFICA 1: DATOS ORIGINALES
# ============================================================================

def graficar_datos_originales(resultados, output_file='figura_1_datos_originales.png'):
    """
    Genera gráfica con 2 subplots: datos separables y no-separables.
    
    Cada subplot muestra:
    - Puntos coloreados por clase (rojo=clase 0, turquesa=clase 1)
    - Parámetros de generación en el título
    - Feature 1 (eje X), Feature 2 (eje Y)
    - Los datos mostrados son del dataset completo (train + validation)
    
    Parameters
    ----------
    resultados : dict
        Diccionario de resultados cargado por cargar_resultados()
    output_file : str
        Nombre del archivo PNG de salida. Default: 'figura_1_datos_originales.png'
    """
    
    print("[Gráfica 1] Generando visualización de datos originales...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Obtener datos separables
    datos_sep = resultados['separable']['datos']
    X_train_sep = datos_sep['X_train'].numpy()
    X_val_sep = datos_sep['X_val'].numpy()
    T_train_sep = datos_sep['T_train'].numpy().ravel()
    T_val_sep = datos_sep['T_val'].numpy().ravel()
    
    # Datos combinados (train + val) para la gráfica
    X_sep_combined = np.vstack([X_train_sep, X_val_sep])
    T_sep_combined = np.hstack([T_train_sep, T_val_sep])
    
    # Obtener datos no-separables
    datos_no_sep = resultados['no_separable']['datos']
    X_train_no_sep = datos_no_sep['X_train'].numpy()
    X_val_no_sep = datos_no_sep['X_val'].numpy()
    T_train_no_sep = datos_no_sep['T_train'].numpy().ravel()
    T_val_no_sep = datos_no_sep['T_val'].numpy().ravel()
    
    # Datos combinados (train + val) para la gráfica
    X_no_sep_combined = np.vstack([X_train_no_sep, X_val_no_sep])
    T_no_sep_combined = np.hstack([T_train_no_sep, T_val_no_sep])
    
    # ========== SUBPLOT 1: DATOS SEPARABLES ==========
    ax = axes[0]
    for class_label in [0, 1]:
        mask = T_sep_combined == class_label
        ax.scatter(X_sep_combined[mask, 1], X_sep_combined[mask, 2],
                  c=COLORS_CLASSES[class_label], label=f'Clase {int(class_label)}',
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Feature 1 ($x_1$)', fontweight='bold')
    ax.set_ylabel('Feature 2 ($x_2$)', fontweight='bold')
    ax.set_title('Datos Linealmente Separables\n(std=1.0, n=500)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 2: DATOS NO-SEPARABLES ==========
    ax = axes[1]
    for class_label in [0, 1]:
        mask = T_no_sep_combined == class_label
        ax.scatter(X_no_sep_combined[mask, 1], X_no_sep_combined[mask, 2],
                  c=COLORS_CLASSES[class_label], label=f'Clase {int(class_label)}',
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Feature 1 ($x_1$)', fontweight='bold')
    ax.set_ylabel('Feature 2 ($x_2$)', fontweight='bold')
    ax.set_title('Datos No-Linealmente Separables\n(std=4.0, n=500)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Generación de Datos - Conjuntos de Entrenamiento y Validación', 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Gráfica 1] ✓ Guardada en {output_file}\n")
    
    return fig


# ============================================================================
# GRÁFICA 2: EVOLUCIÓN DE ERRORES (MATRIZ 2×4)
# ============================================================================

def graficar_evolucion_errores(resultados, output_file='figura_2_evolucion_errores.png'):
    """
    Genera matriz 2×4 de subplots mostrando la evolución del error vs épocas.
    
    Filas: datasets (separable, no-separable)
    Columnas: configuraciones (A, B, C, D)
    
    En cada subplot se grafican 2 curvas: error de train (línea sólida)
    y error de validación (línea punteada).
    
    Parameters
    ----------
    resultados : dict
        Diccionario de resultados cargado por cargar_resultados()
    output_file : str
        Nombre del archivo PNG de salida. Default: 'figura_2_evolucion_errores.png'
    """
    
    print("[Gráfica 2] Generando matriz de evolución de errores (2×4)...")
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    datasets = ['separable', 'no_separable']
    configs = ['A', 'B', 'C', 'D']
    dataset_labels = ['Separable', 'No-Separable']
    
    # Crear matriz de subplots
    for row_idx, (dataset_name, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        for col_idx, config_letter in enumerate(configs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Obtener datos de configuración
            config_data = resultados[dataset_name]['configuraciones'][config_letter]
            train_errors = config_data['train_errors']
            val_errors = config_data['val_errors']
            epochs = np.arange(1, len(train_errors) + 1)
            
            # Obtener parámetros para el título
            params = config_data['params']
            M = params['M']
            gamma = params['gamma']
            
            # Graficar curvas
            ax.plot(epochs, train_errors, linewidth=2, label='Train', 
                   color=COLORS_CONFIGS[config_letter], linestyle='-', marker='o', markersize=3)
            ax.plot(epochs, val_errors, linewidth=2, label='Validación', 
                   color=COLORS_CONFIGS[config_letter], linestyle='--', marker='s', markersize=3)
            
            # Formato
            ax.set_xlabel('Época', fontweight='bold')
            ax.set_ylabel('MSE', fontweight='bold')
            ax.set_title(f'Config {config_letter}: M={M}, γ={gamma:.1f}', 
                        fontweight='bold', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 51])
            
            # Si es la primera columna, agregar label de dataset a la izquierda
            if col_idx == 0:
                ax.text(-0.35, 0.5, dataset_label, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       ha='right', va='center')
    
    plt.suptitle('Evolución del Error MSE vs Épocas - Todas las Configuraciones',
                 fontsize=13, fontweight='bold', y=0.995)
    
    # Guardar figura
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Gráfica 2] ✓ Guardada en {output_file}\n")
    
    return fig


# ============================================================================
# GRÁFICA 3: TABLA DE CONVERGENCIA FINAL
# ============================================================================

def graficar_tabla_convergencia(resultados, output_file='figura_3_tabla_convergencia.png'):
    """
    Genera tabla visual de errores finales por configuración y dataset.
    
    Muestra error MSE final de validación como métrica principal.
    El error de training también se incluye entre paréntesis para referencia.
    
    Formato: matriz de heatmap donde:
    - Filas: Configuraciones (A, B, C, D)
    - Columnas: Datasets (Separable, No-Separable)
    - Celdas: Error final de validación con código de colores
    
    Parameters
    ----------
    resultados : dict
        Diccionario de resultados cargado por cargar_resultados()
    output_file : str
        Nombre del archivo PNG de salida. Default: 'figura_3_tabla_convergencia.png'
    """
    
    print("[Gráfica 3] Generando tabla de convergencia final...")
    
    configs = ['A', 'B', 'C', 'D']
    datasets = ['separable', 'no_separable']
    dataset_labels = ['Separable', 'No-Separable']
    
    # Compilar errores finales en matriz
    error_vals = np.zeros((len(configs), len(datasets)))
    error_trains = np.zeros((len(configs), len(datasets)))
    
    for i, config in enumerate(configs):
        for j, dataset in enumerate(datasets):
            config_data = resultados[dataset]['configuraciones'][config]
            error_vals[i, j] = config_data['final_val_error']
            error_trains[i, j] = config_data['final_train_error']
    
    # Crear figura para la tabla
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para la tabla con etiquetas
    table_data = []
    for i, config in enumerate(configs):
        row = [f'Config {config}']
        for j in range(len(datasets)):
            val_error = error_vals[i, j]
            train_error = error_trains[i, j]
            cell_text = f'{val_error:.6f}\n({train_error:.6f})'
            row.append(cell_text)
        table_data.append(row)
    
    # Crear tabla
    col_labels = ['Configuración'] + dataset_labels
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.35, 0.35])
    
    # Formatear tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Colorear celdas según valor del error (heatmap)
    for i in range(len(configs)):
        for j in range(len(datasets)):
            cell = table[(i+1, j+1)]  # +1 porque la primera fila son labels
            error_val = error_vals[i, j]
            
            # Normalizar a colores: rojo (error alto) a verde (error bajo)
            # Rango aproximado de errores
            min_error = error_vals.min() * 0.95
            max_error = error_vals.max() * 1.05
            
            # Normalizar entre 0 y 1
            norm_val = (error_val - min_error) / (max_error - min_error + 1e-10)
            norm_val = np.clip(norm_val, 0, 1)
            
            # Colormap: rojo (alto) a verde (bajo)
            # Usamos interpolación entre rojo y verde
            if norm_val > 0.5:
                # Rojo (0.5 a 1.0)
                color = (1.0, 1 - 2*(norm_val - 0.5), 1 - 2*(norm_val - 0.5))
            else:
                # Amarillo a rojo (0 a 0.5)
                color = (1.0, 0.75 * (1 - norm_val), 0)
            
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='black')
    
    # Formatear header
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Titulo y subtítulo
    title = 'Tabla de Convergencia Final - Error MSE en Validación'
    subtitle = '(Error de Training between paréntesis)'
    
    plt.figtext(0.5, 0.95, title, ha='center', fontsize=12, fontweight='bold')
    plt.figtext(0.5, 0.90, subtitle, ha='center', fontsize=9, style='italic', color='gray')
    
    # Guardar figura
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Gráfica 3] ✓ Guardada en {output_file}\n")
    
    return fig


# ============================================================================
# ESTADÍSTICAS RESUMEN
# ============================================================================

def imprimir_estadisticas(resultados):
    """
    Imprime estadísticas resumen sobre los entrenamientos.
    
    Incluye:
    - Error mínimo/máximo por dataset
    - Configuración ganadora (menor error final en validación)
    - Diferencia train/val (indicator de overfitting)
    
    Parameters
    ----------
    resultados : dict
        Diccionario de resultados cargado por cargar_resultados()
    """
    
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS RESUMEN")
    print("=" * 80)
    
    for dataset_name in ['separable', 'no_separable']:
        print(f"\n--- Dataset: {dataset_name.upper()} ---")
        
        configs_data = resultados[dataset_name]['configuraciones']
        
        errors_val = {cfg: configs_data[cfg]['final_val_error'] for cfg in configs_data}
        errors_train = {cfg: configs_data[cfg]['final_train_error'] for cfg in configs_data}
        
        # Encontrar mejor configuración
        best_config = min(errors_val, key=errors_val.get)
        best_error = errors_val[best_config]
        
        # Encontrar peor configuración
        worst_config = max(errors_val, key=errors_val.get)
        worst_error = errors_val[worst_config]
        
        print(f"  Error de Validación Mínimo:  Config {best_config} → {best_error:.6f}")
        print(f"  Error de Validación Máximo:  Config {worst_config} → {worst_error:.6f}")
        print(f"  Rango de errores: {(worst_error - best_error):.6f}")
        
        # Analizar overfitting
        print(f"\n  Análisis de Overfitting (Train - Val):")
        for cfg in sorted(errors_val.keys()):
            diff = errors_train[cfg] - errors_val[cfg]
            status = "✓ Sin overfitting" if diff >= -0.01 else "✗ Leve overfitting"
            print(f"    Config {cfg}: {diff:+.6f} {status}")
    
    print("\n" + "=" * 80 + "\n")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que orquesta todo el flujo de visualización.
    """
    
    print("=" * 80)
    print("VISUALIZACIÓN DE RESULTADOS - CLASIFICADOR NEURAL NETWORK (TP3)")
    print("=" * 80 + "\n")
    
    # Cargar resultados
    try:
        resultados = cargar_resultados('resultados_clasificacion.pkl')
    except FileNotFoundError:
        print("ERROR: No se encontró archivo 'resultados_clasificacion.pkl'")
        print("Asegúrate de haber ejecutado primero: python data_classifier.py")
        return
    
    # Generar gráficas
    print("Generando visualizaciones...\n")
    
    fig1 = graficar_datos_originales(resultados)
    fig2 = graficar_evolucion_errores(resultados)
    fig3 = graficar_tabla_convergencia(resultados)
    
    # Mostrar estadísticas
    imprimir_estadisticas(resultados)
    
    print("=" * 80)
    print("✓ VISUALIZACIÓN COMPLETADA")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  1. figura_1_datos_originales.png")
    print("  2. figura_2_evolucion_errores.png")
    print("  3. figura_3_tabla_convergencia.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

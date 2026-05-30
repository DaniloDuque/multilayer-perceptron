"""
Clasificador Neural Network - Tarea Completa (TP3)

Este módulo implementa todas las tareas especificadas en el enunciado:
1. Generación de datos linealmente separables y no separables en R²
2. Entrenamiento de 4 configuraciones de MLP (M=20/2, con/sin momentum)
3. Evaluación con split 80% train / 20% validación
4. Captura de errores vs épocas para visualización

Configuraciones de entrenamiento:
  - Config A: M=20, γ=0.9, α=0.01  (20 neuronas, con momentum)
  - Config B: M=20, γ=0.0, α=0.01  (20 neuronas, sin momentum)
  - Config C: M=2,  γ=0.9, α=0.01  (2 neuronas, con momentum)
  - Config D: M=2,  γ=0.0, α=0.01  (2 neuronas, sin momentum)

Parámetros de generación de datos:
  - Separables: n_samples=500, clusters=2, std=1.0
  - No separables: n_samples=500, clusters=2, std=4.0

Autor: TP3 - Implementación
"""

import torch
import numpy as np
import pickle
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from data_generator import generar_datos_R2
from multilayer_perceptron import MultilayerPerceptron


# ============================================================================
# UTILIDADES DE DISPOSITIVO GPU
# ============================================================================

def get_device():
    """
    Detecta y retorna el dispositivo PyTorch disponible (CUDA si está disponible,
    CPU en caso contrario).
    
    Returns
    -------
    torch.device
        Dispositivo PyTorch para ejecución optimizada.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] CUDA disponible. Usando: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[CPU] CUDA no disponible. Usando CPU.")
    return device


# ============================================================================
# GENERACIÓN DE DATOS CON SPLIT 80/20
# ============================================================================

def generar_datos_80_20(separable=True, n_samples=500, random_state=42, device=None):
    """
    Genera datos de clasificación binaria en R² con split 80% training / 20% validación.
    
    Envuelve la función generar_datos_R2() y repartita los datos al 80/20 usando
    train_test_split de sklearn.
    
    Nota: generar_datos_R2 hace split 70/30 internamente. Este wrapper toma los
    datos combinados y los repartita nuevamente al 80/20 solicitado.
    
    Parameters
    ----------
    separable : bool, optional
        Si True genera datos linealmente separables (std=1.0),
        si False genera datos solapados (std=4.0). Default: True
    n_samples : int, optional
        Número total de muestras a generar. Default: 500
    random_state : int, optional
        Semilla para reproducibilidad. Default: 42
    device : torch.device, optional
        Dispositivo PyTorch (CPU o CUDA). Default: CPU
        
    Returns
    -------
    dict
        Diccionario con keys:
        - 'X_train': tensor (n_train, 3) con bias aumentado
        - 'X_val': tensor (n_val, 3) con bias aumentado
        - 'T_train': tensor (n_train, 1) etiquetas training
        - 'T_val': tensor (n_val, 1) etiquetas validación
        - 'n_train': número de muestras training
        - 'n_val': número de muestras validación
        - 'params': diccionario con parámetros usados
    """
    # Generar datos crudos (sin split aún)
    std_label = "separable" if separable else "no-separable"
    std = 1.0 if separable else 4.0
    
    print(f"\n[Datos] Generando {n_samples} muestras ({std_label}, std={std})...")
    
    X, y = make_blobs(n_samples=n_samples, centers=2, 
                      cluster_std=std, random_state=random_state)
    
    # Agregar columna de bias (unos)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Split 80/20 training/validación
    X_train, X_val, T_train, T_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Convertir a tensores PyTorch
    if device is None:
        device = torch.device("cpu")
    
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    T_train = torch.tensor(T_train, dtype=torch.float32, device=device).unsqueeze(1)
    T_val = torch.tensor(T_val, dtype=torch.float32, device=device).unsqueeze(1)
    
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    print(f"[Datos] Muestras training: {n_train} (80%), validación: {n_val} (20%)")
    print(f"[Datos] Shapes: X_train={X_train.shape}, T_train={T_train.shape}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'T_train': T_train,
        'T_val': T_val,
        'n_train': n_train,
        'n_val': n_val,
        'params': {
            'separable': separable,
            'n_samples': n_samples,
            'std': std,
            'random_state': random_state
        }
    }


# ============================================================================
# ENTRENAMIENTO DE CONFIGURACIONES
# ============================================================================

def entrenar_configuracion(mlp, config_name, X_train, T_train, X_val, T_val,
                          num_epochs=50):
    """
    Entrena una configuración individual de MLP por un número fijo de épocas.
    
    Parameters
    ----------
    mlp : MultilayerPerceptron
        Instancia del modelo a entrenar (debe estar inicializado).
    config_name : str
        Nombre identificador de la configuración (ej: "Config_A_20_0.9").
    X_train : torch.Tensor
        Features training, shape (n_train, 3).
    T_train : torch.Tensor
        Etiquetas training, shape (n_train, 1).
    X_val : torch.Tensor
        Features validación, shape (n_val, 3).
    T_val : torch.Tensor
        Etiquetas validación, shape (n_val, 1).
    num_epochs : int, optional
        Número de épocas de entrenamiento. Default: 50
        
    Returns
    -------
    dict
        Diccionario con keys:
        - 'train_errors': list de errores MSE por época (training)
        - 'val_errors': list de errores MSE por época (validación)
        - 'final_train_error': error MSE final en training
        - 'final_val_error': error MSE final en validación
        - 'epochs': número total de épocas entrenadas
    """
    print(f"\n  [Entrenamiento] {config_name}...")
    
    # Entrenar
    train_errors, val_errors = mlp.train_mlp(
        num_epochs=num_epochs,
        X=X_train,
        T=T_train,
        X_val=X_val,
        T_val=T_val
    )
    
    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    
    print(f"  [Entrenamiento] {config_name} completado.")
    print(f"    Error Train final: {final_train_error:.6f}")
    print(f"    Error Val final:   {final_val_error:.6f}")
    
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'final_train_error': final_train_error,
        'final_val_error': final_val_error,
        'epochs': num_epochs
    }


def entrenar_todas_configuraciones(datos, device, num_epochs=50):
    """
    Entrena las 4 configuraciones de MLP para un dataset dado.
    
    Configuraciones (varían M y γ):
    - A: M=20, γ=0.9  (muchas neuronas, con momentum)
    - B: M=20, γ=0.0  (muchas neuronas, sin momentum)
    - C: M=2,  γ=0.9  (pocas neuronas, con momentum)
    - D: M=2,  γ=0.0  (pocas neuronas, sin momentum)
    
    Parámetros fijos:
    - D = 2 (features de entrada) + 1 bias = 3 total
    - K = 1 (neurona de salida, clasificación binaria)
    - α = 0.01 (tasa de aprendizaje)
    - max_weights = 0.1 (rango inicial de pesos)
    
    Parameters
    ----------
    datos : dict
        Diccionario retornado por generar_datos_80_20() con X_train, X_val, etc.
    device : torch.device
        Dispositivo PyTorch para cálculos.
    num_epochs : int, optional
        Número de épocas para cada configuración. Default: 50
        
    Returns
    -------
    dict
        Diccionario con keys 'A', 'B', 'C', 'D', cada uno conteniendo:
        - 'mlp': instancia entrenada de MultilayerPerceptron
        - 'train_errors': evolución de error training
        - 'val_errors': evolución de error validación
        - 'final_train_error': error MSE final training
        - 'final_val_error': error MSE final validación
        - 'params': parámetros de configuración usados
    """
    # Los tensores en `datos` incluyen una columna de bias en la posición 0
    # (generada por generar_datos_80_20). La clase MultilayerPerceptron espera
    # entradas SIN la columna de bias (añade su propia columna de unos), por
    # lo que aquí eliminamos la primera columna antes de entrenar.
    X_train = datos['X_train'][:, 1:].to(device)
    X_val = datos['X_val'][:, 1:].to(device)
    T_train = datos['T_train'].to(device)
    T_val = datos['T_val'].to(device)
    
    # Definir configuraciones: (M, gamma, nombre)
    configs = [
        (20, 0.9, "A"),  # M=20, con momentum
        (20, 0.0, "B"),  # M=20, sin momentum
        (2,  0.9, "C"),  # M=2, con momentum
        (2,  0.0, "D"),  # M=2, sin momentum
    ]
    
    results = {}
    
    for M, gamma, config_letter in configs:
        # Hiperparámetros fijos por especificación
        D = 2  # features de entrada (no incluye bias)
        K = 1  # neuronas de salida
        alpha = 0.01  # tasa de aprendizaje
        max_weights = 0.1  # rango inicial de pesos
        
        config_name = f"Config_{config_letter}: M={M}, γ={gamma:.1f}, α={alpha}"
        
        # Crear MLP con estos parámetros
        mlp = MultilayerPerceptron(
            neurons_per_layer=[D, M, K],
            alpha=alpha,
            gamma=gamma,
            max_weights=max_weights
        )
        
        # Mover a GPU si es disponible
        mlp.Wo = mlp.Wo.to(device)
        mlp.Ws = mlp.Ws.to(device)
        mlp.dWo_prev = mlp.dWo_prev.to(device)
        mlp.dWs_prev = mlp.dWs_prev.to(device)
        
        # Entrenar
        result = entrenar_configuracion(
            mlp, config_name,
            X_train, T_train, X_val, T_val,
            num_epochs=num_epochs
        )
        
        result['mlp'] = mlp
        result['params'] = {
            'M': M,
            'gamma': gamma,
            'alpha': alpha,
            'max_weights': max_weights,
            'config_letter': config_letter
        }
        
        results[config_letter] = result
    
    return results


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que orquesta todo el flujo:
    1. Detecta dispositivo GPU/CPU
    2. Genera datos separables y no separables (80/20 split)
    3. Entrena 4 configuraciones para cada dataset (50 épocas cada)
    4. Guarda resultados para visualización
    """
    
    print("=" * 80)
    print("CLASIFICACIÓN NEURAL NETWORK - TAREA COMPLETA (TP3)")
    print("=" * 80)
    
    # ========================================================================
    # PASO 1: Detectar dispositivo
    # ========================================================================
    device = get_device()
    
    # ========================================================================
    # PASO 2: Generar datos
    # ========================================================================
    print("\n" + "=" * 80)
    print("PASO 1: GENERACIÓN DE DATOS")
    print("=" * 80)
    
    datos_separables = generar_datos_80_20(
        separable=True, n_samples=500, random_state=42, device=device
    )
    
    datos_no_separables = generar_datos_80_20(
        separable=False, n_samples=500, random_state=42, device=device
    )
    
    # ========================================================================
    # PASO 3: Entrenar configuraciones
    # ========================================================================
    print("\n" + "=" * 80)
    print("PASO 2: ENTRENAMIENTO DE MLP (4 CONFIGURACIONES × 2 DATASETS)")
    print("=" * 80)
    print("\nConfiguración A: M=20, γ=0.9  (momentum activo)")
    print("Configuración B: M=20, γ=0.0  (sin momentum)")
    print("Configuración C: M=2,  γ=0.9  (momentum activo)")
    print("Configuración D: M=2,  γ=0.0  (sin momentum)")
    
    print("\n--- DATASET SEPARABLE (std=1.0) ---")
    resultados_separables = entrenar_todas_configuraciones(
        datos_separables, device, num_epochs=50
    )
    
    print("\n--- DATASET NO-SEPARABLE (std=4.0) ---")
    resultados_no_separables = entrenar_todas_configuraciones(
        datos_no_separables, device, num_epochs=50
    )
    
    # ========================================================================
    # PASO 4: Compilar resultados finales
    # ========================================================================
    print("\n" + "=" * 80)
    print("PASO 3: COMPILACIÓN DE RESULTADOS")
    print("=" * 80)
    
    resultados_finales = {
        'separable': {
            'datos': datos_separables,
            'configuraciones': resultados_separables
        },
        'no_separable': {
            'datos': datos_no_separables,
            'configuraciones': resultados_no_separables
        }
    }
    
    # ========================================================================
    # PASO 5: Tabla de convergencia final
    # ========================================================================
    print("\n" + "=" * 80)
    print("TABLA DE CONVERGENCIA FINAL (ERROR MSE EN VALIDACIÓN)")
    print("=" * 80)
    
    print("\n{:<15} | {:<18} | {:<18}".format(
        "Configuración", "Separable", "No-Separable"
    ))
    print("-" * 55)
    
    for config_letter in ['A', 'B', 'C', 'D']:
        sep_error = resultados_separables[config_letter]['final_val_error']
        non_sep_error = resultados_no_separables[config_letter]['final_val_error']
        
        config_desc = f"Config {config_letter}"
        print("{:<15} | {:<18.6f} | {:<18.6f}".format(
            config_desc, sep_error, non_sep_error
        ))
    
    # ========================================================================
    # PASO 6: Guardar resultados para visualización
    # ========================================================================
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)
    
    # Mover MLPs a CPU antes de guardar (no es eficiente picklean GPU tensors)
    for dataset_name in ['separable', 'no_separable']:
        for config_letter in resultados_finales[dataset_name]['configuraciones']:
            mlp = resultados_finales[dataset_name]['configuraciones'][config_letter]['mlp']
            mlp.Wo = mlp.Wo.cpu()
            mlp.Ws = mlp.Ws.cpu()
            mlp.dWo_prev = mlp.dWo_prev.cpu()
            mlp.dWs_prev = mlp.dWs_prev.cpu()
    
    # También mover datos a CPU
    for dataset_name in ['separable', 'no_separable']:
        datos = resultados_finales[dataset_name]['datos']
        datos['X_train'] = datos['X_train'].cpu()
        datos['X_val'] = datos['X_val'].cpu()
        datos['T_train'] = datos['T_train'].cpu()
        datos['T_val'] = datos['T_val'].cpu()
    
    # Guardar con pickle
    results_file = 'resultados_clasificacion.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(resultados_finales, f)
    
    print(f"\n✓ Resultados guardados en: {results_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETADO: Datos y modelos listos para visualización")
    print("=" * 80)
    print(f"\nEjecutar: python visualization.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

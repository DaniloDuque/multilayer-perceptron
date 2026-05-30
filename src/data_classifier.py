"""
data_classifier.py — biblioteca de clasificación con MLP (TP3).

Expone funciones puras que el notebook orquestador importa directamente.
El dispositivo se detecta en el notebook y se pasa como parámetro.

Configuraciones de entrenamiento:
  A: M=20, γ=0.9, α=0.01  (20 neuronas, con momentum)
  B: M=20, γ=0.0, α=0.01  (20 neuronas, sin momentum)
  C: M=2,  γ=0.9, α=0.01  (2 neuronas, con momentum)
  D: M=2,  γ=0.0, α=0.01  (2 neuronas, sin momentum)
"""

import torch
from multilayer_perceptron import MultilayerPerceptron

# Configuraciones: (M, gamma, letra)
CONFIGS = [
    (20, 0.9, "A"),
    (20, 0.0, "B"),
    (2,  0.9, "C"),
    (2,  0.0, "D"),
]


## @brief Crea y entrena un MLP con los parámetros dados.
#
#  @param M          Número de neuronas en la capa oculta.
#  @param gamma      Coeficiente de momentum.
#  @param X_train    Tensor (n, D) de entrenamiento.
#  @param T_train    Tensor (n, 1) de etiquetas de entrenamiento.
#  @param X_val      Tensor (n_val, D) de validación.
#  @param T_val      Tensor (n_val, 1) de etiquetas de validación.
#  @param num_epochs Número de épocas.
#  @param alpha      Tasa de aprendizaje.
#  @param max_weights Rango inicial de pesos.
#  @param device     Dispositivo PyTorch.
#  @return dict con mlp, train_errors, val_errors, final_train_error, final_val_error, params.
def train_config(M, gamma, X_train, T_train, X_val, T_val,
                 num_epochs=50, alpha=0.01, max_weights=0.1, device=None):
    D = X_train.shape[1]
    mlp = MultilayerPerceptron([D, M, 1], alpha=alpha, gamma=gamma,
                               max_weights=max_weights)
    if device is not None:
        mlp.Wo       = mlp.Wo.to(device)
        mlp.Ws       = mlp.Ws.to(device)
        mlp.dWo_prev = mlp.dWo_prev.to(device)
        mlp.dWs_prev = mlp.dWs_prev.to(device)

    train_errors, val_errors = mlp.train_mlp(
        num_epochs, X_train, T_train, X_val, T_val
    )
    return {
        'mlp': mlp,
        'train_errors': train_errors,
        'val_errors':   val_errors,
        'final_train_error': train_errors[-1],
        'final_val_error':   val_errors[-1],
        'params': {'M': M, 'gamma': gamma, 'alpha': alpha,
                   'max_weights': max_weights},
    }


## @brief Entrena las 4 configuraciones (A–D) sobre un par train/val.
#
#  @param X_train    Tensor (n, D) sin columna de bias.
#  @param T_train    Tensor (n, 1).
#  @param X_val      Tensor (n_val, D) sin columna de bias.
#  @param T_val      Tensor (n_val, 1).
#  @param num_epochs Épocas por configuración.
#  @param device     Dispositivo PyTorch.
#  @return dict {'A': ..., 'B': ..., 'C': ..., 'D': ...} con resultados de train_config.
def train_all_configs(X_train, T_train, X_val, T_val,
                      num_epochs=50, device=None):
    results = {}
    for M, gamma, letter in CONFIGS:
        results[letter] = train_config(
            M, gamma,
            X_train.to(device) if device else X_train,
            T_train.to(device) if device else T_train,
            X_val.to(device)   if device else X_val,
            T_val.to(device)   if device else T_val,
            num_epochs=num_epochs,
            device=device,
        )
        results[letter]['params']['config_letter'] = letter
    return results

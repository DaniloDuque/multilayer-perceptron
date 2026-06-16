"""
data_classifier.py — biblioteca de clasificación con MLP (TP3).

Configuraciones de entrenamiento:
  A: M=20, γ=0.9, α=0.01
  B: M=20, γ=0.0, α=0.01
  C: M=2,  γ=0.9, α=0.01
  D: M=2,  γ=0.0, α=0.01
"""

import torch
from multilayer_perceptron import MultilayerPerceptron

# Configuraciones fijas para comparar distintas combinaciones de M y momentum.
CONFIGS = [
    (20, 0.9, "A"),
    (20, 0.0, "B"),
    (2,  0.9, "C"),
    (2,  0.0, "D"),
]


## @brief Crea y entrena un MLP con los parámetros dados.
#
#  El dispositivo se infiere de X_train para inicializar los pesos del MLP.
#  Los tensores de entrada se asumen ya en el dispositivo correcto.
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
#  @return dict con mlp, train_errors, val_errors, final_train_error, final_val_error, params.
def train_config(M, gamma, X_train, T_train, X_val, T_val,
                 num_epochs=50, alpha=0.01, max_weights=0.1):
    device = X_train.device
    mlp = MultilayerPerceptron([X_train.shape[1], M, 1], alpha=alpha,
                               gamma=gamma, max_weights=max_weights, device=device)

    train_errors, val_errors = mlp.train_mlp(num_epochs, X_train, T_train, X_val, T_val)
    return {
        'mlp': mlp,
        'train_errors': train_errors,
        'val_errors':   val_errors,
        'final_train_error': train_errors[-1],
        'final_val_error':   val_errors[-1],
        'params': {'M': M, 'gamma': gamma, 'alpha': alpha, 'max_weights': max_weights},
    }


## @brief Entrena las 4 configuraciones (A–D) sobre un par train/val.
#
#  Los tensores de entrada deben estar ya en el dispositivo correcto.
#
#  @param X_train    Tensor (n, D).
#  @param T_train    Tensor (n, 1).
#  @param X_val      Tensor (n_val, D).
#  @param T_val      Tensor (n_val, 1).
#  @param num_epochs Épocas por configuración.
#  @return dict {'A': ..., 'B': ..., 'C': ..., 'D': ...}.
def train_all_configs(X_train, T_train, X_val, T_val, num_epochs=50):
    # Ejecuta cada configuración A-D y guarda los resultados de entrenamiento.
    results = {}
    for M, gamma, letter in CONFIGS:
        results[letter] = train_config(M, gamma, X_train, T_train, X_val, T_val,
                                       num_epochs=num_epochs)
        results[letter]['params']['config_letter'] = letter
    return results


## @brief Crea la función objetivo para HyperparameterTuner dado un M fijo.
#
#  @param M          Neuronas en la capa oculta (fijo durante la optimización).
#  @param X_train    Tensor de entrenamiento.
#  @param T_train    Tensor de etiquetas de entrenamiento.
#  @param X_val      Tensor de validación.
#  @param T_val      Tensor de etiquetas de validación.
#  @param num_epochs Épocas de entrenamiento por trial.
#  @return Callable  (dict con 'alpha' y 'gamma') -> float (val MSE).
def make_objective(M, X_train, T_train, X_val, T_val, num_epochs=50):
    def objective(params):
        result = train_config(M, params['gamma'], X_train, T_train, X_val, T_val,
                              num_epochs=num_epochs, alpha=params['alpha'])
        return result['final_val_error']
    return objective
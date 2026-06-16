import torch
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

# Genera datos sintéticos y los convierte a tensores de PyTorch para entrenar.
## @brief Genera datos sintéticos de clasificación binaria y los divide en train/test.
#
#  Usa make_blobs con 2 centros. Si @p separable es True, la desviación estándar
#  es 1.0 (clases bien separadas); si es False, es 4.0 (clases solapadas).
#  NO agrega columna de bias — MultilayerPerceptron la añade internamente.
#
#  @param separable    Si es True genera datos linealmente separables.
#  @param n_samples    Número total de muestras a generar.
#  @param test_size    Proporción del conjunto de prueba (0 a 1).
#  @param random_state Semilla para reproducibilidad.
#  @param device       Dispositivo PyTorch destino (None = CPU).
#  @return Tupla (X_train, X_test, T_train, T_test) como tensores float32.
def generate_data(separable=True, n_samples=500, test_size=0.2,
                  random_state=42, device=None):
    std = 1.0 if separable else 4.0
    X, y = make_blobs(n_samples=n_samples, centers=2,
                      cluster_std=std, random_state=random_state)

    X_train, X_test, T_train, T_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convertir los datos de NumPy a tensores PyTorch float32.
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    T_train = torch.tensor(T_train, dtype=torch.float32, device=device).unsqueeze(1)
    T_test  = torch.tensor(T_test,  dtype=torch.float32, device=device).unsqueeze(1)
    return X_train, X_test, T_train, T_test


## @brief Genera datos sintéticos no linealmente separables en forma de lunas
#         y los divide en train/test.
#
#  Usa make_moons de sklearn. Si @p noise es bajo (ej. 0.05), las lunas están
#  bien definidas; si es alto (ej. 0.3), los puntos se solapan más.
#  NO agrega columna de bias — MultilayerPerceptron la añade internamente.
#
#  @param noise        Desviación estándar del ruido gaussiano aplicado a los datos.
#  @param n_samples    Número total de muestras a generar.
#  @param test_size    Proporción del conjunto de prueba (0 a 1).
#  @param random_state Semilla para reproducibilidad.
#  @param device       Dispositivo PyTorch destino (None = CPU).
#  @return Tupla (X_train, X_test, T_train, T_test) como tensores float32.
def generate_moons_data(noise=0.1, n_samples=500, test_size=0.2,
                        random_state=42, device=None):
    X, y = make_moons(n_samples=n_samples, noise=noise,
                      random_state=random_state)
    X_train, X_test, T_train, T_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    T_train = torch.tensor(T_train, dtype=torch.float32, device=device).unsqueeze(1)
    T_test  = torch.tensor(T_test,  dtype=torch.float32, device=device).unsqueeze(1)
    return X_train, X_test, T_train, T_test
import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

## @brief Genera datos sintéticos de clasificación binaria y los divide en train/test.
#
#  Usa make_blobs con 2 centros. Si @p separable es True, la desviación estándar
#  es 1.0 (clases bien separadas); si es False, es 4.0 (clases solapadas).
#  Agrega una columna de unos como término de bias y convierte todo a tensores PyTorch.
#
#  @param separable    Si es True genera datos linealmente separables.
#  @param n_samples    Número total de muestras a generar.
#  @param random_state Semilla para reproducibilidad.
#  @param device       Dispositivo PyTorch destino (None = CPU).
#  @return Tupla (X_train, X_test, T_train, T_test) como tensores float32.
def generar_datos_R2(separable=True, n_samples=500, random_state=42, device=None):
    std = 1.0 if separable else 4.0
    X, y = make_blobs(n_samples=n_samples, centers=2, 
                      cluster_std=std, random_state=random_state)
    
    # Agregar columna de bias (unos)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    X_train, X_test, T_train, T_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    T_train = torch.tensor(T_train, dtype=torch.float32, device=device).unsqueeze(1)
    T_test  = torch.tensor(T_test,  dtype=torch.float32, device=device).unsqueeze(1)
    return X_train, X_test, T_train, T_test
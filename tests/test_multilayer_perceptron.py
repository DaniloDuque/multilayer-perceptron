import os
import sys
import torch
import numpy as np

# Asegura que el paquete src sea importable desde el directorio de pruebas.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from multilayer_perceptron import MultilayerPerceptron

# Pruebas unitarias para validar comportamiento del perceptrón multicapa.

def seed_everything(seed=42):
    # Fija semillas para que las pruebas sean reproducibles en CPU y GPU.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def test_init_shapes_and_bounds():
    """Verifica dimensiones y límites de los pesos iniciales del MLP."""
    seed_everything()
    mlp = MultilayerPerceptron([3, 5, 2], alpha=0.1, gamma=0.9, max_weights=0.1, device=torch.device('cpu'))

    assert mlp.Wo.shape == (3 + 1, 5)
    assert mlp.Ws.shape == (5 + 1, 2)

    # pesos dentro del rango inicial esperado
    assert float(torch.max(torch.abs(mlp.Wo))) <= 0.100001
    assert float(torch.max(torch.abs(mlp.Ws))) <= 0.100001

    # los términos de momentum deben comenzar en cero
    assert torch.allclose(mlp.dWo_prev, torch.zeros_like(mlp.dWo_prev))
    assert torch.allclose(mlp.dWs_prev, torch.zeros_like(mlp.dWs_prev))


def test_forward_shapes_and_range():
    seed_everything()
    mlp = MultilayerPerceptron([2, 3, 1], alpha=0.1, gamma=0.0, max_weights=0.05, device=torch.device('cpu'))
    X = torch.randn(7, 2)
    ys = mlp.forward(X)

    assert ys.shape == (7, 1)
    assert mlp.yo.shape == (7, 3)

    # las salidas de la función sigmoide deben estar en [0,1]
    assert float(torch.min(ys)) >= 0.0 - 1e-6
    assert float(torch.max(ys)) <= 1.0 + 1e-6


def test_backpropagate_deltas_shapes_and_values():
    seed_everything()
    mlp = MultilayerPerceptron([2, 4, 2], alpha=0.1, gamma=0.0, max_weights=0.05, device=torch.device('cpu'))
    X = torch.tensor([[0.5, -0.2], [1.0, 0.0]])
    T = torch.tensor([[0.1, 0.9], [0.0, 1.0]])

    ys = mlp.forward(X)
    mlp.backpropagate_deltas(T)

    # formas esperadas de los deltas de salida y ocultos
    assert mlp.delta_s.shape == (2, 2)
    assert mlp.delta_o.shape == (2, 4)

    # el delta de salida debe coincidir con la fórmula del gradiente para sigmoid
    manual_delta_s = (mlp.ys - T) * (mlp.ys * (1.0 - mlp.ys))
    assert torch.allclose(mlp.delta_s, manual_delta_s, atol=1e-6)


def test_update_weights_changes_weights_and_updates_momentum():
    seed_everything()
    mlp = MultilayerPerceptron([2, 3, 1], alpha=0.5, gamma=0.8, max_weights=0.05, device=torch.device('cpu'))
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    T = torch.tensor([[0.0], [1.0]])

    # clonar pesos antes de la actualización para comparar cambios
    Wo_before = mlp.Wo.clone()
    Ws_before = mlp.Ws.clone()

    mlp.forward(X)
    mlp.backpropagate_deltas(T)
    mlp.update_weights(X)

    # Al menos algunos pesos deben haber cambiado tras la actualización
    assert not torch.allclose(Wo_before, mlp.Wo)
    assert not torch.allclose(Ws_before, mlp.Ws)

    # los términos de momentum deben ser distintos de cero después de la actualización
    assert float(torch.max(torch.abs(mlp.dWo_prev))) > 0.0
    assert float(torch.max(torch.abs(mlp.dWs_prev))) > 0.0

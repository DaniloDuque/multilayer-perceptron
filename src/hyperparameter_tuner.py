"""
hyperparameter_tuner.py
Wrapper reutilizable de Optuna para el TP3.

Uso:
    tuner = HyperparameterTuner(
        space={'alpha': ('float', 1e-4, 1.0, True), 'gamma': ('float', 0.0, 0.99, False)},
        objective=lambda p: train_and_eval(p['alpha'], p['gamma']),
    )
    best = tuner.run()
"""

import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reducir el nivel de logging de Optuna para no saturar la salida.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Optimizador de hiperparámetros basado en Optuna.

    Parámetros
    ----------
    space : dict
        Espacio de búsqueda. Cada entrada puede ser:
          ('float', low, high, log)      -> valor continuo
          ('int',   low, high, log)      -> valor entero
          ('categorical', [opciones])    -> valor discreto

    objective : callable
        Función (dict de params) -> float a minimizar (o maximizar).

    n_trials : int
    direction : 'minimize' | 'maximize'
    seed : int
    verbose : bool
    """

    def __init__(self, space, objective, n_trials=50,
                 direction='minimize', seed=42, verbose=True):
        self.space     = space
        self.objective = objective
        self.n_trials  = n_trials
        self.direction = direction
        self.seed      = seed
        self.verbose   = verbose
        self.study     = None

    def run(self):
        """Ejecuta la optimización y retorna {'params': ..., 'value': ...}."""
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        self.study.optimize(self._objective, n_trials=self.n_trials)

        best = {'params': self.study.best_params, 'value': self.study.best_value}

        if self.verbose:
            print(f'Mejor valor: {best["value"]:.6f}')
            for k, v in best['params'].items():
                print(f'  {k} = {v}')

        return best

    def plot_history(self, ax=None, title='Historial de optimización'):
        """Error por trial y mejor acumulado."""
        self._require_study()
        values = [t.value for t in self.study.trials if t.value is not None]
        best_curve = np.minimum.accumulate(values) if self.direction == 'minimize' \
                     else np.maximum.accumulate(values)

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.scatter(range(1, len(values) + 1), values, alpha=0.4, s=20, label='Trial')
        ax.plot(range(1, len(values) + 1), best_curve, color='red', linewidth=2, label='Mejor')
        ax.set(xlabel='Trial', ylabel='Objetivo', title=title)
        ax.legend()
        ax.grid(alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_importance(self, ax=None, title='Importancia de hiperparámetros'):
        """Importancia relativa de cada parámetro (requiere >= 10 trials)."""
        self._require_study()
        if len(self.study.trials) < 10:
            print('Se necesitan al menos 10 trials.')
            return

        imp    = optuna.importance.get_param_importances(self.study)
        names  = list(imp.keys())
        values = list(imp.values())

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.6)))

        bars = ax.barh(names, values, color='steelblue')
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set(xlabel='Importancia relativa', title=title, xlim=(0, max(values) * 1.15))
        ax.grid(axis='x', alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_curves(self, curves, title='Curvas de aprendizaje', ax=None):
        """
        Grafica curvas error vs. época para distintas configuraciones.

        curves : dict  {'nombre': [error_ep1, error_ep2, ...]}
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 5))

        for (name, errors), color in zip(curves.items(),
                                         plt.cm.tab10(np.linspace(0, 0.9, len(curves)))):
            ax.plot(errors, label=name, color=color, linewidth=1.5)

        ax.set(xlabel='Época', ylabel='Error', title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        if standalone:
            plt.tight_layout()
            plt.show()

    def summary(self, top_n=5):
        """DataFrame con los mejores top_n trials."""
        self._require_study()

        rows = [{'trial': t.number, 'valor': round(t.value, 6), **t.params}
                for t in self.study.trials if t.value is not None]

        return (pd.DataFrame(rows)
                  .sort_values('valor', ascending=(self.direction == 'minimize'))
                  .head(top_n)
                  .reset_index(drop=True))

    # ── interno ───────────────────────────────────────────────────────────────

    def _objective(self, trial):
        params = {}
        for name, spec in self.space.items():
            kind = spec[0]
            if kind == 'float':
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
            elif kind == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2], log=spec[3])
            elif kind == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])
            else:
                raise ValueError(f'Tipo desconocido: {kind}')
        return self.objective(params)

    def _require_study(self):
        if self.study is None:
            raise RuntimeError('Llama a .run() primero.')
"""Module 7 : Réseau de neurones informé par la physique (PINN).

Résout l'équation d'advection-diffusion  u_t + c*u_x - nu*u_xx = f(x, t)
où f est le terme source exact généré symboliquement (Module 3), avec pour
solution manufacturée u(x, t) = tanh(x - c*t).
"""
from __future__ import annotations

import argparse
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.symbolic_derivation import symbolic_advection_diffusion


def get_device() -> torch.device:
    """Détecte la meilleure architecture disponible (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


class PINN(nn.Module):
    """Perceptron multicouche approximant û(x, t)."""

    def __init__(self, hidden: int = 32, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(inputs)
        return result


def physics_residual(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    c: float,
    nu: float,
    f_val: torch.Tensor,
) -> torch.Tensor:
    """Résidu de l'EDP : û_t + c*û_x - nu*û_xx - f, via différentiation automatique."""
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)

    grad_outputs = torch.ones_like(u)
    du_dx = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    return du_dt + c * du_dx - nu * d2u_dx2 - f_val


def get_physics_loss(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    c: float,
    nu: float,
    f_val: torch.Tensor,
) -> torch.Tensor:
    """Perte physique L_physique = moyenne du carré du résidu de l'EDP (éq. 1 du TP)."""
    residual = physics_residual(model, x, t, c, nu, f_val)
    loss: torch.Tensor = torch.mean(residual**2)
    return loss


def get_data_loss(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    u_exact: torch.Tensor,
) -> torch.Tensor:
    """Perte sur les données/conditions aux limites : écart à la solution exacte."""
    inputs = torch.cat([x, t], dim=1)
    prediction = model(inputs)
    loss: torch.Tensor = torch.mean((prediction - u_exact) ** 2)
    return loss


def _sample_collocation(
    n: int, c: float, nu: float, f_func: Callable[..., object], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Échantillonne n points intérieurs et évalue le terme source f en ces points."""
    x_np = np.random.rand(n, 1).astype(np.float64)
    t_np = np.random.rand(n, 1).astype(np.float64)
    f_np = np.asarray(f_func(x_np, t_np, c, nu), dtype=np.float64).reshape(n, 1)

    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True, device=dev)
    t = torch.tensor(t_np, dtype=torch.float32, requires_grad=True, device=dev)
    f_val = torch.tensor(f_np, dtype=torch.float32, device=dev)
    return x, t, f_val


def _sample_boundary(
    n: int, c: float, u_func: Callable[..., object], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Échantillonne des points au bord du domaine avec leur valeur exacte u."""
    x_np = np.random.rand(n, 1).astype(np.float64)
    t_np = np.random.rand(n, 1).astype(np.float64)
    # Condition initiale (t=0) et bords spatiaux (x in {0, 1})
    x_np[: n // 2] = np.random.choice([0.0, 1.0], size=(n // 2, 1))
    t_np[n // 2 :] = 0.0
    u_np = np.asarray(u_func(x_np, t_np, c, 1.0), dtype=np.float64).reshape(n, 1)

    x = torch.tensor(x_np, dtype=torch.float32, device=dev)
    t = torch.tensor(t_np, dtype=torch.float32, device=dev)
    u_exact = torch.tensor(u_np, dtype=torch.float32, device=dev)
    return x, t, u_exact


def train_pinn(
    epochs: int = 200,
    n_collocation: int = 512,
    n_boundary: int = 128,
    c: float = 1.0,
    nu: float = 0.05,
    lr: float = 1e-3,
    seed: int = 42,
    dev: torch.device | None = None,
) -> tuple[PINN, list[float]]:
    """Entraîne le PINN et retourne le modèle et l'historique de perte."""
    if dev is None:
        dev = device
    torch.manual_seed(seed)
    np.random.seed(seed)

    u_func, f_func = symbolic_advection_diffusion()
    model = PINN().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()

        xc, tc, fc = _sample_collocation(n_collocation, c, nu, f_func, dev)
        loss_phys = get_physics_loss(model, xc, tc, c, nu, fc)

        xb, tb, ub = _sample_boundary(n_boundary, c, u_func, dev)
        loss_data = get_data_loss(model, xb, tb, ub)

        loss = loss_phys + loss_data
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        history.append(float(loss.item()))

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement du PINN")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.05)
    args = parser.parse_args()

    model, history = train_pinn(epochs=args.epochs, c=args.c, nu=args.nu)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Modèle entraîné sauvegardé dans : {args.save_path}")
    print(f"Perte finale : {history[-1]:.6e}")


if __name__ == "__main__":
    main()

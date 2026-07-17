import argparse
import runpy
import sys
from unittest.mock import patch

import torch

from src.deep_pinn import (
    PINN,
    get_data_loss,
    get_device,
    get_physics_loss,
    main,
    physics_residual,
    train_pinn,
)


def test_pinn_initialization():
    model = PINN()
    assert isinstance(model, PINN)


def test_pinn_forward_shape():
    model = PINN()
    inputs = torch.randn(8, 2)
    out = model(inputs)
    assert out.shape == (8, 1)


def test_get_device_valid():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_physics_residual_uses_autograd():
    """Le résidu doit dépendre des dérivées (non nul pour un réseau aléatoire)."""
    model = PINN()
    x = torch.rand(16, 1, requires_grad=True)
    t = torch.rand(16, 1, requires_grad=True)
    f_val = torch.zeros(16, 1)
    residual = physics_residual(model, x, t, c=1.0, nu=0.05, f_val=f_val)
    assert residual.shape == (16, 1)
    # Le résidu n'est pas identiquement nul (les dérivées interviennent réellement)
    assert torch.any(residual != 0)


def test_get_physics_loss_scalar_nonnegative():
    model = PINN()
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    f_val = torch.zeros(10, 1)
    loss = get_physics_loss(model, x, t, c=1.0, nu=0.05, f_val=f_val)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_get_data_loss_zero_when_matching():
    """Si la prédiction égale la cible, la perte de données est nulle."""
    model = PINN()
    x = torch.rand(6, 1)
    t = torch.rand(6, 1)
    with torch.no_grad():
        u_pred = model(torch.cat([x, t], dim=1))
    loss = get_data_loss(model, x, t, u_pred)
    assert loss.item() < 1e-10


def test_train_pinn_reduces_loss():
    """L'entraînement doit faire décroître la perte (le PINN apprend)."""
    model, history = train_pinn(epochs=60, n_collocation=128, n_boundary=32, seed=0)
    assert isinstance(model, PINN)
    assert len(history) == 60
    # La perte finale doit être inférieure à la perte initiale
    assert history[-1] < history[0]


def test_train_pinn_deterministic_with_seed():
    _, h1 = train_pinn(epochs=15, n_collocation=64, n_boundary=16, seed=123)
    _, h2 = train_pinn(epochs=15, n_collocation=64, n_boundary=16, seed=123)
    assert abs(h1[-1] - h2[-1]) < 1e-6


def test_main_saves_model(tmp_path):
    save_path = tmp_path / "weights.pth"
    mock_args = argparse.Namespace(save_path=str(save_path), epochs=5, c=1.0, nu=0.05)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
    assert save_path.exists()
    # Rechargeable dans un PINN
    PINN().load_state_dict(torch.load(str(save_path)))


def test_main_guard_runpy(tmp_path):
    save_path = tmp_path / "guard.pth"
    argv = ["deep_pinn.py", "--save_path", str(save_path), "--epochs", "3"]
    with patch.object(sys, "argv", argv):
        runpy.run_module("src.deep_pinn", run_name="__main__")
    assert save_path.exists()

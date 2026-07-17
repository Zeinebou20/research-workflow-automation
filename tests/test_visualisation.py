import argparse
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.deep_pinn import PINN
from src.visualisation import (
    exact_solution,
    get_inference,
    main,
    plot_interactive_surface,
    plot_static_dual_panel,
)


@pytest.fixture
def mock_model():
    model = PINN()
    model.eval()
    return model


def test_get_inference_shape(mock_model):
    X, T, Z = get_inference(mock_model, (0, 1), (0, 1), n_points=10)
    assert Z.shape == (10, 10)
    assert isinstance(Z, np.ndarray)


def test_get_inference_raises_on_none():
    with pytest.raises(ValueError):
        get_inference(None, (0, 1), (0, 1), n_points=5)


def test_exact_solution_matches_tanh():
    X, T = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    u = exact_solution(X, T, c=1.0)
    np.testing.assert_allclose(u, np.tanh(X - T), atol=1e-12)


def test_plot_static_dual_panel_creates_pdf(mock_model, tmp_path):
    output_path = tmp_path / "fig.pdf"
    result = plot_static_dual_panel(mock_model, (0, 1), (0, 1), output_path=str(output_path))
    assert os.path.exists(output_path)
    assert result == str(output_path)


def test_plot_interactive_surface_creates_html(mock_model, tmp_path):
    output_path = tmp_path / "surface.html"
    plot_interactive_surface(mock_model, (0, 1), (0, 1), output_path=str(output_path))
    assert os.path.exists(output_path)
    # Le HTML est autonome (contient le code Plotly)
    content = output_path.read_text(encoding="utf-8")
    assert "plotly" in content.lower()


def test_main_generates_artifacts(mock_model, tmp_path):
    weights = tmp_path / "w.pth"
    torch.save(mock_model.state_dict(), str(weights))
    pdf = tmp_path / "out.pdf"
    html = tmp_path / "out.html"
    mock_args = argparse.Namespace(weights=str(weights), output=str(pdf), html=str(html))
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()
    assert pdf.exists()
    assert html.exists()

"""Module 8 : Visualisation scientifique pour publication.

Génère une figure statique double-panneau (solution PINN + erreur absolue) au
format PDF vectoriel avec rendu LaTeX, ainsi qu'une surface 3D interactive Plotly
exportée en HTML autonome.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402  backend non interactif avant pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

from src.deep_pinn import PINN  # noqa: E402
from src.symbolic_derivation import symbolic_advection_diffusion  # noqa: E402

sns.set_theme(style="whitegrid", context="paper")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_inference(
    model: PINN,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Évalue le PINN sur une grille (x, t) et retourne X, T et la prédiction."""
    if model is None:
        raise ValueError("Modèle invalide")

    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    X, T = np.meshgrid(x, t)

    input_data = np.stack([X.flatten(), T.flatten()], axis=1)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).reshape(X.shape)
    return X, T, prediction.numpy()


def exact_solution(X: np.ndarray, T: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Solution manufacturée exacte u(x, t) = tanh(x - c*t)."""
    u_func, _ = symbolic_advection_diffusion()
    return np.asarray(u_func(X, T, c, 1.0), dtype=np.float64)


def _enable_latex() -> bool:
    """Active le rendu LaTeX (usetex) si demandé ET si une distribution TeX est présente.

    Le rendu est piloté par la variable d'environnement USE_LATEX pour garantir la
    reproductibilité : sur un runner CI sans distribution TeX (ou une machine où le
    sous-processus latex bloquerait), on retombe sur le rendu mathtext natif.
    """
    from shutil import which

    if os.environ.get("USE_LATEX", "0") == "1" and which("latex") is not None:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
        return True
    plt.rcParams.update({"text.usetex": False})
    return False


def plot_static_dual_panel(
    model: PINN,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    output_path: str = "outputs/figures/solution_surface.pdf",
    c: float = 1.0,
) -> str:
    """Figure double-panneau : solution PINN (heatmap) + erreur absolue. Export PDF."""
    print("Génération de la figure statique double-panneau...")
    _enable_latex()

    X, T, Z = get_inference(model, x_range, t_range)
    U_exact = exact_solution(X, T, c)
    error = np.abs(Z - U_exact)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cf1 = ax1.contourf(X, T, Z, levels=50, cmap="viridis")
    ax1.set_title(r"Solution approchée $\hat{u}(x,t)$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    fig.colorbar(cf1, ax=ax1)

    cf2 = ax2.contourf(X, T, error, levels=50, cmap="magma")
    ax2.set_title(r"Erreur absolue $|\hat{u} - u_{exact}|$")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$t$")
    fig.colorbar(cf2, ax=ax2)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Figure PDF enregistrée : {output_path}")
    return output_path


def plot_interactive_surface(
    model: PINN,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    output_path: str = "outputs/figures/visualisation_pinn_interactive.html",
) -> str:
    """Surface 3D interactive û(x, t) via Plotly, exportée en HTML autonome."""
    print("Génération de la surface interactive Plotly...")
    X, T, Z = get_inference(model, x_range, t_range)

    fig = go.Figure(data=[go.Surface(x=X, y=T, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title="Solution PINN û(x, t)",
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="û"),
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"HTML interactif enregistré : {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation des résultats du PINN")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/figures/solution_surface.pdf")
    parser.add_argument("--html", type=str, default="outputs/figures/visualisation_pinn_interactive.html")
    args = parser.parse_args()

    model = PINN()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    plot_static_dual_panel(model, (0.0, 1.0), (0.0, 1.0), output_path=args.output)
    plot_interactive_surface(model, (0.0, 1.0), (0.0, 1.0), output_path=args.html)


if __name__ == "__main__":
    main()

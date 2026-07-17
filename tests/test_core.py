"""Tests de fumée : garantissent que tous les modules du package s'importent sans erreur."""
import importlib

import pytest

MODULES = [
    "src.symbolic_derivation",
    "src.symbolic_engine",
    "src.numerical_core",
    "src.stability_analysis",
    "src.hpc_acceleration",
    "src.deep_pinn",
    "src.visualisation",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name):
    module = importlib.import_module(module_name)
    assert module is not None


def test_expected_public_functions_exist():
    from src.numerical_core import (
        explore_ndarray_properties,
        make_discretization_grid,
        process_massive_data,
        vectorized_residual,
    )
    from src.symbolic_derivation import symbolic_advection_diffusion
    from src.stability_analysis import (
        analyze_stability,
        compute_residual,
        perturbation_analysis,
    )
    from src.hpc_acceleration import (
        heavy_computation_optimized,
        parameter_sweep,
        profile_computation,
    )
    from src.deep_pinn import PINN, get_physics_loss, train_pinn
    from src.visualisation import get_inference, plot_static_dual_panel

    for fn in (
        explore_ndarray_properties,
        make_discretization_grid,
        process_massive_data,
        vectorized_residual,
        symbolic_advection_diffusion,
        analyze_stability,
        compute_residual,
        perturbation_analysis,
        heavy_computation_optimized,
        parameter_sweep,
        profile_computation,
        get_physics_loss,
        train_pinn,
        get_inference,
        plot_static_dual_panel,
    ):
        assert callable(fn)
    assert callable(PINN)

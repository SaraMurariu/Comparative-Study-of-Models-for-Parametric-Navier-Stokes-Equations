import json
import os
import csv

from PINN import PINN_Methods
from other_utilities import export_folder


# =============================================================================
# PINN lambda tuning script
# =============================================================================
# This script is intentionally separated from main.py.
# It trains short physics-only PINN runs for different lambda configurations and
# saves the best configuration. No FOM snapshots are used here.
# The final PINN training can then load the selected lambdas from:
#     Export/Models/pinn_best_lambdas.json
# =============================================================================

file_path, mesh_path, solution_path = export_folder("./Export")

results_folder = "./Results/PINN_Tuning"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

best_lambdas_path = file_path + "/Models/pinn_best_lambdas.json"
tuning_results_path = results_folder + "/pinn_lambda_tuning_results.csv"

# =============================================================================
# Parameter domain and PINN object
# =============================================================================

mu0_range = [1.0, 10.0]
mu1_range = [1.0, 3.0]

pinn = PINN_Methods(
    mu0_range=mu0_range,
    mu1_range=mu1_range,
    fom_data=None
)


# =============================================================================
# Lambda configurations to test
# =============================================================================

lambda_configs = [
    # Baseline configurations.
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 0.1,
        "lambda_pressure_anchor": 0.1
    },

    # PDE-focused configurations.
    {
        "lambda_pde": 2.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 20.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 50.0,
        "lambda_divergence": 1.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },

    # Divergence-focused configurations.
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 2.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 10.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 1.0,
        "lambda_divergence": 20.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },

    # Joint PDE-divergence configurations.
    {
        "lambda_pde": 2.0,
        "lambda_divergence": 2.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 2.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 10.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 2.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 10.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 20.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 0.1
    },

    # Boundary weight checks. Since velocity can be approximately easy to fit on the boundary,
    # these configurations test whether reducing boundary emphasis helps the PDE residual.
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 0.1,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 0.1,
        "lambda_pressure_anchor": 0.1
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 10.0,
        "lambda_boundary": 0.1,
        "lambda_pressure_anchor": 0.1
    },

    # Pressure-anchor checks.
    {
        "lambda_pde": 5.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 1.0
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 5.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 1.0
    },
    {
        "lambda_pde": 10.0,
        "lambda_divergence": 10.0,
        "lambda_boundary": 1.0,
        "lambda_pressure_anchor": 1.0
    }
]


# =============================================================================
# Run tuning
# =============================================================================

best_model, best_history, best_lambda_config, tuning_results = pinn.tune_lambdas(
    lambda_configs=lambda_configs,
    network_kwargs={
        "width": 128,
        "latent_dim": 64,
        "n_residual_blocks": 4
    },
    train_kwargs={
        "n_epochs": 3000,
        "n_interior": 2048,
        "n_boundary": 512,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-8,
        "print_every": 500
    },
    selection_metric="loss_pde"
)


# =============================================================================
# Save tuning results
# =============================================================================

with open(best_lambdas_path, "w") as file:
    json.dump(best_lambda_config, file, indent=4)

fieldnames = [
    "config_id",
    "lambda_pde",
    "lambda_divergence",
    "lambda_boundary",
    "lambda_pressure_anchor",
    "score",
    "final_total_loss",
    "final_loss_pde",
    "final_loss_divergence",
    "final_loss_boundary",
    "final_loss_pressure_anchor"
]

with open(tuning_results_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for result in tuning_results:
        lambda_config = result["lambda_config"]
        writer.writerow({
            "config_id": result["config_id"],
            "lambda_pde": lambda_config.get("lambda_pde"),
            "lambda_divergence": lambda_config.get("lambda_divergence"),
            "lambda_boundary": lambda_config.get("lambda_boundary"),
            "lambda_pressure_anchor": lambda_config.get("lambda_pressure_anchor"),
            "score": result["score"],
            "final_total_loss": result["final_total_loss"],
            "final_loss_pde": result["final_loss_pde"],
            "final_loss_divergence": result["final_loss_divergence"],
            "final_loss_boundary": result["final_loss_boundary"],
            "final_loss_pressure_anchor": result["final_loss_pressure_anchor"]
        })

print("=" * 100)
print(f"[PINN tuning] Best lambdas saved to: {best_lambdas_path}")
print(f"[PINN tuning] Full tuning table saved to: {tuning_results_path}")
print("=" * 100)
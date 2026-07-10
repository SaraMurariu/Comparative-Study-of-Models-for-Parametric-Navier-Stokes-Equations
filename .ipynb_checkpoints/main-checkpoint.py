import json
import os
import numpy as np

from pypolydim import polydim, gedim
from pypolydim.export_vtk_utilities import ExportVTKUtilities

from other_utilities import export_folder
from Discretization import Discretize
from Solver import Solver
from ROM import ROM_Methods, ROM_NN_Methods, ROMPerformanceEvaluator
from PINN import PINN_Methods

file_path, mesh_path, solution_path = export_folder("./Export")
reduced_model_path = file_path + "/Models/reduced_model.pkl"
podnn_model_path = file_path + "/Models/podnn_model.pkl"
pinn_model_path = file_path + "/Models/pinn_model.pkl"
pinn_best_lambdas_path = file_path + "/Models/pinn_best_lambdas.json"

# Available: 'PODGalerkin', 'PODNN', 'PINN', 'all'
method = 'PINN'  #<--- change method here
visualize = True

mesh_size = 0.001
domain_area = 1.0
# Create unit square domain (0,1)^2
vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0]])
# Create mesh, mesh connectivity, geometric data and VTK utilities needed by FEM solver
# smaller mesh_size means more mesh elements, dofs, more accurate but slower (0.01/0.02 for testing)
discretizer = Discretize(vertices,mesh_size,domain_area,mesh_path)
solver = Solver(discretizer,solution_path)

# boundary-condition labels
info_internal,info_dirichlet,info_neumann_none = solver.set_dofs()

# two dictionaries that define boundary conditions for pressure and velocity
# pressure: enforces p=0 at one point only
p_boundary_info = {
    0: info_internal,
    1: info_dirichlet,
    2: info_neumann_none,
    3: info_neumann_none,
    4: info_neumann_none,
    5: info_neumann_none,
    6: info_neumann_none,
    7: info_neumann_none,
    8: info_neumann_none
}

# velocity 0:internal, 1-8 dirichlet. Fixed on the boundary
u_boundary_info = {
    0: info_internal,
    1: info_dirichlet,
    2: info_dirichlet,
    3: info_dirichlet,
    4: info_dirichlet,
    5: info_dirichlet,
    6: info_dirichlet,
    7: info_dirichlet,
    8: info_dirichlet
}

# Set FOM parameters
mu0 = 1.0
mu1 = 2.0
tol = 1.0e-6
max_it = 10

# solve full finite element problem
# FOM_solution: contains actual numerical solution. If false converged solution may be unreliable.
# FOM_Operators: FEM matrices needed later for ROM. J_S, J_A_x. A= diffusion matrix, B= divergence, C= nonlinear, f= forcing term
# FOM_data= contains mesh/FEM metadata, to plot, compute errors, reconstruct fields, evaluate PINN on FEM dofs, assemble new operators
FOM_solution, FOM_Operators, FOM_data = solver.solve_FOM(p_boundary_info,
                                                u_boundary_info,
                                                mu0=mu0,
                                                mu1=mu1,
                                                newton_tol=tol,
                                                max_iterations=max_it,
                                                plot_solution=visualize)
# TRAINING PARAMETERS
# Set POD parameters
np.random.seed(26)
# Number of snapshots used to build POD
snapshot_num = 100
# Maximum number of POD retained
N_max = 100
# high mu0 smoother more diffusive solution. Lower convection more important and nonlinear problem harder
mu0_range = [0.1, 10.]
mu1_range = [1., 3.]

P = np.array([mu0_range, mu1_range])
training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(snapshot_num, P.shape[0]))
"""
Create random parameter samples (mu0,mu1), for each parameter pair the project solves FOM and stores solution, SNAPSHOT.
Snapshot matrix S = [ sol at mu^1,
                        sol at mu^2,...]
Each column is a full FEM solution. Expensive because requires many FOM solves. Offline phase
"""
# Offline
# POD purpose is to find a low-dimensional basis. Instead of representing solution with thous/mill of FEM DoF, POD approximates it as u(mu)= V a(mu). where V POD basis matrix and a(mu) reduced coefficients.
# compute snapshots S=[u(mu1), u(mu2), ... u(mu_n)], then performs eigenvalue/SVD analysis. Large eigenvalues are important modes. Basis built from most important modes N_max = at most 100 POD modes are retained.
# Good POD behavior = eigenvalues decay fast, if os small basis is enough
rom = ROM_Methods(FOM_solution,FOM_Operators,FOM_data,training_set=training_set)
reduced_elements = rom.reduce(solver,p_boundary_info,u_boundary_info,tol=tol,N_max=N_max)

"""
Offline phase
Expensive, done once:Solve many FOM snapshots. Build POD basis. Project matrices. Precompute nonlinear reduced tensors. Save reduced model.

Online phase
Cheap, done for new parameters:Given new μ. Solve small reduced Newton system. Reconstruct approximate full solution.

The project saves the model here: reduced_model_path = file_path + "/Models/reduced_model.pkl". Then PODGalerkin_online.py can reload it without recomputing all snapshots.
"""

#---------------------------------------------------------------------------------#

if method == 'PODGalerkin' or method == 'all':
    """
    uses POD basis to project FOM equations onto a reduced space. Full model J(u)u=f. Reduced model V^T J(Va) V a=V^T f.Unknown is a not the full FEM solution.
    """
    # Evaluate the model performance
    metrics_evaluator = ROMPerformanceEvaluator(
        solver=solver,
        rom=rom,
        reduced_data=reduced_elements,
        p_boundary_info=p_boundary_info,
        u_boundary_info=u_boundary_info,
        fom_reference_solution=FOM_solution,
        parameter_ranges=P,
        newton_tol=tol,
        max_iterations=max_it,
        method_name="POD-Galerkin",
        results_prefix="fom_vs_pod_galerkin"
    )

    metrics_results = metrics_evaluator.evaluate(n_test=10, seed=123, results_folder="./Results/POD_Galerkin")

    # Save Reduced Model to run it online in -> "PODGalerkin_online.py"
    rom.save_reduced_model(reduced_elements,reduced_model_path,metadata={
                                                                        "snapshot_num": snapshot_num,
                                                                        "mu0_range": mu0_range,
                                                                        "mu1_range": mu1_range,
                                                                        "tol": tol,
                                                                        "N_max": N_max
                                                                        }   
                                                                        )
    # Online -> test to see plot
    tol = 1.0e-6
    max_it = 10
    mu0 = 1.0
    mu1 = 2.0

    print("\n[Online] Solving POD-Galerkin ROM")
    print(f"[Online] mu0={mu0}, mu1={mu1}")

    rom_sol = rom.solve_POD_Galerkin(
        reduced_elements,
        mu0=mu0,
        mu1=mu1,
        newton_tol=tol,
        max_iterations=max_it,
        plot_solution=visualize
    )

    print("\n[Online] Completed")
    print(f"[Online] converged={rom_sol['converged']}")
    print(f"[Online] iterations={rom_sol['iterations']}")
    print(f"[Online] relative_increment={rom_sol['relative_increment']:.3e}")
    print(f"[Online] coefficients shape={rom_sol['coefficients'].shape}")
    print(f"[Online] reconstructed solution shape={rom_sol['u'].shape}")


    # Split reconstructed ROM solution
    # Reduced solution reconstructed as u_rom = V @ coefficients and then split into u_x_rom, u_y_rom and p_rom

    speed_n_dofs = reduced_elements["speed_n_dofs"]

    u_rom = rom_sol["u"]
    u_x_rom = u_rom[0:speed_n_dofs]
    u_y_rom = u_rom[speed_n_dofs:2 * speed_n_dofs]
    p_rom = u_rom[2 * speed_n_dofs:]

    print("\n[Online] Solution components")
    print(f"u_x_rom shape={u_x_rom.shape}")
    print(f"u_y_rom shape={u_y_rom.shape}")
    print(f"p_rom shape={p_rom.shape}")
    print()
    print('='*100)

if method == 'PODNN' or method == 'all':
    """
    It still uses the POD basis, but instead of solving reduced equations, it trains a neural network.
    The neural network learns:(μ0, μ1) → POD coefficients. So:input: μ0, μ1 and output: a1, a2, ..., aN. Then reconstruction is: u(μ) ≈ V a_NN(μ)
    """
    # Offline PODNN training
    podnn = ROM_NN_Methods(
        fom_data = FOM_data,
        reduced_data=reduced_elements,
        training_set=training_set
    )

    podnn_data = podnn.train(
        hidden_layers=(128, 128, 64),
        epochs=5000,
        learning_rate=1.0e-3,
        weight_decay=1.0e-8,
        batch_size=None,
        seed=26,
        print_every=500
    )

    # Evaluate PODNN performance
    podnn_metrics_evaluator = ROMPerformanceEvaluator(
        solver=solver,
        rom=podnn,
        reduced_data=reduced_elements,
        p_boundary_info=p_boundary_info,
        u_boundary_info=u_boundary_info,
        fom_reference_solution=FOM_solution,
        parameter_ranges=P,
        newton_tol=tol,
        max_iterations=max_it,
        reduced_solver=lambda mu0_test, mu1_test: podnn.solve_PODNN(
            mu0=mu0_test,
            mu1=mu1_test,
            podnn_data=podnn_data,
            plot_solution=False
        ),
        method_name="PODNN",
        results_prefix="fom_vs_podnn"
    )

    podnn_metrics_results = podnn_metrics_evaluator.evaluate(
        n_test=10,
        seed=123,
        results_folder="./Results/PODNN"
    )

    # Save Reduced Model to run it online in -> "PODNN_online.py"
    podnn.save_podnn_model(
        podnn_data=podnn_data,
        export_path=podnn_model_path,
        metadata={
            "snapshot_num": snapshot_num,
            "mu0_range": mu0_range,
            "mu1_range": mu1_range,
            "tol": tol,
            "N_max": N_max,
            "hidden_layers": (128, 128, 64),
            "epochs": 5000,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-8
        }
    )

    # Online PODNN evaluation
    mu0 = 1.0
    mu1 = 2.0

    print("\n[Online] Solving PODNN ROM")
    print(f"[Online] mu0={mu0}, mu1={mu1}")

    podnn_sol = podnn.solve_PODNN(
        mu0=mu0,
        mu1=mu1,
        podnn_data=podnn_data,
        plot_solution = visualize
    )

    print("\n[Online] Completed")
    print(f"[Online] converged={podnn_sol['converged']}")
    print(f"[Online] coefficients shape={podnn_sol['coefficients'].shape}")
    print(f"[Online] reconstructed solution shape={podnn_sol['u'].shape}")

    speed_n_dofs = reduced_elements["speed_n_dofs"]

    u_podnn = podnn_sol["u"]
    u_x_podnn = u_podnn[0:speed_n_dofs]
    u_y_podnn = u_podnn[speed_n_dofs:2 * speed_n_dofs]
    p_podnn = u_podnn[2 * speed_n_dofs:]

    print("\n[Online] Solution components")
    print(f"u_x_podnn shape={u_x_podnn.shape}")
    print(f"u_y_podnn shape={u_y_podnn.shape}")
    print(f"p_podnn shape={p_podnn.shape}")
    print()
    print('='*100)

"""
POD-Galerkin
Uses physics equations. Solves reduced nonlinear system. Usually more physically reliable. Still requires Newton iterations
PODNN
Uses data only. Learns parameter → coefficients. Very fast online. Accuracy depends heavily on training data. Can generalize poorly outside training range

So online: POD-Galerkin: solve small system. PODNN: one neural network forward pass, usually faster online but POD-Galerkin may be more stable/accurate.
"""

if method == 'PINN' or method == 'all':
    """
    A PINN tries to learn the solution directly: (x, y, μ0, μ1) → (u_x, u_y, p). So the input is spatial coordinates plus parameters.
The network output is: velocity and pressure
The loss contains physics terms:PDE residual loss, divergence loss, boundary condition loss, pressure anchor loss. These balance different parts of the loss.
    """
    # Offline PINN training
    pinn = PINN_Methods(
        mu0_range=mu0_range,
        mu1_range=mu1_range,
        fom_data=FOM_data
    )

    pinn_model = pinn.build_network(
        width=128,
        latent_dim=64,
        n_residual_blocks=4
    )

    if os.path.exists(pinn_best_lambdas_path):
        with open(pinn_best_lambdas_path, "r") as file:
            pinn_lambdas = json.load(file)

        print("=" * 100)
        print(f"[PINN] Loaded tuned lambdas from: {pinn_best_lambdas_path}")
        print(f"[PINN] lambdas={pinn_lambdas}")
        print("=" * 100)
    else:
        pinn_lambdas = {
            "lambda_pde": 1.0,
            "lambda_divergence": 5.0,
            "lambda_boundary": 0.0,
            "lambda_pressure_anchor": 0.1
            "lambda_data":0.0
        }
        # your network already imposes u= 0 on the boundary using the factor x(1-x)y(1-y), so lambda_boundary = 10 is redundant and too strong.

        print("=" * 100)
        print(f"[PINN] Tuned lambda file not found: {pinn_best_lambdas_path}")
        print("[PINN] Using default lambdas")
        print(f"[PINN] lambdas={pinn_lambdas}")
        print("=" * 100)

    pinn_model, pinn_history = pinn.train(
        model=pinn_model,
        n_epochs=10000, #<- len(training)
        n_interior=4096,
        n_boundary=1024,
        learning_rate=5.0e-4, #more stable
        weight_decay=1.0e-8,
        lambda_pde=pinn_lambdas["lambda_pde"],
        lambda_divergence=pinn_lambdas["lambda_divergence"],
        lambda_boundary=pinn_lambdas["lambda_boundary"],
        lambda_pressure_anchor=pinn_lambdas["lambda_pressure_anchor"],
        lambda_data=pinn_lambdas.get("lambda_data", 0.0),
        print_every=500
    )
    

    pinn_reduced_data = {
        "speed_n_dofs": FOM_solution["speed_n_dofs"],
        "model_dofs": sum(parameter.numel() for parameter in pinn_model.parameters())
    }

    # Evaluate PINN performance
    pinn_metrics_evaluator = ROMPerformanceEvaluator(
        solver=solver,
        rom=pinn,
        reduced_data=pinn_reduced_data,
        p_boundary_info=p_boundary_info,
        u_boundary_info=u_boundary_info,
        fom_reference_solution=FOM_solution,
        parameter_ranges=P,
        newton_tol=tol,
        max_iterations=max_it,
        reduced_solver=lambda mu0_test, 
        mu1_test: pinn.solve_PINN_on_FEM_dofs(
                                            model=pinn_model,
                                            mu0=mu0_test,
                                            mu1=mu1_test,
                                            plot_solution=False
                                            ),
        method_name="PINN",
        results_prefix="fom_vs_pinn"
    )

    pinn_metrics_results = pinn_metrics_evaluator.evaluate(
        n_test=10,
        seed=123,
        results_folder="./Results/PINN"
    )

    # Save PINN model to run it online in -> "PINN_online.py"
    pinn.save_PINN_model(
        model=pinn_model,
        export_path=pinn_model_path,
        metadata={
            "mu0_range": mu0_range,
            "mu1_range": mu1_range,
            "width": 128,
            "latent_dim": 64,
            "n_residual_blocks": 4,
            "epochs": 10000,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-8,
            "lambda_pde": pinn_lambdas["lambda_pde"],
            "lambda_divergence": pinn_lambdas["lambda_divergence"],
            "lambda_boundary": pinn_lambdas["lambda_boundary"],
            "lambda_pressure_anchor": pinn_lambdas["lambda_pressure_anchor"],
            "lambda_data": pinn_lambdas.get("lambda_data", 0.0),
            "lambda_source": pinn_best_lambdas_path if os.path.exists(pinn_best_lambdas_path) else "default"
        }
    )

    # Online PINN evaluation
    mu0 = 1.0
    mu1 = 2.0

    print("\n[Online] Solving PINN")
    print(f"[Online] mu0={mu0}, mu1={mu1}")

    pinn_sol = pinn.solve_PINN_on_FEM_dofs(
        model=pinn_model,
        mu0=mu0,
        mu1=mu1,
        plot_solution=visualize
    )

    print("\n[Online] Completed")
    print(f"[Online] converged={pinn_sol['converged']}")
    print(f"[Online] iterations={pinn_sol['iterations']}")
    print(f"[Online] relative_increment={pinn_sol['relative_increment']}")
    print(f"[Online] reconstructed solution shape={pinn_sol['u'].shape}")

    print("\n[Online] Solution components")
    print(f"u_x_pinn shape={pinn_sol['u_x'].shape}")
    print(f"u_y_pinn shape={pinn_sol['u_y'].shape}")
    print(f"p_pinn shape={pinn_sol['p'].shape}")
    print()
    print('='*100)

if method not in ['PODGalerkin', 'PODNN', 'PINN', 'all']:
    raise ValueError(f"Unknown method: {method}")

"""
ROMPerformanceEvaluator
This compares:
FOM vs POD-Galerkin
FOM vs PODNN
FOM vs PINN
"""
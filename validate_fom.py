import os
import numpy as np
import pandas as pd

import Assembler as assembler_module
from Discretization import Discretize
from Solver import Solver
from other_utilities import export_folder
from pypolydim import polydim


def exact_u_x(mu1):
    def fun(x, y, z):
        A = mu1
        return 0.5 * A * np.sin(np.pi * x)**2 * np.sin(np.pi * y) * np.cos(np.pi * y)
    return fun


def exact_u_y(mu1):
    def fun(x, y, z):
        A = mu1
        return -0.5 * A * np.sin(np.pi * y)**2 * np.sin(np.pi * x) * np.cos(np.pi * x)
    return fun


def exact_p(mu1):
    def fun(x, y, z):
        A = mu1
        return A * np.sin(np.pi * x) * np.sin(np.pi * y)
    return fun


def relative_l2_error(fem_data, numerical, strong, reference_element_data, dofs_data, exact_function):
    err = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
        fem_data["geometry_utilities"],
        fem_data["mesh"],
        fem_data["mesh_geometric_data"],
        dofs_data,
        reference_element_data,
        numerical,
        strong,
        exact_function
    )

    return err.error_l2 / err.numeric_norm_l2


def main():
    out_dir = "./Results/FOM_validation"
    os.makedirs(out_dir, exist_ok=True)

    # IMPORTANT: activate manufactured forcing only inside this script
    assembler_module.VALIDATE_FOM = True

    file_path, mesh_path, solution_path = export_folder("./Export")

    mesh_size = 0.02
    domain_area = 1.0

    vertices = np.array([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    discretizer = Discretize(vertices, mesh_size, domain_area, mesh_path)
    solver = Solver(discretizer, solution_path)

    info_internal, info_dirichlet, info_neumann_none = solver.set_dofs()

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

    test_parameters = [
        (0.5, 1.2),
        (1.0, 2.0),
        (3.0, 2.5),
        (7.0, 1.5),
        (10.0, 3.0),
    ]

    rows = []

    for mu0, mu1 in test_parameters:
        print("=" * 100)
        print(f"[FOM validation] mu0={mu0}, mu1={mu1}")

        sol, ops, fem_data = solver.solve_FOM(
            p_boundary_info=p_boundary_info,
            u_boundary_info=u_boundary_info,
            mu0=mu0,
            mu1=mu1,
            newton_tol=1.0e-6,
            max_iterations=20,
            plot_solution=False
        )

        err_ux = relative_l2_error(
            fem_data=fem_data,
            numerical=sol["u_x"],
            strong=fem_data["u_x_strong"],
            reference_element_data=fem_data["speed_reference_element_data"],
            dofs_data=fem_data["speed_dofs_data"],
            exact_function=exact_u_x(mu1)
        )

        err_uy = relative_l2_error(
            fem_data=fem_data,
            numerical=sol["u_y"],
            strong=fem_data["u_y_strong"],
            reference_element_data=fem_data["speed_reference_element_data"],
            dofs_data=fem_data["speed_dofs_data"],
            exact_function=exact_u_y(mu1)
        )

        err_p = relative_l2_error(
            fem_data=fem_data,
            numerical=sol["p"],
            strong=fem_data["p_strong"],
            reference_element_data=fem_data["pressure_reference_element_data"],
            dofs_data=fem_data["pressure_dofs_data"],
            exact_function=exact_p(mu1)
        )

        err_velocity = np.sqrt(err_ux**2 + err_uy**2)

        rows.append({
            "mu0": mu0,
            "mu1": mu1,
            "err_velocity": err_velocity,
            "err_u_x": err_ux,
            "err_u_y": err_uy,
            "err_p": err_p,
            "newton_iterations": sol["iterations"],
            "relative_increment": sol["relative_increment"],
            "converged": sol["converged"],
            "tot_dofs": sol["tot_dofs"]
        })

        print(f"err_u_x      = {err_ux:.3e}")
        print(f"err_u_y      = {err_uy:.3e}")
        print(f"err_velocity = {err_velocity:.3e}")
        print(f"err_p        = {err_p:.3e}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "fom_validation.csv")
    report_path = os.path.join(out_dir, "fom_validation_report.txt")

    df.to_csv(csv_path, index=False)

    with open(report_path, "w") as f:
        f.write("FOM validation with manufactured Navier-Stokes solution\n")
        f.write("=" * 65 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("The exact manufactured solution is:\n")
        f.write("u_x = 0.5 * mu1 * sin(pi*x)^2 * sin(pi*y)*cos(pi*y)\n")
        f.write("u_y = -0.5 * mu1 * sin(pi*y)^2 * sin(pi*x)*cos(pi*x)\n")
        f.write("p   = mu1 * sin(pi*x)*sin(pi*y)\n")

    assembler_module.VALIDATE_FOM = False

    print("\nSaved validation results:")
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
import numpy as np
import scipy
import pickle
import os
import time
import csv
import torch
import torch.nn as nn

from pypolydim import polydim
from other_utilities import make_np_sparse,plot_FOM_solution


class ROM_Methods(object):
    def __init__(self,fom_sol,operators,fom_data,training_set):
        """
        Initialize the ROM utility class.

        The object stores the FOM solution, assembled FOM operators, FEM/FOM
        metadata and the parameter training set used to build snapshots and
        reduced operators.
        """
        self.training_set = training_set
        self.operators = operators
        self.fom_sol = fom_sol
        self.fom_data = fom_data

    def save_reduced_model(self, reduced_data, export_path, metadata=None):
        """
        Save the POD-Galerkin reduced model to disk.

        The saved object contains the reduced operators, basis functions,
        precomputed tensors and optional metadata needed to run the online
        POD-Galerkin phase without rebuilding the full offline pipeline.
        """
        reduced_model = {
            "reduced_data": reduced_data,
            "metadata": metadata if metadata is not None else {}
        }

        with open(export_path, "wb") as file:
            pickle.dump(reduced_model, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[ROM] Reduced model saved to: {export_path}")

    @staticmethod
    def load_reduced_model(export_path):
        """
        Load a previously saved POD-Galerkin reduced model.

        The method checks that the file exists and is not empty before loading
        the serialized reduced model dictionary from disk.
        """

        if not os.path.exists(export_path):
            raise FileNotFoundError(
                f"Reduced model file not found: {export_path}. Run the offline script first."
            )

        if os.path.getsize(export_path) == 0:
            raise ValueError(
                f"Reduced model file is empty: {export_path}. Delete it and rerun the offline script to regenerate it."
            )

        with open(export_path, "rb") as file:
            reduced_model = pickle.load(file)

        print()
        print('='*100)
        print(f"[ROM] Reduced model loaded from: {export_path}")

        return reduced_model

    def solve_POD_Galerkin(self, reduced_data, mu0, mu1, newton_tol=1e-6, max_iterations=10,plot_solution=False):
        """
        Online POD-Galerkin solver.

        This method intentionally works only with reduced quantities.
        The nonlinear reduced convection operators must be precomputed offline
        and stored in reduced_data before calling this method.
        """

        V = reduced_data["V"]
        J_A_N = reduced_data["J_A_N"]
        J_B_N = reduced_data["J_B_N"]
        C_jacobian_tensor = reduced_data["C_jacobian_tensor"]
        C_residual_tensor = reduced_data["C_residual_tensor"]

        f_N = self.evaluate_reduced_rhs_nn(mu1, reduced_data)

        n_red = V.shape[1]
        a_k = np.zeros(n_red)

        increment_norm = 1.0
        solution_norm = 1.0
        num_iteration = 1

        J_S_N = mu0 * J_A_N - J_B_N

        while num_iteration < max_iterations and increment_norm > newton_tol * solution_norm:
            C_residual_N = np.einsum("ijk,j,k->i", C_residual_tensor, a_k, a_k)
            J_C_N = np.einsum("ijk,k->ij", C_jacobian_tensor, a_k)

            rhs_N = f_N - C_residual_N - J_S_N @ a_k
            da = np.linalg.solve(J_S_N + J_C_N, rhs_N)

            a_k = a_k + da

            increment_norm = np.linalg.norm(da)
            solution_norm = max(np.linalg.norm(a_k), 1e-14)

            print(
                f"[ROM Newton] iter {num_iteration:02d}/{max_iterations} | "
                f"||da||/||a||={increment_norm / solution_norm:.3e} | "
                f"tol={newton_tol:.1e}"
            )

            num_iteration += 1

        U_N = V @ a_k


        print("\n" + "-" * 100)
        print("[Online ROM] Solving POD-Galerkin reduced problem")
        print(f"[Online ROM] mu0={mu0}, mu1={mu1}, newton_tol={newton_tol}, max_iterations={max_iterations}")
        print("\n[Online ROM] Completed")
        print(f"[Online ROM] converged={increment_norm <= newton_tol * solution_norm}")
        print(f"[Online ROM] iterations={num_iteration - 1}")
        print(f"[Online ROM] relative_increment={increment_norm / solution_norm}")
        print(f"[Online ROM] reduced_coefficients_shape={a_k.shape}")
        print(f"[Online ROM] reconstructed_solution_shape={U_N.shape}")
        print("-" * 100)

        if plot_solution:
            self.plot_ROM_solution(reduced_data,U_N,self.fom_data['vtk_utilities'])

        return {
            "u": U_N,
            "coefficients": a_k,
            "iterations": num_iteration - 1,
            "relative_increment": increment_norm / solution_norm,
            "converged": increment_norm <= newton_tol * solution_norm
        }
    
    def plot_ROM_solution(
        self,
        reduced_data,
        U_N,
        vtk_utilities,
        export_solution_path="./Export/Solution/POD_Galerkin",
        plot_path = "./Plots/POD_Galerkin"
        ):
        """
        Plot/export a reconstructed ROM solution.

        This method intentionally remains separate from solve_POD_Galerkin.
        Pure online execution loaded from a reduced_model.pkl does not contain
        FEM mesh/dof objects, because they are not pickle-serializable.
        Therefore plotting is available only when fom_data is present, e.g.
        during the offline/main workflow.
        """
        if self.fom_data is None:
            raise ValueError(
                "plot_ROM_solution requires fom_data. It cannot be used with a pure reduced model loaded from pickle."
            )

        if not os.path.exists(export_solution_path):
            os.makedirs(export_solution_path)

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        speed_n_dofs = reduced_data["speed_n_dofs"]
        u_x_rom = U_N[0:speed_n_dofs]
        u_y_rom = U_N[speed_n_dofs:2 * speed_n_dofs]
        p_rom = U_N[2 * speed_n_dofs:]

        plot_FOM_solution(
            mesh=self.fom_data["mesh"],
            speed_dofs_data=self.fom_data["speed_dofs_data"],
            u_x_numeric=u_x_rom,
            u_x_strong=self.fom_data["u_x_strong"],
            u_y_numeric=u_y_rom,
            u_y_strong=self.fom_data["u_y_strong"],
            pressure_dofs_data=self.fom_data["pressure_dofs_data"],
            p_numeric=p_rom,
            p_strong=self.fom_data["p_strong"],
            vtk_utilities=vtk_utilities,
            export_solution_path=export_solution_path,
            plot_path= plot_path,
            method='_PODGalerkin'
        )
    
    def reduce(self,
                solver,
                p_boundary_info,
                u_boundary_info,
                tol=1e-6,
                N_max=20
                ):
        """
        Execute the offline POD-Galerkin reduction pipeline.

        The method builds velocity, pressure and supremizer snapshots, performs
        POD truncation, assembles the global reduced basis, projects the linear
        FOM operators, precomputes the nonlinear convective tensors and trains a
        reduced RHS surrogate for the online phase.
        """
        
        A_op = self.operators['A_operator']
        B_x_op = self.operators["B_x_operator"]
        B_y_op = self.operators["B_y_operator"]
        J_A = self.operators["J_A"]
        J_B = self.operators["J_B"]

        speed_n_dofs = self.fom_sol['speed_n_dofs']
        pressure_n_dofs = self.fom_sol['pressure_n_dofs']

        # Inner product = X1 + X2 (Grad-Grad)
        inner_product_u,B_1,B_2 = self.assemble_supremizer_matricies(A_op,
                                                                     B_x_op,
                                                                     B_y_op,
                                                                     speed_n_dofs,
                                                                     pressure_n_dofs)
        snapshot_matrix_u, snapshot_matrix_s,\
              snapshot_matrix_p, C_u, C_s, C_p = self.create_POD_matricies( solver,
                                                                            p_boundary_info,
                                                                            u_boundary_info,
                                                                            speed_n_dofs,
                                                                            inner_product_u,
                                                                            B_1,
                                                                            B_2,
                                                                            tol=1e-6,
                                                                            N_max=20
                                                                            )
        
        N_u, eigs_u = self.eig_analysis(C_u, N_max=N_max, tol=tol)
        N_s, eigs_s = self.eig_analysis(C_s, N_max=N_max, tol=tol)
        N_p, eigs_p = self.eig_analysis(C_p, N_max=N_max, tol=tol)

        basis_functions_u = self.create_basis_functions_matrix(N_u, snapshot_matrix_u, eigs_u, inner_product=inner_product_u)
        basis_functions_s = self.create_basis_functions_matrix(N_s, snapshot_matrix_s, eigs_s, inner_product=inner_product_u)
        basis_functions_p = self.create_basis_functions_matrix(N_p, snapshot_matrix_p, eigs_p)

        global_basis_function_matrix = np.zeros((basis_functions_u.shape[0] + basis_functions_p.shape[0],N_u + N_s + N_p))
        global_basis_function_matrix[0:basis_functions_u.shape[0], 0:N_u] = basis_functions_u
        global_basis_function_matrix[0:basis_functions_u.shape[0], N_u : N_u + N_s] = basis_functions_s
        global_basis_function_matrix[basis_functions_u.shape[0]:, N_u + N_s:] = basis_functions_p

        global_basis_function_matrix_no_sup = np.zeros((basis_functions_u.shape[0] + basis_functions_p.shape[0],N_u + N_p))
        global_basis_function_matrix_no_sup[0:basis_functions_u.shape[0], 0:N_u] = basis_functions_u
        global_basis_function_matrix_no_sup[basis_functions_u.shape[0]:, N_u:] = basis_functions_p

        V = global_basis_function_matrix
        
        J_A_N = self.assemble_reduced_matrix(V, J_A)
        J_B_N = self.assemble_reduced_matrix(V, J_B)

        C_residual_tensor, C_jacobian_tensor = self.precompute_reduced_convective_tensors(solver,V)
        rhs_nn = self.train_reduced_rhs_nn(V)

        print("\n" + "-" * 100)
        print("[Offline ROM] Starting POD basis construction and reduced operator assembly")
        print("-" * 100)

        return {
            "V": V,
            "J_A_N": J_A_N,
            "J_B_N": J_B_N,
            "snapshot_matrix_u": snapshot_matrix_u,
            "snapshot_matrix_s": snapshot_matrix_s,
            "snapshot_matrix_p": snapshot_matrix_p,
            "N_u": N_u,
            "N_s": N_s,
            "N_p": N_p,
            "basis_functions_u": basis_functions_u,
            "basis_functions_s": basis_functions_s,
            "basis_functions_p": basis_functions_p,
            "C_jacobian_tensor": C_jacobian_tensor,
            "C_residual_tensor": C_residual_tensor,
            "speed_n_dofs": speed_n_dofs,
            "rhs_nn": rhs_nn
            }

    def precompute_reduced_convective_tensors(self, solver, V):
        """
        Offline precomputation of the reduced quadratic convection term.

        This method uses the FEM data already generated by the FOM solver and
        stored in self.fom_data.

        The residual tensor satisfies
            C_N(a)_i = sum_{j,k} C_residual_tensor[i,j,k] a_j a_k.

        The Jacobian tensor satisfies
            J_C_N(a)_{i,j} = sum_k C_jacobian_tensor[i,j,k] a_k.
        """

        geometry_utilities = self.fom_data["geometry_utilities"]
        mesh = self.fom_data["mesh"]
        mesh_geometric_data = self.fom_data["mesh_geometric_data"]
        speed_dofs_data = self.fom_data["speed_dofs_data"]
        speed_reference_element_data = self.fom_data["speed_reference_element_data"]
        u_x_strong = self.fom_data["u_x_strong"]
        u_y_strong = self.fom_data["u_y_strong"]

        speed_n_dofs = self.fom_sol["speed_n_dofs"]
        pressure_n_dofs = self.fom_sol["pressure_n_dofs"]

        n_red = V.shape[1]
        C_residual_tensor = np.zeros((n_red, n_red, n_red))

        def project_convective_rhs(full_vector):
            u_x_numeric = full_vector[0:speed_n_dofs]
            u_y_numeric = full_vector[speed_n_dofs:2 * speed_n_dofs]

            c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(
                geometry_utilities,
                mesh,
                mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                u_x_numeric,
                u_y_numeric,
                u_x_strong,
                u_y_strong
            )

            f_C = np.concatenate([
                c_operator.convective_rhs,
                np.zeros(pressure_n_dofs)
            ])

            return self.assemble_reduced_vector(V, f_C)

        print("Precomputing reduced convective tensors...")

        single_terms = []
        for j in range(n_red):
            single_terms.append(project_convective_rhs(V[:, j]))

        for j in range(n_red):
            C_residual_tensor[:, j, j] = single_terms[j]

            for k in range(j + 1, n_red):
                mixed_term = 0.5 * (
                    project_convective_rhs(V[:, j] + V[:, k])
                    - single_terms[j]
                    - single_terms[k]
                )

                C_residual_tensor[:, j, k] = mixed_term
                C_residual_tensor[:, k, j] = mixed_term

        C_jacobian_tensor = C_residual_tensor + np.swapaxes(C_residual_tensor, 1, 2)

        return C_residual_tensor, C_jacobian_tensor

    def train_reduced_rhs_nn(self, V, hidden_neurons=64, ridge=1e-10, seed=26):
        """
        Offline training of a small one-hidden-layer neural model for
        mu1 -> f_N(mu1).

        This avoids assembling the FEM right-hand side during the online phase.
        The hidden layer is fixed randomly and only the output weights are fitted
        by ridge regression. This is an Extreme Learning Machine, i.e. a shallow NN.
        """
        if self.training_set is None:
            raise ValueError("training_set is required to train the reduced RHS neural model.")

        mu1_values = np.asarray(self.training_set)[:, 1]
        mu1_values = np.unique(mu1_values)

        x_mean = np.mean(mu1_values)
        x_std = np.std(mu1_values)
        if x_std == 0.0:
            x_std = 1.0

        rng = np.random.default_rng(seed)
        W_in = rng.normal(loc=0.0, scale=1.0, size=(1, hidden_neurons))
        b_in = rng.normal(loc=0.0, scale=1.0, size=(hidden_neurons,))

        X = ((mu1_values - x_mean) / x_std).reshape(-1, 1)
        H = np.tanh(X @ W_in + b_in)
        H_aug = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)

        Y = []
        for mu1 in mu1_values:
            Y.append(self.assemble_reduced_rhs_from_fem(mu1, V))
        Y = np.asarray(Y)

        lhs = H_aug.T @ H_aug + ridge * np.eye(H_aug.shape[1])
        rhs = H_aug.T @ Y
        W_out = np.linalg.solve(lhs, rhs)

        training_prediction = H_aug @ W_out
        training_error = np.linalg.norm(training_prediction - Y) / max(np.linalg.norm(Y), 1e-14)

        print(f"[Offline ROM] RHS neural model trained | samples={len(mu1_values)} | hidden={hidden_neurons} | rel_train_error={training_error:.3e}")

        return {
            "x_mean": x_mean,
            "x_std": x_std,
            "W_in": W_in,
            "b_in": b_in,
            "W_out": W_out,
            "hidden_neurons": hidden_neurons,
            "ridge": ridge,
            "training_error": training_error
        }

    def evaluate_reduced_rhs_nn(self, mu1, reduced_data):
        """
        Evaluate the neural surrogate for the reduced right-hand side.

        Given the parameter `mu1`, the method uses the stored shallow neural
        model to approximate the reduced forcing vector without assembling the
        full FEM right-hand side during the online phase.
        """
        rhs_nn = reduced_data["rhs_nn"]

        x = np.array([[(mu1 - rhs_nn["x_mean"]) / rhs_nn["x_std"]]])
        H = np.tanh(x @ rhs_nn["W_in"] + rhs_nn["b_in"])
        H_aug = np.concatenate([H, np.ones((1, 1))], axis=1)

        return (H_aug @ rhs_nn["W_out"]).ravel()

    def assemble_reduced_rhs_from_fem(self, mu1, V):
        """
        Assemble and project the FEM source term for a given `mu1`.

        This method is used offline to generate training targets for the reduced
        RHS neural surrogate by assembling the full source term and projecting it
        onto the reduced basis.
        """
        f_x_function, f_y_function = self.fom_data["assembler"].build_f_components(mu1)

        f_x = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            self.fom_data["speed_reference_element_data"],
            f_x_function
        )

        f_y = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            self.fom_data["speed_reference_element_data"],
            f_y_function
        )

        f_S = np.concatenate([
            f_x,
            f_y,
            np.zeros(self.fom_sol["pressure_n_dofs"])
        ])

        return self.assemble_reduced_vector(V, f_S)

    def create_POD_matricies(self,
                                solver,
                                p_boundary_info,
                                u_boundary_info,
                                speed_n_dofs,
                                inner_product_u,
                                B_1,
                                B_2,
                                tol=1e-6,
                                N_max=20
                                ):
        """
        Generate FOM snapshots and POD correlation matrices.

        For every training parameter, the FOM is solved and the resulting global
        solution is split into velocity and pressure snapshots. Supremizer
        snapshots are also computed to enrich the reduced velocity space for the
        saddle-point structure of the Navier-Stokes problem.
        """
        
        snapshot_matrix_u = []
        snapshot_matrix_s = []
        snapshot_matrix_p = []

        print("Performing FOM training...")

        n_train = len(self.training_set)

        for it, (mu0, mu1) in enumerate(self.training_set):
            print(
                f"[Training] sample {it + 1:03d}/{n_train} | "
                f"mu0={mu0:.4g}, mu1={mu1:.4g}"
            )

            fom_result = solver.solve_FOM(
                p_boundary_info=p_boundary_info,
                u_boundary_info=u_boundary_info,
                mu0=mu0,
                mu1=mu1,
                newton_tol=tol,
                max_iterations=N_max
            )

            if isinstance(fom_result, tuple):
                solution = fom_result[0]
            else:
                solution = fom_result

            snapshot = solution["u"]

            snapshot_u = snapshot[0:2 * speed_n_dofs]
            snapshot_p = snapshot[2 * speed_n_dofs:]
            snapshot_s = scipy.sparse.linalg.spsolve(inner_product_u,np.transpose(B_1 + B_2) @ snapshot_p)

            snapshot_matrix_u.append(snapshot_u)
            snapshot_matrix_p.append(snapshot_p)
            snapshot_matrix_s.append(snapshot_s)

        snapshot_matrix_u = np.array(snapshot_matrix_u)
        snapshot_matrix_p = np.array(snapshot_matrix_p)
        snapshot_matrix_s = np.array(snapshot_matrix_s)

        C_u = snapshot_matrix_u @ inner_product_u @ np.transpose(snapshot_matrix_u)
        C_s = snapshot_matrix_s @ inner_product_u @ np.transpose(snapshot_matrix_s)
        C_p = snapshot_matrix_p @ np.transpose(snapshot_matrix_p)

        return snapshot_matrix_u, snapshot_matrix_s, snapshot_matrix_p, C_u, C_s, C_p

    def assemble_supremizer_matricies(self,
                                      A_op,
                                      B_x_op,
                                      B_y_op,
                                      speed_n_dofs,
                                      pressure_n_dofs):
        """
        Assemble the inner-product and divergence matrices used for supremizers.

        The method builds the velocity inner-product matrix and the pressure-to-
        velocity divergence coupling matrices needed to compute supremizer
        snapshots for pressure stability in the reduced problem.
        """
        
        X_1 = make_np_sparse(A_op.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [0, 0])
        X_2 = make_np_sparse(A_op.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [speed_n_dofs, speed_n_dofs])
        B_1 = make_np_sparse(B_x_op.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, 0])
        B_2 = make_np_sparse(B_y_op.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, speed_n_dofs])

        return X_1 + X_2, B_1, B_2
    
    def assemble_reduced_matrix(self, basis, fom_matrix):
        """
        Project a FOM matrix onto the reduced basis.

        The returned matrix is `basis.T @ fom_matrix @ basis` and is used in the
        reduced Galerkin system.
        """
        return basis.T @ fom_matrix @ basis

    def assemble_reduced_vector(self, basis, fom_vector):
        """
        Project a FOM vector onto the reduced basis.

        The returned vector is `basis.T @ fom_vector` and represents the reduced
        coordinates of a full-order vector or residual contribution.
        """
        return basis.T @ fom_vector
    
    def eig_analysis(self, C, N_max=None, tol=1e-9):
        """
        Perform POD eigenvalue analysis on a snapshot correlation matrix.

        The method sorts eigenvalues in descending order, computes the retained
        energy and selects the number of modes according to the tolerance and the
        optional maximum dimension.
        """
        eigenvalues, eigenvectors_matrix = np.linalg.eigh(C)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order].real
        eigenvectors_matrix = eigenvectors_matrix[:, order].real

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_energy = np.sum(eigenvalues)

        if total_energy <= 0.0:
            raise ValueError("POD covariance matrix has zero total energy.")

        relative_retained_energy = np.cumsum(eigenvalues) / total_energy
        target_energy = 1.0 - tol
        N = np.argmax(relative_retained_energy >= target_energy) + 1

        if N_max is not None:
            N = min(N, N_max)

        eigenvectors = [eigenvectors_matrix[:, i] for i in range(N)]

        return N, eigenvectors
    
    def create_basis_functions_matrix(self, N, snapshot_matrix, eigenvectors, inner_product=None):
        """
        Build normalized POD basis functions from snapshots and eigenvectors.

        Each basis vector is obtained as a linear combination of snapshots and is
        normalized either with the Euclidean norm or with the supplied inner
        product matrix.
        """
        basis_functions = []
        
        for n in range(N):
            eigenvector =  eigenvectors[n]
            basis = np.transpose(snapshot_matrix)@eigenvector
            if inner_product is not None:
                norm = np.sqrt(np.transpose(basis) @ inner_product @ basis) ## metti inner product
            else:
                norm = np.sqrt(np.transpose(basis) @ basis)

            basis /= norm
            basis_functions.append(np.copy(basis))

        basis_function_matrix = np.transpose(np.array(basis_functions))
        
        return basis_function_matrix
    


class ROM_NN_Methods(object):
    def __init__(self, reduced_data, training_set, fom_data):
        """
        Initialize the PODNN utility class.

        The object stores the reduced basis and snapshot data, the parameter
        training set and the optional FOM/FEM metadata needed for plotting during
        the offline workflow.
        """
        self.reduced_data = reduced_data
        self.training_set = np.asarray(training_set)
        self.fom_data = fom_data

    def solve_PODNN(self, mu0, mu1, podnn_data, plot_solution=False):
        """
        Evaluate the PODNN surrogate for a new parameter value.

        The method predicts the POD coefficients with the trained neural network,
        reconstructs the full solution through the reduced basis and optionally
        exports/plots the reconstructed solution when FOM data are available.
        """
        V = self.reduced_data["V"]
        coefficients = self.predict_coefficients(mu0, mu1, podnn_data)
        U_N = V @ coefficients

        if plot_solution:
            self.plot_PODNN_solution(self.reduced_data, U_N, self.fom_data["vtk_utilities"])

        return {
            "u": U_N,
            "coefficients": coefficients,
            "converged": np.nan,
            "iterations": np.nan,
            "relative_increment": np.nan
        }

    def build_global_snapshot_matrix(self):
        """
        Build the global snapshot matrix used for PODNN training.

        The method concatenates velocity and pressure snapshot matrices so that
        each row corresponds to a full-order solution snapshot.
        """
        snapshot_matrix_u = self.reduced_data["snapshot_matrix_u"]
        snapshot_matrix_p = self.reduced_data["snapshot_matrix_p"]

        snapshot_matrix = np.concatenate(
            [snapshot_matrix_u, snapshot_matrix_p],
            axis=1
        )

        return snapshot_matrix

    def compute_pod_coefficients(self):
        """
        Compute POD coefficients associated with the FOM snapshots.

        The method solves a least-squares projection problem to express each full
        snapshot in the POD reduced basis. These coefficients are the supervised
        targets used to train the PODNN model.
        """
        V = self.reduced_data["V"]
        snapshot_matrix = self.build_global_snapshot_matrix()

        coefficients_matrix = np.linalg.lstsq(V,snapshot_matrix.T,rcond=None)[0].T

        return coefficients_matrix

    def build_mlp(self, input_dim, output_dim, hidden_layers):
        """
        Build the multilayer perceptron used by PODNN.

        The network maps normalized parameters to normalized POD coefficients
        using fully connected layers with Tanh activations.
        """
        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.Tanh())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))

        return nn.Sequential(*layers)

    def train(
        self,
        hidden_layers=(128, 128, 64),
        epochs=5000,
        learning_rate=1.0e-3,
        weight_decay=1.0e-8,
        batch_size=None,
        seed=26,
        print_every=500
    ):
        """
        Train a nonlinear MLP for the map
            mu = (mu0, mu1) -> POD coefficients.

        The network is trained with Adam on normalized inputs and normalized POD
        coefficients. This is a stronger PODNN than the previous fixed-random
        shallow model.
        """
        if self.training_set is None:
            raise ValueError("training_set is required to train PODNN.")

        torch.manual_seed(seed)
        np.random.seed(seed)

        X_raw = self.training_set.astype(np.float64)
        Y_raw = self.compute_pod_coefficients().astype(np.float64)

        x_mean = np.mean(X_raw, axis=0)
        x_std = np.std(X_raw, axis=0)
        x_std[x_std == 0.0] = 1.0

        y_mean = np.mean(Y_raw, axis=0)
        y_std = np.std(Y_raw, axis=0)
        y_std[y_std == 0.0] = 1.0

        X = (X_raw - x_mean) / x_std
        Y = (Y_raw - y_mean) / y_std

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        input_dim = X_tensor.shape[1]
        output_dim = Y_tensor.shape[1]

        model = self.build_mlp(input_dim, output_dim, hidden_layers)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        loss_function = nn.MSELoss()

        n_samples = X_tensor.shape[0]
        if batch_size is None or batch_size > n_samples:
            batch_size = n_samples

        model.train()
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                indices = permutation[start:start + batch_size]
                X_batch = X_tensor[indices]
                Y_batch = Y_tensor[indices]

                optimizer.zero_grad()
                Y_prediction = model(X_batch)
                loss = loss_function(Y_prediction, Y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.shape[0]

            epoch_loss /= n_samples

            if print_every is not None and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                print(f"[PODNN] epoch {epoch:05d}/{epochs} | mse={epoch_loss:.3e}")

        model.eval()
        with torch.no_grad():
            Y_prediction = model(X_tensor).cpu().numpy()

        Y_prediction_physical = Y_prediction * y_std + y_mean
        train_error = np.linalg.norm(Y_raw - Y_prediction_physical) / max(np.linalg.norm(Y_raw), 1.0e-14)

        podnn_data = {
            "model_state_dict": model.state_dict(),
            "hidden_layers": tuple(hidden_layers),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_error_coefficients": train_error
        }

        print(
            f"[PODNN] trained | samples={n_samples} | "
            f"hidden_layers={hidden_layers} | "
            f"rel_train_error_coefficients={train_error:.3e}"
        )

        return podnn_data

    def predict_coefficients(self, mu0, mu1, podnn_data):
        """
        Predict POD coefficients for a new parameter value.

        The trained neural network is rebuilt from the saved metadata, the input
        parameter is normalized, and the predicted coefficients are transformed
        back to their physical scale.
        """
        model = self.build_mlp(
            podnn_data["input_dim"],
            podnn_data["output_dim"],
            podnn_data["hidden_layers"]
        )
        model.load_state_dict(podnn_data["model_state_dict"])
        model.eval()

        mu = np.array([[mu0, mu1]], dtype=np.float64)
        X = (mu - podnn_data["x_mean"]) / podnn_data["x_std"]
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            coefficients_normalized = model(X_tensor).cpu().numpy().ravel()

        coefficients = coefficients_normalized * podnn_data["y_std"] + podnn_data["y_mean"]

        return coefficients

    def plot_PODNN_solution(
        self,
        reduced_data,
        U_N,
        vtk_utilities,
        export_solution_path="./Export/Solution/PODNN",
        plot_path="./Plots/PODNN"
        ):
        """
        Plot/export a reconstructed PODNN solution.

        This method intentionally remains separate from solve_PODNN.
        Pure online execution loaded from a podnn_model.pkl does not contain
        FEM mesh/dof objects, because they are not pickle-serializable.
        Therefore plotting is available only when fom_data is present, e.g.
        during the offline/main workflow.
        """
        if self.fom_data is None:
            raise ValueError(
                "plot_PODNN_solution requires fom_data. It cannot be used with a pure PODNN model loaded from pickle."
            )

        if not os.path.exists(export_solution_path):
            os.makedirs(export_solution_path)

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        speed_n_dofs = reduced_data["speed_n_dofs"]
        u_x_podnn = U_N[0:speed_n_dofs]
        u_y_podnn = U_N[speed_n_dofs:2 * speed_n_dofs]
        p_podnn = U_N[2 * speed_n_dofs:]

        plot_FOM_solution(
            mesh=self.fom_data["mesh"],
            speed_dofs_data=self.fom_data["speed_dofs_data"],
            u_x_numeric=u_x_podnn,
            u_x_strong=self.fom_data["u_x_strong"],
            u_y_numeric=u_y_podnn,
            u_y_strong=self.fom_data["u_y_strong"],
            pressure_dofs_data=self.fom_data["pressure_dofs_data"],
            p_numeric=p_podnn,
            p_strong=self.fom_data["p_strong"],
            vtk_utilities=vtk_utilities,
            export_solution_path=export_solution_path,
            plot_path=plot_path,
            method='_PODNN'
        )

    def save_podnn_model(self, podnn_data, export_path, metadata=None):
        """
        Save the trained PODNN model to disk.

        The saved dictionary contains the neural-network weights, normalization
        data, reduced basis information and optional metadata needed for online
        PODNN evaluation.
        """
        podnn_model = {
            "podnn_data": podnn_data,
            "reduced_data": self.reduced_data,
            "metadata": metadata if metadata is not None else {}
        }

        with open(export_path, "wb") as file:
            pickle.dump(podnn_model, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[PODNN] model saved to: {export_path}")

    @staticmethod
    def load_podnn_model(export_path):
        """
        Load a previously saved PODNN model.

        The method checks that the file exists and is not empty before loading
        the serialized PODNN data and reduced model information.
        """
        if not os.path.exists(export_path):
            raise FileNotFoundError(
                f"PODNN model file not found: {export_path}. Run the offline script first."
            )

        if os.path.getsize(export_path) == 0:
            raise ValueError(
                f"PODNN model file is empty: {export_path}. Delete it and regenerate it."
            )

        with open(export_path, "rb") as file:
            podnn_model = pickle.load(file)

        print()
        print("="*100)
        print(f"[PODNN] model loaded from: {export_path}")

        return podnn_model



class ROMPerformanceEvaluator(object):
    def __init__(
        self,
        solver,
        rom,
        reduced_data,
        p_boundary_info,
        u_boundary_info,
        fom_reference_solution,
        parameter_ranges,
        newton_tol=1.0e-6,
        max_iterations=10,
        reduced_solver=None,
        method_name="POD-Galerkin",
        results_prefix="fom_vs_pod_galerkin"
    ):
        """
        Initialize the performance evaluator for FOM-vs-surrogate comparisons.

        The evaluator stores the FOM solver, reduced/surrogate model, boundary
        information, parameter ranges and output naming options used to compute
        errors, timings, speedups and CSV reports.
        """
        self.solver = solver
        self.rom = rom
        self.reduced_data = reduced_data
        self.p_boundary_info = p_boundary_info
        self.u_boundary_info = u_boundary_info
        self.fom_reference_solution = fom_reference_solution
        self.parameter_ranges = np.asarray(parameter_ranges)
        self.newton_tol = newton_tol
        self.max_iterations = max_iterations
        self.reduced_solver = reduced_solver
        self.method_name = method_name
        self.results_prefix = results_prefix

    def solve_reduced_model(self, mu0, mu1):
        """
        Solve or evaluate the selected reduced/surrogate model.

        If a custom `reduced_solver` callback is provided, it is used directly.
        Otherwise, the default POD-Galerkin solver is called.
        """
        if self.reduced_solver is not None:
            return self.reduced_solver(mu0, mu1)

        return self.rom.solve_POD_Galerkin(
            self.reduced_data,
            mu0=mu0,
            mu1=mu1,
            newton_tol=self.newton_tol,
            max_iterations=self.max_iterations
        )

    def relative_error(self, reference, approximation):
        """
        Compute the relative Euclidean error between two vectors.

        A small lower bound is used in the denominator to avoid division by zero
        when the reference vector norm is extremely small.
        """
        denominator = max(np.linalg.norm(reference), 1.0e-14)
        return np.linalg.norm(reference - approximation) / denominator

    def split_global_solution(self, global_solution, speed_n_dofs):
        """
        Split a global solution vector into velocity and pressure components.

        The global ordering is assumed to be `[u_x, u_y, p]`, with `u_x` and
        `u_y` containing `speed_n_dofs` entries each.
        """
        u_x = global_solution[0:speed_n_dofs]
        u_y = global_solution[speed_n_dofs:2 * speed_n_dofs]
        p = global_solution[2 * speed_n_dofs:]
        return u_x, u_y, p

    def compute_errors(self, fom_solution, rom_solution):
        """
        Compute component-wise relative errors between FOM and surrogate outputs.

        The method evaluates errors on the global solution, velocity components,
        velocity magnitude and pressure.
        """
        speed_n_dofs = self.reduced_data["speed_n_dofs"]

        U_fom = fom_solution["u"]
        U_rom = rom_solution["u"]

        u_x_fom, u_y_fom, p_fom = self.split_global_solution(U_fom, speed_n_dofs)
        u_x_rom, u_y_rom, p_rom = self.split_global_solution(U_rom, speed_n_dofs)

        u_mag_fom = np.sqrt(u_x_fom**2 + u_y_fom**2)
        u_mag_rom = np.sqrt(u_x_rom**2 + u_y_rom**2)

        return {
            "err_U": self.relative_error(U_fom, U_rom),
            "err_u_x": self.relative_error(u_x_fom, u_x_rom),
            "err_u_y": self.relative_error(u_y_fom, u_y_rom),
            "err_p": self.relative_error(p_fom, p_rom),
            "err_u_mag": self.relative_error(u_mag_fom, u_mag_rom)
        }

    def print_metric_summary(self, metric_name, values):
        """
        Print mean, maximum and standard deviation for a metric.

        This helper is used to summarize the testing-set results in the terminal.
        """
        values = np.asarray(values)
        print(
            f"{metric_name:<18} | "
            f"mean={np.mean(values):.3e} | "
            f"max={np.max(values):.3e} | "
            f"std={np.std(values):.3e}"
        )

    def evaluate(self, n_test=10, seed=123, results_folder="./Results/POD_Galerkin"):
        """
        Evaluate FOM and reduced/surrogate models on a random testing set.

        For each sampled parameter value, the method solves the FOM, evaluates
        the selected reduced/surrogate model, computes errors and timings, prints
        a summary and writes per-sample and aggregate CSV result files.
        """
        print("\n" + "=" * 100)
        print(f"[Metrics] FOM vs {self.method_name} on testing set")
        print("=" * 100)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        rng = np.random.default_rng(seed)
        testing_set = rng.uniform(
            low=self.parameter_ranges[:, 0],
            high=self.parameter_ranges[:, 1],
            size=(n_test, self.parameter_ranges.shape[0])
        )

        results = {
            "testing_set": testing_set,
            "fom_times": [],
            "rom_times": [],
            "speedups": [],
            "fom_iterations": [],
            "rom_iterations": [],
            "err_U": [],
            "err_u_x": [],
            "err_u_y": [],
            "err_p": [],
            "err_u_mag": []
        }

        per_sample_rows = []

        fom_dofs = self.fom_reference_solution["tot_dofs"]

        if "V" in self.reduced_data:
            rom_dofs = self.reduced_data["V"].shape[1]
        elif "model_dofs" in self.reduced_data:
            rom_dofs = self.reduced_data["model_dofs"]
        else:
            rom_dofs = np.nan

        compression_ratio = fom_dofs / rom_dofs

        for test_id, (mu0_test, mu1_test) in enumerate(testing_set):
            print(
                f"\n[Metrics] test {test_id + 1:03d}/{n_test} | "
                f"mu0={mu0_test:.6g}, mu1={mu1_test:.6g}"
            )

            tic = time.perf_counter()
            fom_solution, _, _ = self.solver.solve_FOM(
                self.p_boundary_info,
                self.u_boundary_info,
                mu0=mu0_test,
                mu1=mu1_test,
                newton_tol=self.newton_tol,
                max_iterations=self.max_iterations,
                plot_solution=False
            )
            fom_time = time.perf_counter() - tic

            tic = time.perf_counter()
            rom_solution = self.solve_reduced_model(mu0_test, mu1_test)
            rom_time = time.perf_counter() - tic

            errors = self.compute_errors(fom_solution, rom_solution)
            speedup = fom_time / max(rom_time, 1.0e-14)

            results["fom_times"].append(fom_time)
            results["rom_times"].append(rom_time)
            results["speedups"].append(speedup)
            results["fom_iterations"].append(fom_solution["iterations"])
            results["rom_iterations"].append(rom_solution["iterations"])
            results["err_U"].append(errors["err_U"])
            results["err_u_x"].append(errors["err_u_x"])
            results["err_u_y"].append(errors["err_u_y"])
            results["err_p"].append(errors["err_p"])
            results["err_u_mag"].append(errors["err_u_mag"])

            per_sample_rows.append({
                                    "test_id": test_id + 1,
                                    "mu0": mu0_test,
                                    "mu1": mu1_test,
                                    "err_U": errors["err_U"],
                                    "err_u_x": errors["err_u_x"],
                                    "err_u_y": errors["err_u_y"],
                                    "err_u_mag": errors["err_u_mag"],
                                    "err_p": errors["err_p"],
                                    "fom_time_s": fom_time,
                                    "rom_time_s": rom_time,
                                    "speedup": speedup,
                                    "fom_iterations": fom_solution["iterations"],
                                    "rom_iterations": rom_solution["iterations"],
                                    "fom_converged": fom_solution["converged"],
                                    "rom_converged": rom_solution["converged"]
                                    })

            print(
                f"[Metrics] err_U={errors['err_U']:.3e} | "
                f"err_u_mag={errors['err_u_mag']:.3e} | "
                f"err_p={errors['err_p']:.3e} | "
                f"FOM_time={fom_time:.3f}s | "
                f"ROM_time={rom_time:.6f}s | "
                f"speedup={speedup:.2f}x"
            )

        print("\n" + "-" * 100)
        print("[Metrics] Summary")
        print("-" * 100)
        print(f"FOM dofs              : {fom_dofs}")
        print(f"ROM dofs              : {rom_dofs}")
        print(f"Compression ratio     : {compression_ratio:.2f}x")
        self.print_metric_summary("err_U", results["err_U"])
        self.print_metric_summary("err_u_x", results["err_u_x"])
        self.print_metric_summary("err_u_y", results["err_u_y"])
        self.print_metric_summary("err_u_mag", results["err_u_mag"])
        self.print_metric_summary("err_p", results["err_p"])
        self.print_metric_summary("FOM time [s]", results["fom_times"])
        self.print_metric_summary("ROM time [s]", results["rom_times"])
        self.print_metric_summary("speedup", results["speedups"])
        self.print_metric_summary("FOM Newton it.", results["fom_iterations"])
        self.print_metric_summary("ROM Newton it.", results["rom_iterations"])
        print("=" * 100)

        results["fom_dofs"] = fom_dofs
        results["rom_dofs"] = rom_dofs
        results["compression_ratio"] = compression_ratio

        per_sample_csv = os.path.join(results_folder, self.results_prefix + "_metrics.csv")
        with open(per_sample_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)

        summary_rows = []
        summary_data = {
            "err_U": results["err_U"],
            "err_u_x": results["err_u_x"],
            "err_u_y": results["err_u_y"],
            "err_u_mag": results["err_u_mag"],
            "err_p": results["err_p"],
            "fom_time_s": results["fom_times"],
            "rom_time_s": results["rom_times"],
            "speedup": results["speedups"],
            "fom_iterations": results["fom_iterations"],
            "rom_iterations": results["rom_iterations"]
        }

        for metric_name, values in summary_data.items():
            values = np.asarray(values)
            summary_rows.append({
                "metric": metric_name,
                "mean": np.mean(values),
                "max": np.max(values),
                "min": np.min(values),
                "std": np.std(values)
            })

        summary_rows.append({
            "metric": "fom_dofs",
            "mean": fom_dofs,
            "max": fom_dofs,
            "min": fom_dofs,
            "std": 0.0
        })

        summary_rows.append({
            "metric": "rom_dofs",
            "mean": rom_dofs,
            "max": rom_dofs,
            "min": rom_dofs,
            "std": 0.0
        })

        summary_rows.append({
            "metric": "compression_ratio",
            "mean": compression_ratio,
            "max": compression_ratio,
            "min": compression_ratio,
            "std": 0.0
        })

        summary_csv = os.path.join(results_folder, self.results_prefix + "_summary.csv")
        with open(summary_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["metric", "mean", "max", "min", "std"])
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"[Metrics] Saved per-sample metrics to: {per_sample_csv}")
        print(f"[Metrics] Saved summary metrics to: {summary_csv}")

        results["per_sample_csv"] = per_sample_csv
        results["summary_csv"] = summary_csv

        return results
import numpy as np
import scipy
from pypolydim import polydim

from Assembler import Assembler
from other_utilities import make_np_sparse,plot_FOM_solution

class Solver(object):
    def __init__(self,discretizer:object,export_path:str):
        """
        Initialize the full-order solver.

        The discretizer is used to build the computational mesh, connectivity,
        geometric data, geometry utilities and VTK utilities. The export path is
        stored and later used when exporting or plotting FOM solutions.
        """
        self.method_type,self.mesh,\
            self.mesh_connectivity_data,self.mesh_geometric_data,\
                self.geometry_utilities,self.vtk_utilities = discretizer.discretize()
        
        self.export_path = export_path
        
    def solve_FOM(
        self,
        p_boundary_info: dict,
        u_boundary_info: dict,
        mu0: float,
        mu1: float,
        newton_tol: float = 1.0e-6,
        max_iterations: int = 10,
        plot_solution: str = False,
        plot_path:str = "./Plots/FOM"
        ):
        """
        Solve the full-order Navier-Stokes finite element problem.

        The method builds Taylor-Hood FEM spaces, assembles the linear Stokes
        contribution and source term, then applies Newton's method to solve the
        nonlinear convective problem. It returns the converged FOM solution, the
        assembled operators needed for ROM construction and the FEM metadata used
        by plotting, post-processing and reduced/surrogate methods.
        """
        
        pressure_reference_element_data,speed_reference_element_data,\
                    pressure_mesh_dofs_info,pressure_dofs_data,\
                            speed_mesh_dofs_info,speed_dofs_data,\
                                 speed_n_dofs,pressure_n_dofs,tot_dofs,\
                                     speed_n_strongs,pressure_n_strongs = self.Taylor_Hood_FEM(self.method_type,
                                                                                        self.mesh,
                                                                                        self.mesh_connectivity_data,
                                                                                        p_boundary_info,
                                                                                        u_boundary_info)
        configuration = self.geometry_utilities,self.mesh,\
                            self.mesh_geometric_data,speed_dofs_data,\
                                speed_reference_element_data,speed_mesh_dofs_info,\
                                    pressure_dofs_data,pressure_mesh_dofs_info,pressure_reference_element_data
        
        assembler = Assembler(configuration=configuration)

        operators = assembler.assemble_linear_system(speed_n_dofs=speed_n_dofs,
                                                    pressure_n_dofs=pressure_n_dofs,
                                                    tot_dofs=tot_dofs,
                                                    mu0=mu0,
                                                    mu1=mu1
                                                    )
        J_S = operators["J_S"]
        f_S = operators["f_S"]
        u_x_strong = operators["u_x_strong"]
        u_y_strong = operators["u_y_strong"]
        p_strong = operators["p_strong"]

        u_x_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(self.geometry_utilities,
                                                                                        self.mesh,
                                                                                        self.mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        self.speed_x_initial_condition).function_dofs
        
        u_y_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(self.geometry_utilities,
                                                                                        self.mesh,
                                                                                        self.mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        self.speed_y_initial_condition).function_dofs
        
        p_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(self.geometry_utilities,
                                                                                        self.mesh,
                                                                                        self.mesh_geometric_data,
                                                                                        pressure_dofs_data,
                                                                                        pressure_reference_element_data,
                                                                                        self.pressure_initial_condition).function_dofs
        
        # Newton Method
        u_k = np.concatenate([u_x_numeric, u_y_numeric, p_numeric])
        du_x_strong = np.zeros(speed_n_strongs)
        du_y_strong = np.zeros(speed_n_strongs)
        dp_strong = np.zeros(pressure_n_strongs)
        residual_norm = 1.0
        solution_norm = 1.0
        num_iteration = 1

        print()
        print("-" * 100)
        while num_iteration < max_iterations and residual_norm > newton_tol * solution_norm:
            c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                u_x_numeric,
                u_y_numeric,
                u_x_strong,
                u_y_strong
            )

            # Convetion Jacobian 
            J_C = make_np_sparse(
                c_operator.convective_operator.operator_dofs,
                [tot_dofs, tot_dofs],
                [0, 0]
            )

            # Convective contribute on forcing term
            f_C = np.concatenate([
                c_operator.convective_rhs,
                np.zeros(pressure_n_dofs)
            ])
            
            # Linearized Residual R(Uk) = F - C(Uk) - JsUk
            J_rhs = f_S - f_C - J_S @ u_k

            # Solve and Update
            du = scipy.sparse.linalg.spsolve(J_S + J_C, J_rhs)
            u_k = u_k + du

            # Take contribution decomposing solution
            du_x = du[0:speed_n_dofs]
            du_y = du[speed_n_dofs:2 * speed_n_dofs]
            dp = du[2 * speed_n_dofs:]

            u_x_numeric = u_k[0:speed_n_dofs]
            u_y_numeric = u_k[speed_n_dofs:2 * speed_n_dofs]
            p_numeric = u_k[2 * speed_n_dofs:]

            # Compute norm of increments
            du_x_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                du_x,
                du_x_strong
            )

            du_y_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                du_y,
                du_y_strong
            )

            dp_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                pressure_dofs_data,
                pressure_reference_element_data,
                dp,
                dp_strong
            )

            # Compute norm of solution, i.e. norms of corrected (with increment) solution
            u_x_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                u_x_numeric,
                u_x_strong
            )

            u_y_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                u_y_numeric,
                u_y_strong
            )

            p_norm_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(
                self.geometry_utilities,
                self.mesh,
                self.mesh_geometric_data,
                pressure_dofs_data,
                pressure_reference_element_data,
                p_numeric,
                p_strong
            )

            # Ratio res/sol tells how small is increment w.r.t solution 
            residual_norm = np.sqrt(
                du_x_norm_L2.numeric_norm_l2**2
                + du_y_norm_L2.numeric_norm_l2**2
                + dp_norm_L2.numeric_norm_l2**2
            )

            solution_norm = np.sqrt(
                u_x_norm_L2.numeric_norm_l2**2
                + u_y_norm_L2.numeric_norm_l2**2
                + p_norm_L2.numeric_norm_l2**2
            )

            relative_increment = residual_norm / solution_norm

            print(
                f"[Newton] iter {num_iteration:02d}/{max_iterations} | "
                f"dofs={tot_dofs} | "
                f"||du||/||u||={relative_increment:.3e} | "
                f"tol={newton_tol:.1e}"
            )

            num_iteration += 1

        relative_increment = residual_norm / solution_norm

        print()
        print(
            f"FOM solve completed | "
            f"mu0={mu0:.6g}, mu1={mu1:.6g} | "
            f"iterations={num_iteration - 1} | "
            f"relative_increment={relative_increment:.3e} | "
            f"converged={residual_norm <= newton_tol * solution_norm}"
        )
        print("-" * 100)

        if plot_solution:
            plot_FOM_solution(self.mesh,speed_dofs_data,
                            u_x_numeric,
                            u_x_strong,
                            u_y_numeric,
                            u_y_strong,
                            pressure_dofs_data,
                            p_numeric,
                            p_strong,
                            self.vtk_utilities,
                            self.export_path,
                            plot_path,
                            method = '_FOM'
                            )
        
        ops = {
                "J_S": operators["J_S"],
                'J_A_x': operators["J_A_x"],
                'J_A_y': operators["J_A_y"],
                'J_B_x': operators["J_B_x"],
                'J_B_y': operators["J_B_y"],
                'J_BT_x': operators["J_BT_x"],
                'J_BT_y': operators["J_BT_y"],
                'A_operator': operators["A_operator"],
                'B_x_operator': operators["B_x_operator"],
                'B_y_operator': operators["B_y_operator"],
                "J_A": operators["J_A"],
                "J_B": operators["J_B"],
                }

        sol = {
                "u": u_k,
                "u_x": u_x_numeric,
                "u_y": u_y_numeric,
                "p": p_numeric,
                "iterations": num_iteration - 1,
                "relative_increment": relative_increment,
                "converged": residual_norm <= newton_tol * solution_norm,
                "speed_n_dofs": speed_n_dofs,
                "pressure_n_dofs": pressure_n_dofs,
                "tot_dofs": tot_dofs
                }
        
        fem_data = {
                    "geometry_utilities": self.geometry_utilities,
                    "vtk_utilities": self.vtk_utilities,
                    "mesh": self.mesh,
                    "mesh_geometric_data": self.mesh_geometric_data,
                    "pressure_reference_element_data": pressure_reference_element_data,
                    "speed_reference_element_data": speed_reference_element_data,
                    "pressure_mesh_dofs_info": pressure_mesh_dofs_info,
                    "pressure_dofs_data": pressure_dofs_data,
                    "speed_mesh_dofs_info": speed_mesh_dofs_info,
                    "speed_dofs_data": speed_dofs_data,
                    "speed_n_strongs": speed_n_strongs,
                    "pressure_n_strongs": pressure_n_strongs,
                    "u_x_strong": u_x_strong,
                    "u_y_strong": u_y_strong,
                    "p_strong": p_strong,
                    "assembler": assembler
                    }

        return sol,ops,fem_data   


    def speed_x_initial_condition(self,x, y, z):  
        """
        Define the initial condition for the x-component of the velocity.

        The current FOM Newton initialization uses a zero velocity field in the
        x-direction at all spatial points.
        """
        return 0.0
    def speed_y_initial_condition(self,x, y, z):  
        """
        Define the initial condition for the y-component of the velocity.

        The current FOM Newton initialization uses a zero velocity field in the
        y-direction at all spatial points.
        """
        return 0.0
    def pressure_initial_condition(self,x, y, z):  
        """
        Define the initial condition for the pressure field.

        The current FOM Newton initialization uses a zero pressure field at all
        pressure degrees of freedom.
        """
        return 0.0

    def set_dofs(self):
        """
        Create the boundary-information objects used by the DOF manager.

        The method defines internal, Dirichlet and Neumann/none boundary markers
        in the format expected by `pypolydim` and returns them for later use in
        the Taylor-Hood FEM discretization.
        """
        info_internal = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_internal.marker = 0

        info_dirichlet = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong)
        info_dirichlet.marker = 1

        info_neumann_none = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_neumann_none.marker = 0

        return info_internal,info_dirichlet,info_neumann_none
    
    def Taylor_Hood_FEM(
            self,
            method_type,
            mesh,
            mesh_connectivity_data,
            p_boundary_info,
            u_boundary_info
            ):
        """
        Build the Taylor-Hood finite element spaces for pressure and velocity.

        The method creates the pressure and velocity reference elements, assigns
        boundary information to the mesh DOFs, builds the corresponding DOF data
        structures and returns all FEM quantities required by assembly, Newton
        iteration and post-processing.
        """
        
        pressure_reference_element_data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, 1)
        speed_reference_element_data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type,2) 
        dof_manager = polydim.pde_tools.do_fs.DOFsManager()
        pressure_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(pressure_reference_element_data, 
                                                                                            mesh, 
                                                                                            p_boundary_info)
        pressure_dofs_data = dof_manager.create_do_fs_2_d(pressure_mesh_dofs_info, 
                                                        mesh_connectivity_data)

        speed_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(speed_reference_element_data, 
                                                                                        mesh, 
                                                                                        u_boundary_info)
        speed_dofs_data = dof_manager.create_do_fs_2_d(speed_mesh_dofs_info, 
                                                    mesh_connectivity_data)
        
        pressure_n_dofs = pressure_dofs_data.number_do_fs
        pressure_n_strongs = pressure_dofs_data.number_strongs
        speed_n_dofs = speed_dofs_data.number_do_fs
        speed_n_strongs = speed_dofs_data.number_strongs
        tot_dofs = 2 * speed_n_dofs + pressure_n_dofs
        tot_strongs = 2 * speed_n_strongs + pressure_n_strongs

        print('-'*60)
        print("P dofs\t", "P stgs\t", "U dofs\t", "U stgs\t", "T dofs\t", "T stgs")
        print(pressure_n_dofs,"\t", pressure_n_strongs,"\t", speed_n_dofs,"\t", speed_n_strongs,"\t", tot_dofs,"\t", tot_strongs)
        print('-'*60)
        
        return pressure_reference_element_data,speed_reference_element_data,\
                pressure_mesh_dofs_info,pressure_dofs_data,\
                speed_mesh_dofs_info,speed_dofs_data,\
                speed_n_dofs,pressure_n_dofs,tot_dofs,\
                speed_n_strongs,pressure_n_strongs
                
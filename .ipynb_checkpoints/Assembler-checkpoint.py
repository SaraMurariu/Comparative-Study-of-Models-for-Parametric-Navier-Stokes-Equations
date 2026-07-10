import numpy as np
from pypolydim import polydim

from other_utilities import make_np_sparse

class Assembler(object):
    def __init__(self,configuration:tuple):
        """
        Initialize the assembler with the FEM configuration data.

        The configuration tuple contains geometry utilities, mesh data, geometric
        data, velocity and pressure DOF structures, mesh DOF information and
        reference element data. These quantities are stored and reused by the
        assembly routines.
        """
        self.geometry_utilities,self.mesh,\
            self.mesh_geometric_data,self.speed_dofs_data,\
                self.speed_reference_element_data,self.speed_mesh_dofs_info,\
                    self.pressure_dofs_data,self.pressure_mesh_dofs_info,self.pressure_reference_element_data = configuration
        
    def assemble_linear_system(self,speed_n_dofs:int,pressure_n_dofs:int,tot_dofs:int,mu0:float,mu1:float):
        """
        Assemble the linear Stokes contribution and forcing vector.

        The method assembles diffusion, pressure-velocity coupling, source term
        and strong boundary data. It builds the global block operators used in
        the FOM Newton procedure and returns both the combined matrices and the
        component blocks needed later by the ROM construction.
        """

        A_operator = self.assemble_A_operator()
        B_x_operator, B_y_operator = self.assemble_B_operator()
        f_x_function,f_y_function = self.build_f_components(mu1)
        f_S = self.assemble_f(f_x_function, f_y_function, pressure_n_dofs)
        u_x_strong,u_y_strong = self.assemble_u_strong()
        p_strong = self.assemble_p_strong()

        J_A_x = make_np_sparse(
            A_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [0, 0]
        )

        J_A_y = make_np_sparse(
            A_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [speed_n_dofs, speed_n_dofs]
        )

        J_B_x = make_np_sparse(
            B_x_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [2 * speed_n_dofs, 0]
        )

        J_B_y = make_np_sparse(
            B_y_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [2 * speed_n_dofs, speed_n_dofs]
        )

        J_BT_x = make_np_sparse(
            B_x_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [2 * speed_n_dofs, 0],
            True
        )

        J_BT_y = make_np_sparse(
            B_y_operator.operator_dofs,
            [tot_dofs, tot_dofs],
            [2 * speed_n_dofs, speed_n_dofs],
            True
        )

        J_A = J_A_x + J_A_y
        J_B = J_B_x + J_B_y + J_BT_x + J_BT_y
        J_S = mu0 * J_A - J_B

        return {
            "J_S": J_S,
            "f_S": f_S,
            "u_x_strong": u_x_strong,
            "u_y_strong": u_y_strong,
            "p_strong": p_strong,
            "J_A": J_A,
            "J_B": J_B,
            "J_A_x": J_A_x,
            "J_A_y": J_A_y,
            "J_B_x": J_B_x,
            "J_B_y": J_B_y,
            "J_BT_x": J_BT_x,
            "J_BT_y": J_BT_y,
            "A_operator": A_operator,
            "B_x_operator": B_x_operator,
            "B_y_operator": B_y_operator
        }

    def build_f_components(self,mu1:float):
        """
        Build the parameter-dependent source-term components.

        The returned functions represent the two velocity components of the
        forcing term `f(x, y; mu1)` and are passed to the FEM source-term
        assembler.
        """
        def f_x_function(x:float,y:float,z:float):
            """
            Evaluate the x-component of the source term at a spatial point.
            """
            return -(mu1**3 * np.pi**2 * np.cos(mu1**2 * np.pi * x)- mu1**2 * np.pi**2) * np.sin(mu1 * np.pi * y) * np.cos(mu1 * np.pi * y) \
                    + (mu1 * np.pi * np.cos(mu1 * np.pi * x) * np.cos(mu1 * np.pi * y))
        
        def f_y_function(x:float,y:float,z:float):
            """
            Evaluate the y-component of the source term at a spatial point.
            """
            return -(mu1**3 * np.pi**2 * np.cos(mu1**2 * np.pi * y)+ mu1**2 * np.pi**2) * np.sin(mu1 * np.pi * x) * np.cos(mu1 * np.pi * x) \
                    + (-mu1 * np.pi * np.sin(mu1 * np.pi * x) * np.sin(mu1 * np.pi * y))

        return f_x_function,f_y_function
    
    def assemble_f(self,f_x_function,f_y_function,pressure_n_dofs):
        """
        Assemble the global source vector.

        The x- and y-components are assembled on the velocity space and then
        concatenated with a zero pressure block to match the global ordering
        `[u_x, u_y, p]`.
        """
        f_x = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(self.geometry_utilities,
                                                                                self.mesh,
                                                                                self.mesh_geometric_data,
                                                                                self.speed_dofs_data,
                                                                                self.speed_reference_element_data,
                                                                                self.speed_reference_element_data,
                                                                                f_x_function
                                                                            )                                                                   

        f_y = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(self.geometry_utilities,
                                                                                self.mesh,
                                                                                self.mesh_geometric_data,
                                                                                self.speed_dofs_data,
                                                                                self.speed_reference_element_data,
                                                                                self.speed_reference_element_data,
                                                                                f_y_function
                                                                            )

        J_f = np.concatenate([
            f_x,
            f_y,
            np.zeros(pressure_n_dofs)
        ])

        return J_f
    
    def pressure_strong_function(self,marker:int,x:float,y:float,z:float):
        """
        Define the strong pressure value associated with constrained pressure DOFs.

        The current pressure constraint uses a zero prescribed value.
        """
        return 0.0
    
    def speed_x_strong_function(self,marker:int,x:float,y:float,z:float):
        """
        Define the strong boundary value for the x-component of velocity.

        The current no-slip boundary condition prescribes zero velocity.
        """
        return 0.0

    def speed_y_strong_function(self,marker:int,x:float,y:float,z:float):
        """
        Define the strong boundary value for the y-component of velocity.

        The current no-slip boundary condition prescribes zero velocity.
        """
        return 0.0
    
    def assemble_u_strong(self):
        """
        Assemble the strong velocity boundary data.

        The method assembles the prescribed values for both velocity components
        on the strong velocity DOFs and returns them for use in the FOM and ROM
        workflows.
        """
        u_x_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(self.geometry_utilities,
                                                                                            self.mesh,
                                                                                            self.mesh_geometric_data,
                                                                                            self.speed_mesh_dofs_info,
                                                                                            self.speed_dofs_data,
                                                                                            self.speed_reference_element_data,
                                                                                            self.speed_x_strong_function
                                                                                        )

        u_y_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(self.geometry_utilities,
                                                                                            self.mesh,
                                                                                            self.mesh_geometric_data,
                                                                                            self.speed_mesh_dofs_info,
                                                                                            self.speed_dofs_data,
                                                                                            self.speed_reference_element_data,
                                                                                            self.speed_y_strong_function
                                                                                        )
        
        return u_x_strong,u_y_strong
    
    def assemble_p_strong(self):
        """
        Assemble the strong pressure data.

        The method assembles the prescribed pressure values on constrained
        pressure DOFs, including the pressure reference condition.
        """
        p_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(self.geometry_utilities,
                                                                                        self.mesh,
                                                                                        self.mesh_geometric_data,
                                                                                        self.pressure_mesh_dofs_info,
                                                                                        self.pressure_dofs_data,
                                                                                        self.pressure_reference_element_data,
                                                                                        self.pressure_strong_function
                                                                                    )
        
        return p_strong
    
    def nu_term(self, x, y, z):
        """
        Define the scalar coefficient used in the diffusion operator.

        The viscosity scaling parameter is applied outside this local coefficient,
        so this function returns the unscaled coefficient equal to one.
        """
        return 1.0
    
    def b_x_term(self,x, y, z):  
        """
        Define the vector field used to assemble the x pressure-velocity coupling.

        The returned vector selects the x-direction contribution in the advection
        style operator used for the divergence/gradient coupling blocks.
        """
        return np.array([\
            1.0,\
            0.0,\
            0.0])
    
    def b_y_term(self,x, y, z):  
        """
        Define the vector field used to assemble the y pressure-velocity coupling.

        The returned vector selects the y-direction contribution in the advection
        style operator used for the divergence/gradient coupling blocks.
        """
        return np.array([\
            0.0,\
            1.0,\
            0.0])
    
    def assemble_A_operator(self):
        """
        Assemble the unscaled velocity diffusion operator.

        The resulting operator is later embedded into the global block matrix for
        both velocity components and multiplied by the parameter `mu0`.
        """
        A_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_diffusion_operator(self.geometry_utilities,
                                                                                                self.mesh,
                                                                                                self.mesh_geometric_data,
                                                                                                self.speed_dofs_data,
                                                                                                self.speed_dofs_data,
                                                                                                self.speed_reference_element_data,
                                                                                                self.speed_reference_element_data,
                                                                                                self.nu_term)
        
        return A_operator
    
    def assemble_B_operator(self):
        """
        Assemble the pressure-velocity coupling operators.

        The method builds the x- and y-direction coupling matrices used to form
        the global saddle-point contribution involving divergence and pressure
        gradient terms.
        """
        B_x_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(self.geometry_utilities,
                                                                                                self.mesh,
                                                                                                self.mesh_geometric_data,
                                                                                                self.speed_dofs_data,
                                                                                                self.pressure_dofs_data,
                                                                                                self.speed_reference_element_data,
                                                                                                self.pressure_reference_element_data,
                                                                                                self.b_x_term)
        
        B_y_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(self.geometry_utilities,
                                                                                                self.mesh,
                                                                                                self.mesh_geometric_data,
                                                                                                self.speed_dofs_data,
                                                                                                self.pressure_dofs_data,
                                                                                                self.speed_reference_element_data,
                                                                                                self.pressure_reference_element_data,
                                                                                                self.b_y_term)
        
        return B_x_operator,B_y_operator
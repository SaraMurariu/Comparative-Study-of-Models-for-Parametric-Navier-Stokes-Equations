import numpy as np

from pypolydim import polydim,gedim
from pypolydim.export_vtk_utilities import ExportVTKUtilities
from other_utilities import plot_mesh

class Discretize(object):
    def __init__(self,vertices:np.array,
                 mesh_size:float,
                 domain_area:float,
                 mesh_path:str):
        """
        Initialize the discretization object.

        The class stores the domain vertices, target mesh size, domain area and
        mesh export path. These quantities are later used to generate the PDE
        domain, create the computational mesh and export the mesh in VTK format.
        """
        
        self.vertices = vertices
        self.mesh_size = mesh_size
        self.domain_area = domain_area
        self.mesh_path = mesh_path

    def discretize(self):
        """
        Execute the full mesh-generation workflow.

        The method creates the geometry, mesh and VTK utilities, builds the PDE
        domain, generates the computational mesh, computes mesh connectivity and
        geometric data, exports the mesh and returns all discretization objects
        required by the finite element solver.
        """
        geometry_utilities, mesh_utilities, vtk_utilities = self.generate_utilies_()
        pde_domain = self.generate_pde_domain_()
        mesh,mesh_geometric_data,method_type = self.generate_mesh_(
                                                    utilities=(geometry_utilities,mesh_utilities),
                                                    pde_domain=pde_domain,
                                                    )
        mesh_connectivity_data = polydim.pde_tools.mesh.MeshMatricesDAO_mesh_connectivity_data(mesh)
        self.export_mesh_(self.mesh_path,mesh,vtk_utilities)

        return method_type,mesh,mesh_connectivity_data,mesh_geometric_data,geometry_utilities,vtk_utilities

    def generate_utilies_(self):
        """
        Create the utility objects required for geometry handling, mesh generation,
        and VTK export.

        The function initializes the GeDiM geometry configuration by setting the
        numerical tolerances used in geometric operations. It then creates the geometry
        utilities, the mesh utilities, and the VTK export utilities used throughout the
        discretization workflow.

        Returns
        -------
        geometry_utilities : gedim.GeometryUtilities
            Object used to perform geometric operations with the prescribed numerical
            tolerances.
        mesh_utilities : gedim.MeshUtilities
            Object used to generate and manipulate mesh data structures.
        vtk_utilities : ExportVTKUtilities
            Object used to export the generated mesh and simulation data in VTK format.
        """
        geometry_utilities_config = gedim.GeometryUtilitiesConfig()
        geometry_utilities_config.tolerance1_d = 1.0e-6
        geometry_utilities_config.tolerance2_d = 1.0e-12
        geometry_utilities = gedim.GeometryUtilities(geometry_utilities_config)
        mesh_utilities = gedim.MeshUtilities()
        vtk_utilities = ExportVTKUtilities()

        return geometry_utilities,mesh_utilities,vtk_utilities

    def generate_pde_domain_(self):
        """
        Create and initialize a two-dimensional PDE domain from a given set of vertices.

        The function builds a pypolydim PDE_Domain_2D object and assigns the input
        vertices to define the computational domain used for the PDE discretization.
        The vertices are expected to describe a 2D geometry embedded in 3D coordinates,
        with each column representing one vertex of the domain.

        Parameters
        ----------
        vertices : np.array
            Array containing the coordinates of the domain vertices. For a unit square,
            the columns correspond to (0,0), (1,0), (1,1), and (0,1), with zero third
            coordinate if the geometry is represented in the plane z = 0.
        domain_area : float
            Area of the computational domain. For the unit square Ω = (0,1)^2, this
            value is equal to 1.0.

        Returns
        -------
        pde_domain : pypolydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D
            Initialized PDE domain object that can be used to generate the mesh and
            define the high-fidelity discretization of the PDE problem.
        """

        pde_domain = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D()
        pde_domain.vertices = self.vertices
        pde_domain.shape_type = polydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D.Domain_Shape_Types.parallelogram
        pde_domain.area = self.domain_area

        return pde_domain
    
    def generate_mesh_(self, utilities: tuple, pde_domain):
        """
        Generate the computational mesh associated with the given PDE domain.

        The function creates a two-dimensional triangular mesh using the GeDiM mesh
        utilities and the pypolydim mesh generation routines. The mesh is generated on
        the input PDE domain with the prescribed mesh size. The method also computes the
        geometric data associated with the mesh, which are required later for the finite
        element discretization.

        The mesh data object is stored as an attribute of the class in order to keep its
        memory alive after the function returns. This is important because the mesh DAO
        object internally depends on the underlying GeDiM mesh data structure.

        Parameters
        ----------
        utilities : tuple
            Tuple containing the geometry utilities and mesh utilities used by GeDiM.
            It is expected to have the form (geometry_utilities, mesh_utilities).
        pde_domain : pypolydim.pde_tools.mesh.pde_mesh_utilities.PDE_Domain_2D
            Two-dimensional computational domain on which the mesh is generated.
        size : float, optional
            Characteristic mesh size used by the mesh generator. Smaller values produce
            a finer mesh. The default value is 0.001.

        Returns
        -------
        mesh : gedim.MeshMatricesDAO
            Data-access object containing the generated mesh information.
        mesh_geometric_data
            Geometric data associated with the generated mesh, required by the finite
            element assembly routines.
        method_type
            Local finite element method selected for the discretization.
        """
        mesh_type = polydim.pde_tools.mesh.pde_mesh_utilities.MeshGenerator_Types_2D.triangular
        method_type = polydim.pde_tools.local_space_pcc_2_d.MethodTypes.fem_pcc
        mesh_size = self.mesh_size

        geometry_utilities, mesh_utilities = utilities
        self.mesh_data = gedim.MeshMatrices()
        mesh = gedim.MeshMatricesDAO(self.mesh_data)

        polydim.pde_tools.mesh.pde_mesh_utilities.create_mesh_2_d(
            geometry_utilities,
            mesh_utilities,
            mesh_type,
            pde_domain,
            mesh_size,
            mesh
        )

        mesh_geometric_data = polydim.pde_tools.mesh.pde_mesh_utilities.compute_mesh_2_d_geometry_data(
            geometry_utilities,
            mesh_utilities,
            mesh
        )

        return mesh, mesh_geometric_data, method_type
    
    def export_mesh_(self,export_mesh_path,mesh,vtk_utilities):
        """
        Export the generated mesh to a VTK file and display a mesh plot.

        The function writes the mesh to the path specified by the user using the VTK
        export utilities, then calls the plotting utility to visualize the mesh inside
        the current Python environment.

        Parameters
        ----------
        export_mesh_path : str
            Output path where the mesh has to be saved in VTK format.
        mesh : gedim.MeshMatricesDAO
            Mesh data structure to be exported and plotted.
        vtk_utilities : ExportVTKUtilities
            Utility object used to export mesh data in VTK format.

        Returns
        -------
        None
            The function only performs export and visualization side effects.
        """
        vtk_utilities.export_mesh(export_mesh_path, mesh)
        plot_mesh(mesh,export_folder='./Plots')
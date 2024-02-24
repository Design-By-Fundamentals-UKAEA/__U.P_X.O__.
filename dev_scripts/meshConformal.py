"""
This module contains cmesh3d and cmesh2d to enable conformal meshing of poly-crystalline grain structure.
"""
class cmesh_geotess_3d():
    """
    cmesh_geotess_3d: Conformal mesh Geometric Tessellation 3D

    This is a core UPXO class. Use to generate conformal meshes for 3D grain
    structures.

    Capabilities:
    -------------
        * 3D tetrahedral mesh generation
        * Optimization of 3D tetrahedral mesh

    Dependencies:
    -------------
        * gmsh
        * pyvtk
        * Dream3D

    Targetted at:
    -------------
        * research: understanding algorithm sensitivity
        * research: understanding parameter sensitivity
        * research: understanding grain growth evolution
        * development: UPXO

    Data structures:
    ----------------
        * Each PXTAl object is stored in a DICT under the instance number key.
        * Most properties are stored as Pandas dataframes.
        * The user input parameters are preserved under self.uiAAAAA,
        where, AAAAA is the additional name string of the corresponsing
        parameter set.
        * Mesh instances are stored seperately as a DICT under the instance
        number key, followed by a nested DICT under the mcstep key.

    Data attributes:
    ----------------
    domain: Size of the domain
    dim: dimensionality
    gmp: global morphological parameters
    qmp: Q-partitioned morphological parameters
    purge_previous: Purge previous study objects
    save_sims: BOOL: flag to pickle raw databases of simulations
    __GENLOCK__: str: flag to lock simulation capability of UPXO

    __GENLOCK__:
    ------------
        If 'locked', no simulations will be performed and grain structures
        will not be developed. Otherwise, if 'open'. Status set to 'locked'
        whenever following cases:
            * when user input validation fails.
            * when computations branch towards any lock imposed by developer
            in UPXO internals.
    """
    __slots__ = ('N', 'nstates', 'domain', 'dim',
                 'algo_hop', 'algo_hops',
                 'purge_previous', '_save_sims_',
                 'gsi',
                 'gmp', 'qmp',
                 'mesh_instances',
                 )
    __paramater_gsi_mapping_behaviour__ = 'one-many'
    __default_mcalg__ = '200'

"""
Created on Thu Aug 15 14:19:00 2024

@author: Dr. Sunil Anandatheertha
"""
# # -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
"""Import the monte-carlo grain structure class."""
from upxo.ggrowth.mcgs import mcgs
"""Instantiate ythe mcgs class"""
pxt = mcgs()
# #############################################################################
"""
Let's look at the data structure a bit.
"""
# START
pxt
'''
This provides the most basic information about the grain struct7ure and the
use inputs used to geenrate it.

UPXO 3D.MCGS
(A: GRID)::   x:(0.0,10.0,1.0),   y:(0.0,10.0,1.0),   z:(0.0,10.0,1.0)
(B: SIMPAR)::   nstates: 16  mcsteps: 20  algorithms: (('310', 100),)
(C: MESHPAR)::   GB Conformity: Non-Conformal
               Target FE Software: Abaqus  Element type: quad4
------------------------------------------------------------
'''
pxt.uigrid
'''
Attribues of gridding definitions:
     TYPE: square
     DIMENSIONALITY: 3
     X: (0.0, 10.0, 1.0)
     Y: (0.0, 10.0, 1.0)
     Z: (0.0, 10.0, 1.0)
     PIXEL SIZE: 1.0
     TRANSFORMATION: none
'''
pxt.uisim
'''
Attributes of Simulation parameters:
     MCSTEPS: 20
     S: 16 - will be deprecated.
     STATE SAMPLING SCHEME: rejection
     CONSIDER BOLTZMANN PROBABILITY: False
     S BOLTZAMNN PROBABILITY: [1. 0.99917335 0.99157797 0.99150421
                               0.98991416 0.97386999 0.97431476 0.95513334
                               0.95758912 0.97843915 0.98672295 0.95940137
                               0.95392641 0.94994557 0.96599799 0.91348975]
     MAXIMUM BOLTZMANN TEMPERATURE FACTOR: 0.1
     BOUNDARY CONDITION TYPE: wrapped
     NON LOCALITY: 1
     KINETICITY: static
'''
pxt.vox_size
pxt.vox_length
'''
1.0 and 1.0
'''
# END
# #############################################################################
"""Simulate the grain structuer and inspect the data-structure."""
pxt.simulate(verbose=False)
''' Sinulation has created temporally evolved grain structure. These are
contained in pxt.gs. We will now inspect this and provide details of it below.
'''
pxt.gs
'''
This is a dictionatry containing monte-carlo time step as keys and the
corresponding UPXO grain structue objects as the value.
{0: UPXO. gs-tslice.3d. 1964951084160,
 1: UPXO. gs-tslice.3d. 1964951084608,
 2: UPXO. gs-tslice.3d. 1964953579584,
 .
 .
 .
 19: UPXO. gs-tslice.3d. 1964948225600}

Thoushgh grain struvcture have been created, these are not fully populated
with the information needed for the completion of either conformal meshing
pipeline or the representativeness qualification pipeline or any other
major thing for that matter. We will achivve these in steps and at the
complettion of each step, the necessary data will be populated and enable
user to perform certain functyions / operations. We will not, at this moment
look at all details of the this object, but rather look at its details in steps
below.
'''
# #############################################################################
"""Lets extract a slice (say 5th) from the grain structue temporal stack"""
tslice = 5
gstslice = pxt.gs[tslice]

gstslice.char_morphology_of_grains(label_str_order=3)
gstslice.set_mprops()
"""
All moorphologuical properti3es can be accessed from the dictionary
gstslice.mprop. See gstslice.mprop.keys(), which returns

>> dict_keys(['volnv', 'volsr', 'volch', 'sanv', 'savi', 'sasr', 'pernv',
              'pervl', 'pergl', 'eqdia', 'kx', 'ky', 'kz', 'kxyz', 'ksr',
              'arbbox', 'arefit', 'sol', 'ecc', 'com', 'sph', 'fn', 'rnd',
              'mi', 'fdim'])

The keys and thier ecxplanations as of 15-08-2024 is proided below:
    volnv: Volume by number of voxels
    volsr: Volume after grain boundary surface reconstruction
    volch: Volume of convex hull
    sanv: surface area by number of voxels
    savi: surface area by voxel interfaces
    sasr: surface area after grain boundary surface reconstruction
    pernv: perimeter by number of voxels
    pervl: perimeter by voxel edge lines
    pergl: perimeter by geometric grain boundary line segments
    eqdia: eqvivalent diameter
    kx: grain boundary voxel local curvature in yz plane
    ky: grain boundary voxel local curvature in xz plane
    kz: grain boundary voxel local curvature in xy plane
    kxyz: mean(kx, ky, kz)
    ksr: k computed from surface reconstruction.
    arbbox: aspect ratio by bounding box
    arefit: aspect ratio by ellipsoidal fit
    sol: solidity
    ecc: eccentricity - how much the shape of the grain differs from a sphere.
    com: compactness
    sph: sphericity
    fn: flatness
    rnd: roundness
    mi: moment of inertia tensor
    fdim: fractal dimension
"""
# #############################################################################
"""We can use the following to extract specific grain structure IDs."""
gstslice.get_largest_gids()
gstslice.get_smallest_gids()
gstslice.single_voxel_grains
gstslice.small_grains(vth=2)  # vth: Volume Threshold
gstslice.large_grains(vth=2)
gstslice.find_grains_by_nvoxels(nvoxels=2)
gstslice.find_grains_by_mprop_range(prop_name='volnv', low=10, high=15,
                                    low_ineq='ge', high_ineq='le')
# #############################################################################
"""The following properties can be used to querry specific morphologuical
properties."""
gstslice.smallest_volume
gstslice.largest_volume
# #############################################################################
"""You can use gstslice.gpos to know the relative locations of grains in the
grain structure position. It is a nested dictionary with the following keys:
    dict_keys(['internal', 'boundary', 'corner', 'face', 'edges'])

gstslice.gpos['internal'] is a set of gids which are completely internal to the
grain strucyure. Internal grains are those which share 0 voxels with any of the
boundaries of the grain strcture.

gstslice.gpos['boundary'] is a set of gids which has atleast 1 voxel at the
grain boundary.

gstslice.gpos['corner'] is a dictionary with the folloiwng keys:
    dict_keys(['all', 'left_back_bottom', 'back_right_bottom',
               'right_front_bottom', 'front_left_bottom', 'left_back_top',
               'back_right_top', 'right_front_top', 'front_left_top'])
gstslice.gpos['corner']['all'] is the set of gids of all grains which contain
any of the corners of the grain structure boundary.
gstslice.gpos['corner']['left_back_bottom'] is the set of gid of the grain
which has a voxel at the vertex of left, back and bottom faces of the grain
structure.
The rest have similar meanings.

gstslice.gpos['face'] is a dictinary of following keys:
    dict_keys(['left', 'right', 'front', 'back', 'top', 'bottom'])
gstslice.gpos['face']['left'] is the set of gids of grains which share atleast
1 voxel with the left face of the grain strycture.

gstslice.gpos['edges'] is a dictionary with the folloiwgh keys:
    dict_keys(['left', 'right', 'back', 'front', 'bottom', 'top', 'front_top',
               'top_back', 'back_bottom', 'bottom_front', 'top_right',
               'right_bottom', 'bottom_left', 'left_top', 'front_right',
               'right_back', 'back_left', 'left_front', 'top_front',
               'back_top', 'bottom_back', 'front_bottom', 'right_top',
               'bottom_right', 'left_bottom', 'top_left', 'right_front',
               'back_right', 'left_back', 'front_left'])
gstslice.gpos['edges']['left'] is the set of gids of all grains which share
atleast 1 voxel with any of the four edges of the left face of the grain
structure. The rest have similar meanings.
gstslice.gpos['edges']['front_top'] is the set of gids of all grains which
share 1 voxel with the edge formed at the intersection of front and top faces
of the grain structure.

I encourage you to try these out.

Note 1: After you grab the grian IDs, you can easily plot them:
    gstslice.plot_grains(gstslice.gpos['edges']['left'])

Note 2: You can get gids of all edge grains and plot them as follows:
    le = gstslice.gpos['edges']['left']
    re = gstslice.gpos['edges']['back']
    fe = gstslice.gpos['edges']['front']
    be = gstslice.gpos['edges']['bottom']
    te = gstslice.gpos['edges']['top']
    all_edge_gids = le.union(re, fe, be, te)

    gstslice.plot_grains(all_edge_gids)
    # See that, if the grain strucyture if grain structure size is small and
    grain sizes are large, here is a high chance, this could cover a large
    portion of the entire grai structure. I suggest you do this at a slightly
    large grain structure (say, 20*20*20) and a tslice of 2 to 5 for the
    simulation algorithm of 310.
"""
# #############################################################################
"""Let us extract values of a scalar variable along a stright line, with the
straight line being specified by starting and ending coordinate indice
locatoins."""
gstslice.get_values_along_line([0, 0, 0], [9, 9, 9], scalars='lgi')
"""This returns a numpy array of lgi values. The line between the two
specified end points is generated using Bresenham algorithm in 3D."""
# #############################################################################
# #############################################################################
"""You can also calculate the intercept - grain size in a couple of different
ways."""
gstslice.get_igs_properties_along_line([0, 0, 0], [9, 9, 9], scalars='lgi')
"""This gives a dictionary with the following keys:
    * ng: Number of grains
    * nv: np array of number ofvoxels between all grain boundaris on the line
    * igs: intercept grain size (mean)
    * igs_median: median value of the igs
    * igs_range: range of grain sizes along the line
    * igs_std: standard deviation of the of grain sizes along the line
    * igs_var: variance of the of grain sizes along the line
    * sv: scalar values along the line between the two specified locaytions
    * sv_unique: unique scalar values along the line between the two specified
        locaytions
"""
gstslice.get_igs_along_line([0, 0, 0], [9, 9, 9], metric='mean', minimum=True,
                       maximum=True, std=True, variance=True, verbose=True)
"""This gives us a dictionaryu of following keys:
    * 'igs': value of the metric of igs values.
    * 'metric': metric specified by the user.
    * 'min': Minimum value. Key only present if minimum is specified True.
    * 'max': Maximum value. Key only present if maximum is specified True.
    * 'std': Std. deviation value. Key only present if std is specified True.
    * 'var': Variance value. Key only present if variance is specified True.
"""
gstslice.get_igs_along_lines(metric='mean', minimum=True, maximum=True,
                             std=True, variance=True, lines_gen_method=1,
                             lines_kwargs1={'plane': 'z',
                                            'start_skip1': 0,
                                            'start_skip2': 0,
                                            'incr1': 10, 'incr2': 10,
                                            'inclination': 'none',
                                            'inclination_extent': 0,
                                            'shift_seperately': False,
                                            'shift_starts': False,
                                            'shift_ends': False,
                                            'start_shift': 0, 'end_shift': 0})
"""
This produces a dictionary with the folloiwng keys:
    * 'igs': overall intercept grain size of the grain structuer.
    * 'metric': metric used for calulation of the overall igs value.
    * 'min': minimum values of igs in each sampling line.
    * 'max': maximum values of igs in each sampling line.
    * 'std': standard deviation of igs values in each sampling line.
    * 'var': varianbce of igs values in each sampline line.
    * 'igs_all': individsual igs over each sampling line.
    * 'ngrains': total number of grains in grain structure.

The values are retained for understanding purpose.
{'igs': 4.282439005439005,
 'metric': 'mean',
 'min': array([1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1,1, 1, 1]),
 'max': array([11,  7, 10, 10,  7,  6, 12,  9,  7, 12,  9,  9,  8, 11, 11, 10,
               8, 13, 10,  8,  7,  8, 10,  8,  7]),
 'std': array([3.02605763, 1.88034951, 3.01817412, 3.26955654, 1.88034951,
               1.63663418, 3.56249323, 2.58650343, 2.23051712, 3.13088951,
               2.69129254, 2.80891438, 2.25610283, 3.04023939, 3.08555686,
               2.63566795, 2.17460086, 3.70716274, 2.3222526 , 2.67526163,
               2.14372469, 2.00590843, 2.49958674, 1.92768678, 1.93469779]),
 'var': array([ 9.15702479,  3.53571429,  9.109375  , 10.69      ,  3.53571429,
                2.67857143, 12.69135802,  6.69      ,  4.97520661,  9.80246914,
                7.24305556,  7.89      ,  5.09      ,  9.24305556,  9.52066116,
                6.94674556,  4.72888889, 13.74305556,  5.39285714,  7.15702479,
                4.59555556,  4.02366864,  6.24793388,  3.71597633,  3.74305556]
              ),
 'igs_all': array([4.45454545, 3.5       , 6.125     , 4.9       , 3.5       ,
                   3.5       , 5.44444444, 4.9       , 4.45454545, 5.44444444,
                   4.08333333, 4.9       , 4.9       , 4.08333333, 4.45454545,
                   3.76923077, 3.26666667, 4.08333333, 3.5       , 4.45454545,
                   3.26666667, 3.76923077, 4.45454545, 3.76923077, 4.08333333]
                  ),
 'ngrains': 983}
"""
gstslice.igs_sed_ratio(metric='mean', lines_gen_method=1,
                  reset_grain_size=True, base_size_spec='volnv',
                  lines_kwargs1={'plane': 'z',
                                 'start_skip1': 0, 'start_skip2': 0,
                                 'incr1': 3, 'incr2': 3,
                                 'inclination': 'random',
                                 'inclination_extent': 0,
                                 'shift_seperately': False,
                                 'shift_starts': False,
                                 'shift_ends': False,
                                 'start_shift': 0, 'end_shift': 0})
# #############################################################################
gstslice.sss_rel_morpho(slice_plane='xy', loc=0, reset_lgi=True,
                        kernel_order=4,
                        mprop_names_2d=['eqdia', 'arbbox', 'solidity'],
                        mprop_names_3d=['eqdia', 'arbbox', 'solidity'],
                        ignore_border_grains_2d=True,
                        ignore_border_grains_3d=True,
                        reset_mprops=False, kdeplot=True, save_plots=True)

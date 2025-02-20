import pandas as pd
import numpy as np
from copy import deepcopy

''' Read the required misorientation dataframe'''
folder = r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_orientation'
file = r'\fcc_std_miso_90_90_90.txt'
filePath = folder + file
df_misori = pd.read_csv(filePath, sep='\s+')

'''Read the standard orientations dataframe. Below is the nomeclature
for column names: w-Cube, g-Goss, b-Brass, s-S, c-Copper, rw-rotated cube,
                  rcu-rotated copper, gt-goss twin, cut-copper twin'''
file = r'\STD_ORI_FCC_ROLLED.txt'
filePath = folder + file
df_std_oris = pd.read_csv(filePath, sep='\s+')

''' LIst of all texture components to be included in the model'''
TC = ['w', 'g', 'b', 's', 'c']

'''List of naames of the EUler angles in the datafgrame'''
ea_col_names = ['ea1', 'ea2', 'ea3']

'''Total number of grains in the poly-crystqalline model'''
N_GRAINS = len(pxt.gs[2].gid)

'''Minimum misorientation angle which random orientations are to have with
each Texture component'''
mo_min_tc_rand = {'w': 17.5, 'g': 17.5, 'b': 17.5, 's': 17.5, 'c': 17.5}

'''Maximum spread of each texture compomnebnt in ythe Eule rnagle space'''
tc_spread = {'w': 15, 'g': 15, 'b': 15, 's': 15, 'c': 15}

'''Volume fractions of te3xture components'''
Vf = {'w': 0.1, 'g': 0.2, 'b': 0.25, 's': 0.15, 'c': 0.1}
Vf_total = sum(Vf.values())
if Vf_total < 1:
    randvf = round(1-Vf_total, 6)
Vf['rand'] = randvf

'''Estimation of total number of grains to be allocated to each
texture component including the random orienations'''
# remember from ,my PhD work that, this should be an iterative process
ngrains = {'w': round(Vf['w'] * N_GRAINS),
           'g': round(Vf['g'] * N_GRAINS),
           'b': round(Vf['b'] * N_GRAINS),
           's': round(Vf['s'] * N_GRAINS),
           'c': round(Vf['c'] * N_GRAINS),
           'rand': round(randvf * N_GRAINS)
           }
# Adjust for the residuals by updating ngrains['rand']
while sum(ngrains.values()) != N_GRAINS:
    if sum(ngrains.values()) < N_GRAINS:
        ngrains['rand'] += 1
    else:
        ngrains['rand'] -= 1

'''Find the location (in pandas dataframe) of orientations which satisfy the
minimum misorientation criteria set in mo_min_tc_rand'''
rand_w = df_misori['w'] > mo_min_tc_rand['w']
rand_g = df_misori['g'] > mo_min_tc_rand['g']
rand_b = df_misori['g'] > mo_min_tc_rand['b']
rand_s = df_misori['g'] > mo_min_tc_rand['s']
rand_c = df_misori['g'] > mo_min_tc_rand['c']
'''Find the union of all these locations. The end result must be the lcations
of oprientations which satisfy the minimum misorientation angle requiremewnt
with all the texture compoments'''
rand_loc = np.where((rand_w & rand_g & rand_b & rand_s & rand_c))[0]

'''From ythe above list of all possible random orientrations, unique select
at random, n number of orientations as per the volume fraction requirements
provided in Vf'''
rand_loc = np.sort(np.random.choice(rand_loc,
                                    size=ngrains['rand'],
                                    replace=False,
                                    p=None))
'''Build the random orien6tation euler angle array using the above locations.
These euler angles are gaurenteed to have a misorintation greqter than 15
(i.e. the specified value) with each of the named texture compoment'''
EA_RAND = np.vstack((df_misori['ea1'][rand_loc].to_numpy(),
                     df_misori['ea2'][rand_loc].to_numpy(),
                     df_misori['ea3'][rand_loc].to_numpy())).T
'''Store the actual value of the misorientations as well. This is done from
the developer perspective and the user very needy of additional background
informations'''

mo_rand_w = df_misori.loc[rand_loc, ea_col_names + TC]

# ------------------------------------------------------
'''Get neighbour grain id list in the grain structure temporal slice of
coice'''
neigh_gid = pxt.gs[2].neigh_gid
'''Get all tyhe grain ids'''
gids = deepcopy(pxt.gs[2].gid)

'''Choose number of grains to be allocated to each texture component'''
gtc = {'w': None,
       'g': None,
       'b': None,
       's': None,
       'c': None,
       'rand': None
       }
'''
CONSTRAINT 1:
    Number of grains in each texture component must be exact
CONSTRAINT 2:
    Two neighbouring grains must never have the same texture component
'''
gids_remaining = np.array(deepcopy(gids))
gid_unsorted = []
gidtc_unsorted = []
# Step 1. Allocate texture compoments at random, respecting the Vf
# Step 2. Remove the above grains from gid dataset
for tc in gtc.keys():
    gtc[tc] = np.random.choice(gids_remaining, ngrains[tc], replace=False)
    gids_remaining = gids_remaining[~np.isin(gids_remaining, gtc[tc])]
    for gid in gtc[tc]:
        gid_unsorted.append(gid)
        gidtc_unsorted.append(tc)
gid_unsorted = np.array(gid_unsorted)
gidtc_unsorted = np.array(gidtc_unsorted)
# gid_sorted = gid_unsorted[np.argsort(gid_unsorted)]  @ Unnecessay
gidtc_sorted = gidtc_unsorted[np.argsort(gid_unsorted)]

'''Now we make a data structure similar to ther neigh_gid, but with the
texture components in values
'''
neigh_tc = {}
for gid in gids:
    tclist = []
    for neighgid in neigh_gid[gid]:
        tc = gidtc_sorted[neighgid-1]
        tclist.append(tc)
    neigh_tc[gid] = tclist



'''.NOW WE START THE PROCESS OF OPTIMIZING THE GRAIN ORINTATION ALLOCATION.'''



def find_max_possible_vf_and_ng(neigh_gid, NRUNS=None):
    if not NRUNS:
        NRUNS = len(neigh_gid)
    ng_max_run = []
    Vf_max_run = []
    gid_solutions = {run: None for run in range(NRUNS)}
    for run in range(NRUNS):
        print(60*'='+f'\nRun number {run}')
        n_selected_grains = []

        maximum = 0.8  # Maximum permitted: 1
        minimum = 0.05  # Must be greater than 0
        increment = 0.05  # Must be lesser than (1-maximum)
        num_iterations_saturation = 5  # Number of iterations which would indicate saturation

        factors = np.arange(minimum, maximum+increment, increment)
        total_grains = len(neigh_gid)
        # Shuffle the grain IDs to start selection randomly
        grain_ids = list(neigh_gid.keys())
        np.random.shuffle(grain_ids)

        for i, f in enumerate(factors):
            grains_to_select = int(f * total_grains)
            selected_grains = []
            for gid in grain_ids:
                # Check if current grain is a neighbor of any already selected grains
                if any(gid in neigh_gid[sgid] or sgid in neigh_gid[gid] for sgid in selected_grains):
                    continue  # Skip this grain if it's a neighbor
                # Add current grain to the selection
                selected_grains.append(gid)
                # Break the loop if we've selected enough grains
                if len(selected_grains) >= grains_to_select:
                    break
            n_selected_grains.append(len(selected_grains))
            # Check for saturation
            if len(n_selected_grains) > num_iterations_saturation:
                if len(list(set(n_selected_grains[-5:]))) == 1:
                    print(f"Iteration {i}. Max possible no. of grains = {len(selected_grains)}")
                    print("Maximum possible number of grains - saturation achieved.")
                    print("Iterations stopped")
                    gid_solutions[run] = selected_grains
                    _ng_ = n_selected_grains[-1]
                    print(f"--Max. num. of non-neigh. grains: {_ng_} out of {len(grain_ids)}")
                    _vf_ = _ng_/len(grain_ids)
                    print(f"--Max. possible Vf of a TC: {_vf_}")
                    break
        ng_max_run.append(_ng_)
        Vf_max_run.append(_vf_)
    ng_max_run = np.array(ng_max_run)
    Vf_max_run = np.array(Vf_max_run)

    ng_max_run_min = ng_max_run.min()
    ng_max_run_max = ng_max_run.max()
    ng_max_run_mean = int(ng_max_run.mean())
    print(60*'+')
    print('ALL RUNS COMPLETED')
    print(f"Bounds of maximum possible Ng: Min: {ng_max_run_min}, Max: {ng_max_run_max} Mean: {ng_max_run_mean}")
    Vf_max_run_min = np.round(Vf_max_run.min(), 4)
    Vf_max_run_max = np.round(Vf_max_run.max(), 4)
    Vf_max_run_mean = np.round(Vf_max_run.mean(), 4)
    print(f"Bounds of maximum possible Vf: Min: {Vf_max_run_min}, Max: {Vf_max_run_max} Mean: {Vf_max_run_mean}")
    print(60*'+')
    ng = [ng_max_run_min, ng_max_run_max, ng_max_run_mean]
    vf = [Vf_max_run_min, Vf_max_run_max, Vf_max_run_mean]
    return ng, Vf, gid_solutions

'''Before any orientation mapping even starts, validate the user provided
value of texture component volume fraction values'''
ng, vf, gid_solutions = find_max_possible_vf_and_ng(neigh_gid, NRUNS=100)

for SOLUTION in [1, 4, 7, 99]:
    plt.figure()
    pxt.gs[2].plot_grains(gids = gid_solutions[SOLUTION])

pxt.gs[2].plot()


print(f"Selected {len(selected_grains)} grains (out of {total_grains}): {selected_grains}")

pxt.gs[2].plot_grains(selected_grains)



import damask
import numpy as np
size = np.ones(3)*1e-5
cells = [100, 100, 100]
N_grains = 200
seeds = damask.seeds.from_random(size, N_grains, cells, True)
grid = damask.GeomGrid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'Polycystal_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')
grid


import damask
import numpy as np
size = np.ones(3)*1e-5
cells = [50, 50, 50]
N_seeds = 100
mindistance = min(size)/20
seeds = damask.seeds.from_Poisson_disc(size, N_seeds, 100, mindistance, False)
grid = damask.GeomGrid.from_Voronoi_tessellation(cells, size, seeds)
grid.save(f'Polycystal_{N_seeds}_{cells[0]}x{cells[1]}x{cells[2]}')
grid






rng = np.random.default_rng(20191102)
rnd = damask.Rotation.from_random(100,rng_seed=rng)
fbr = damask.Rotation.from_fiber_component(crystal=[1, 0],
                                           sample=[1 , 0],
                                           sigma=5.0,
                                           shape=500,
                                           degrees=True,
                                           rng_seed=rng)

fbr.as_Euler_angles()
fbr.average().as_Euler_angles()
fbr.misorientation(fbr.average()).as_Euler_angles()








from orix import data, plot
from orix.vector import Vector3d
import numpy as np
from orix.quaternion import Rotation
from orix.quaternion import Orientation
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
# from diffpy.structure import Atom, Lattice, Structure
from diffpy.structure import Lattice, Structure
from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
from orix.quaternion import Rotation
from orix.io import loadctf

fn = r"D:\export_folder\sunil.ctf"
upxo_map = loadctf(fn)


fbr.quaternion

upxo_map = Rotation(fbr.quaternion)

vec_sample = Vector3d([1, 1, 1])

upxo_map = upxo_map * vec_sample

# Specify a crystal structure and symmetry
phase = Phase(
    point_group="6/mmm",
    structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
)



fig = plt.figure(figsize=(8, 8))
ax0 = fig.add_subplot(111, direction=vec_sample, projection="ipf")
ax0.pole_density_function(Vector3d(upxo_map), log=False, resolution=1, sigma=5)
ax0.scatter(Vector3d(upxo_map), alpha=1.0)









sph = damask.Rotation.from_spherical_component(center=damask.Rotation(),
                                               sigma=7.5,
                                               shape=3,
                                               degrees=True,
                                               rng_seed=rng)

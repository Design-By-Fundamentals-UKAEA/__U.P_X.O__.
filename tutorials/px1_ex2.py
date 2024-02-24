from get_mp_1 import get_mp
import numpy as np
from polyxtal import polyxtal
from mulpoint2d_3 import mulpoint2d
import matplotlib.pyplot as plt
#-----------------------------------------------
xbound, ybound = [-1, 1], [-1, 1]
#-----------------------------------------------
'''
values @METHOD
    1. 1: random > random > uniform
    2. 2: random > pds > bridson
    3. 3: random > random > dart
    4. 4: recgrid > linear
    5. 5: trigrid1 > linear
'''
#-----------------------------------------------
NSEEDPOINTS = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
               120, 140, 160, 180,
               200, 230, 260, 290, 310, 340, 370,
               400, 450, 500, 550, 600, 650,
               700, 750, 800, 850, 900, 950, 1000]

# NSEEDPOINTS = [50, 100]
#-----------------------------------------------
SAMPLES = [int(round(1000/i, 0)) for i in NSEEDPOINTS]
#-----------------------------------------------
xtals_n = []

xtals_all_areas = []
xtals_all_areas_mean = []
xtals_all_areas_sum = []

xtals_boundary_n = []
xtals_boundary_areas = []
xtals_boundary_areas_mean = []
xtals_boundary_areas_sum = []

xtals_bae_percentiles_A_val = []
xtals_bae_percentiles_A_val_min = []
xtals_bae_percentiles_A_val_mean = []
xtals_bae_percentiles_A_val_std = []
xtals_bae_percentiles_A_val_max = []
xtals_bae_percentiles_A_val_var = []
xtals_bae_percentiles_A_n = []

xtals_bae_percentiles_B_val = []
xtals_bae_percentiles_B_val_min = []
xtals_bae_percentiles_B_val_mean = []
xtals_bae_percentiles_B_val_std = []
xtals_bae_percentiles_B_val_max = []
xtals_bae_percentiles_B_val_var = []
xtals_bae_percentiles_B_n = []

xtals_bae_percentiles_C_val = []
xtals_bae_percentiles_C_val_min = []
xtals_bae_percentiles_C_val_mean = []
xtals_bae_percentiles_C_val_std = []
xtals_bae_percentiles_C_val_max = []
xtals_bae_percentiles_C_val_var = []
xtals_bae_percentiles_C_n = []
#-----------------------------------------------
for i, (nrndpnts, samples) in enumerate(zip(NSEEDPOINTS, SAMPLES)):
    print([i, nrndpnts, samples])
    xtals_n_ = []
    
    xtals_all_areas_ = []
    xtals_all_areas_mean_ = []
    xtals_all_areas_sum_ = []
    
    xtals_boundary_n_ = []
    xtals_boundary_areas_ = []
    xtals_boundary_areas_mean_ = []
    xtals_boundary_areas_sum_ = []
    
    xtals_bae_percentiles_A_val_ = []
    xtals_bae_percentiles_A_val_min_ = []
    xtals_bae_percentiles_A_val_mean_ = []
    xtals_bae_percentiles_A_val_std_ = []
    xtals_bae_percentiles_A_val_max_ = []
    xtals_bae_percentiles_A_val_var_ = []
    xtals_bae_percentiles_A_n_ = []
    
    xtals_bae_percentiles_B_val_ = []
    xtals_bae_percentiles_B_val_min_ = []
    xtals_bae_percentiles_B_val_mean_ = []
    xtals_bae_percentiles_B_val_std_ = []
    xtals_bae_percentiles_B_val_max_ = []
    xtals_bae_percentiles_B_val_var_ = []
    xtals_bae_percentiles_B_n_ = []
    
    xtals_bae_percentiles_C_val_ = []
    xtals_bae_percentiles_C_val_min_ = []
    xtals_bae_percentiles_C_val_mean_ = []
    xtals_bae_percentiles_C_val_std_ = []
    xtals_bae_percentiles_C_val_max_ = []
    xtals_bae_percentiles_C_val_var_ = []
    xtals_bae_percentiles_C_n_ = []
    #-----------------------------------------------
    for j in range(samples):
        print(f'i = {i}, SAMPLE = {j}')
        m = mulpoint2d(method = 'random',
                       gridding_technique = 'random',
                       sampling_technique = 'uniform',
                       nrndpnts = nrndpnts,
                       randuni_calc = 'by_points',
                       space = 'linear',
                       xbound = xbound,
                       ybound = ybound,
                       lean = 'veryhigh',
                       make_point_objects = True,
                       make_ckdtree = True,
                       vis = False
                       )
        #-----------------------------------------------
        pxtal = polyxtal(gsgen_method = 'vt',
                         vt_base_tool = 'shapely',
                         point_method = 'mulpoints',
                         mulpoint_object = m,
                         xbound = xbound,
                         ybound = ybound,
                         vis_vtgs = False
                         )
        #-----------------------------------------------
        pxtal.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                         base_data_structure_to_use = 'shapely',
                                         build_scalar_fields = True,
                                         scalar_field_names = ['bx_ape'],
                                         viz = False,
                                         vis_dpi = 200,
                                         throw = False
                                         )
        #-----------------------------------------------
        field_values, xtal_ids = pxtal.identify_grains_from_field_threshold(field_name = 'areas_polygonal_exterior',
                                                                            threshold_definition = 'percentiles',
                                                                            threshold_limits_values = [[0.0, 0.05], [0.05, 0.10], [0.10, 0.15]],
                                                                            threshold_limits_percentiles = [[0, 10], [10, 90], [90, 100]],
                                                                            inequality_definitions = [['>=', '<='], ['>=', '<='], ['>=', '<=']],
                                                                            exclude_grains = None,
                                                                            save_as_attribute = True,
                                                                            throw = True
                                                                            )
        
        xtals_bae_percentiles_A_val_.append(field_values[0])
        xtals_bae_percentiles_A_val_min_.append(field_values[0].min())
        xtals_bae_percentiles_A_val_mean_.append(field_values[0].mean())
        xtals_bae_percentiles_A_val_std_.append(field_values[0].std())
        xtals_bae_percentiles_A_val_max_.append(field_values[0].max())
        xtals_bae_percentiles_A_val_var_.append(field_values[0].var())
        xtals_bae_percentiles_A_n_.append(field_values[0].size)
        
        xtals_bae_percentiles_B_val_.append(field_values[1])
        xtals_bae_percentiles_B_val_min_.append(field_values[1].min())
        xtals_bae_percentiles_B_val_mean_.append(field_values[1].mean())
        xtals_bae_percentiles_B_val_std_.append(field_values[1].std())
        xtals_bae_percentiles_B_val_max_.append(field_values[1].max())
        xtals_bae_percentiles_B_val_var_.append(field_values[1].var())
        xtals_bae_percentiles_B_n_.append(field_values[1].size)
        
        xtals_bae_percentiles_C_val_.append(field_values[2])
        xtals_bae_percentiles_C_val_min_.append(field_values[2].min())
        xtals_bae_percentiles_C_val_mean_.append(field_values[2].mean())
        xtals_bae_percentiles_C_val_std_.append(field_values[2].std())
        xtals_bae_percentiles_C_val_max_.append(field_values[2].max())
        xtals_bae_percentiles_C_val_var_.append(field_values[2].var())
        xtals_bae_percentiles_C_n_.append(field_values[2].size)
        
        #-----------------------------------------------
        xtals_n_.append(pxtal.L0.xtals_n)

        xtals_all_areas_.append(pxtal.L0.xtal_ape_val)
        xtals_all_areas_mean_.append(pxtal.L0.xtal_ape_val.mean())
        xtals_all_areas_sum_.append(pxtal.L0.xtal_ape_val.sum())
        
        xtals_boundary_n_.append(pxtal.L0.xtal_ss_boundary.n)
        xtals_boundary_areas_.append(np.array(pxtal.L0.xtal_ss_boundary.ape_val))
        xtals_boundary_areas_mean_.append(np.array(pxtal.L0.xtal_ss_boundary.ape_val).mean())
        xtals_boundary_areas_sum_.append(np.array(pxtal.L0.xtal_ss_boundary.ape_val).sum())
    #-----------------------------------------------
    xtals_n.append(np.array(xtals_n_).mean())
    
    xtals_all_areas.append(xtals_all_areas_)
    xtals_all_areas_mean.append(np.array(xtals_all_areas_mean_).mean())
    xtals_all_areas_sum.append(np.array(xtals_all_areas_sum_).mean())
    
    xtals_boundary_n.append(round(np.array(xtals_boundary_n_).mean(), 0))
    xtals_boundary_areas.append(xtals_boundary_areas_)
    xtals_boundary_areas_mean.append(np.array(xtals_boundary_areas_mean_).mean())
    xtals_boundary_areas_sum.append(np.array(xtals_boundary_areas_sum_).mean())
    
    xtals_bae_percentiles_A_val.append(xtals_bae_percentiles_A_val_)
    xtals_bae_percentiles_A_val_min.append(np.array(xtals_bae_percentiles_A_val_min_).min())
    xtals_bae_percentiles_A_val_mean.append(np.array(xtals_bae_percentiles_A_val_mean_).max())
    xtals_bae_percentiles_A_val_std.append(np.array(xtals_bae_percentiles_A_val_std_).std())
    xtals_bae_percentiles_A_val_max.append(np.array(xtals_bae_percentiles_A_val_max_).max())
    xtals_bae_percentiles_A_val_var.append(np.array(xtals_bae_percentiles_A_val_var_).var())
    xtals_bae_percentiles_A_n.append(np.array(xtals_bae_percentiles_A_n_).mean())
    
    xtals_bae_percentiles_B_val.append(xtals_bae_percentiles_B_val_)
    xtals_bae_percentiles_B_val_min.append(np.array(xtals_bae_percentiles_B_val_min_).min())
    xtals_bae_percentiles_B_val_mean.append(np.array(xtals_bae_percentiles_B_val_mean_).max())
    xtals_bae_percentiles_B_val_std.append(np.array(xtals_bae_percentiles_B_val_std_).std())
    xtals_bae_percentiles_B_val_max.append(np.array(xtals_bae_percentiles_B_val_max_).max())
    xtals_bae_percentiles_B_val_var.append(np.array(xtals_bae_percentiles_B_val_var_).var())
    xtals_bae_percentiles_B_n.append(np.array(xtals_bae_percentiles_B_n_).mean())
    
    xtals_bae_percentiles_C_val.append(xtals_bae_percentiles_C_val_)
    xtals_bae_percentiles_C_val_min.append(np.array(xtals_bae_percentiles_C_val_min_).min())
    xtals_bae_percentiles_C_val_mean.append(np.array(xtals_bae_percentiles_C_val_mean_).max())
    xtals_bae_percentiles_C_val_std.append(np.array(xtals_bae_percentiles_C_val_std_).std())
    xtals_bae_percentiles_C_val_max.append(np.array(xtals_bae_percentiles_C_val_max_).max())
    xtals_bae_percentiles_C_val_var.append(np.array(xtals_bae_percentiles_C_val_var_).var())
    xtals_bae_percentiles_C_n.append(np.array(xtals_bae_percentiles_C_n_).mean())
    #-----------------------------------------------
#-----------------------------------------------
fig = plt.figure(dpi = 100)
ax = plt.axes()
ax.plot(NSEEDPOINTS, SAMPLES, '-ks', label = 'pxtals_n')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Number of pxtal samples')
ax.legend()
plt.show()
#-----------------------------------------------
fig = plt.figure(dpi = 100)
ax = plt.axes()
ax.plot(NSEEDPOINTS, xtals_n, '-ks', label = 'xtals_n')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Number of xtals')
ax.legend()
plt.show()
#-----------------------------------------------
fig = plt.figure(dpi = 100)
ax = plt.axes()
ax.plot(xtals_n, xtals_boundary_n, '-ks', label = 'xtals_n_boundary')
ax.set_xlabel('Number of xtals in pxtal')
ax.set_ylabel('Number of xtals on pxtal boundary')
ax.legend()
plt.show()
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.plot(NSEEDPOINTS, xtals_all_areas_mean, '-ks', label = 'A_xtal_all.min')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
ax.legend()
plt.show()
#-----------------------------------------------
factor = [a/b for a, b in zip(xtals_boundary_areas_mean, xtals_all_areas_mean)]
plt.figure(dpi = 100)
ax = plt.axes()
plt.plot(NSEEDPOINTS, factor, '-ks')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Ratio of mean area of xtals on boundary \n to all')
ax.legend()
plt.show()
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.plot(NSEEDPOINTS, xtals_all_areas_sum, '-ks', label = 'Sum all xtals')
plt.plot(NSEEDPOINTS, xtals_boundary_areas_sum, '-bo', label = 'Sum boundary xtals')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
ax.set_xlim([min(NSEEDPOINTS), max(NSEEDPOINTS)])
ax.set_ylim([0, np.abs(xbound).sum()*np.abs(ybound).sum()])
ax.legend()
plt.show()
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.title('[0, 10] percentiles of area distribution')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
plt.plot(xtals_n, xtals_bae_percentiles_A_val_min, ':bx', label = 'Min. [0, 10] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_A_val_mean, '-r.', label = 'Mean. [0, 10] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_A_val_max, '-g+', label = 'Max. [0, 10] percentile')
plt.legend()
plt.show()
#-----------------------------------------------
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.title('[10, 90] percentiles of area distribution')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
plt.plot(xtals_n, xtals_bae_percentiles_B_val_min, ':bx', label = 'Min. [10, 90] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_B_val_mean, '-r.', label = 'Mean. [10, 90] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_B_val_max, '-g+', label = 'Max. [10, 90] percentile')
plt.legend()
plt.show()
#-----------------------------------------------
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.title('[90, 100] percentiles of area distribution')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
plt.plot(xtals_n, xtals_bae_percentiles_C_val_min, ':bx', label = 'Min. [90, 100] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_C_val_mean, '-r.', label = 'Mean. [90, 100] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_C_val_max, '-g+', label = 'Max. [90, 100] percentile')
plt.legend()
plt.show()
#-----------------------------------------------
#-----------------------------------------------
plt.figure(dpi = 100)
plt.plot(xtals_n, xtals_bae_percentiles_C_val_std, '-ks')
plt.show()
#-----------------------------------------------
#-----------------------------------------------
plt.figure(dpi = 100)
ax = plt.axes()
plt.title('Standard deviation of distributions in percentiles bands')
ax.set_xlabel('Number of seed points')
ax.set_ylabel('Area')
plt.plot(xtals_n, xtals_bae_percentiles_A_val_std, ':bx', label = 'Min. [0, 10] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_B_val_std, '-r.', label = 'Mean. [10, 90] percentile')
plt.plot(xtals_n, xtals_bae_percentiles_C_val_std, '-g+', label = 'Max. [90, 100] percentile')
plt.legend()
plt.show()
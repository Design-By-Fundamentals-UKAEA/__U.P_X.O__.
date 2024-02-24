from mcgsv2 import monte_carlo_grain_structure as mcgs
import matplotlib.pyplot as plt
from mcgsv2 import mcrepr
import pandas as pd
import numpy as np
import seaborn as sns

TARGET_GRAIN_STRUCTURE = mcgs()
TARGET_GRAIN_STRUCTURE.simulate()
TARGET_GRAIN_STRUCTURE.detect_grains()
TARGETtslice = 10
TARGET_GRAIN_STRUCTURE.gs[TARGETtslice].char_morph_2d()
# #############################################################################
NUM_SAMPLE_SETS = 5
SAMPLE_SETS_LIST = [sset+1 for sset in range(NUM_SAMPLE_SETS)]
sample_TARGETtslices = [7, 8, 9, 10, 11, 12, 13, 14, 15]
SAMPLE_DATASET = {}
# PREPARE SAMPLE DATA SETS
for sample_set_n in SAMPLE_SETS_LIST:
    SAMPLE_GS = mcgs()
    SAMPLE_GS.simulate()
    SAMPLE_GS.detect_grains()
    sample_names = ["sample"+str(sample_set_n)+"_slice"+str(sts) for i, sts in enumerate(sample_TARGETtslices)]
    samples = {sk: SAMPLE_GS.gs[sts] for sk, sts, in zip(sample_names, sample_TARGETtslices)}
    for sample_name in sample_names:
        print(f"Characterising SAMPLESET_: {sample_set_n} :: {sample_name}")
        samples[sample_name].char_morph_2d()
    SAMPLE_DATASET['SAMPLESET_'+str(sample_set_n)] = samples
# #############################################################################
# ASSESS REPRESENTATIVENESS OVER ALL SAMPLE DATASETS
REPRESENTATIVENESS = {}

# ASSESS REPRESENTATIVENESS OVER ALL SAMPLE DATASETS
ASSESSMENT_PARAMETERS = ["area", "perimeter", "npixels", "npixels_gb",
                         "eq_diameter", "perimeter_crofton", "compactness",
                         "gb_length_px", "aspect_ratio", "solidity", "eccentricity",
                         "feret_diameter", "major_axis_length", "minor_axis_length"
                         ]

# MORPHOLOGICAL_PARAMETER_OF_INTEREST: MPOI
MPOI_names = ["area",
              "perimeter",
              "aspect_ratio",
              "major_axis_length",
              "minor_axis_length",
              "compactness",
              "solidity",
              "eccentricity"]
MPOI_units = {"area": r"$\mu m^2$",
              "perimeter": r"$\mu m$",
              "aspect_ratio": "",
              "major_axis_length": r"$\mu m$",
              "minor_axis_length": r"$\mu m$",
              "compactness": "",
              "solidity": "",
              "eccentricity": ""}

MPOI_subplot_num = {p: i for i, p in enumerate(MPOI_names, start=1)}
RQM1_metrics = ["kurtosis", "skewness"]
Y = ["MWU", "KW", "KS"]
RQM1 = {}
RQM2_P = {}
RQM2_D = {}
for sds, sample_set_n in zip(SAMPLE_DATASET, SAMPLE_SETS_LIST):
    REPRESENTATIVENESS[sds] = mcrepr(target_type="umc2",
                                     target=TARGET_GRAIN_STRUCTURE.gs[TARGETtslice],
                                     samples=SAMPLE_DATASET[sds])
    REPRESENTATIVENESS[sds].parameters = ASSESSMENT_PARAMETERS
    REPRESENTATIVENESS[sds].determine_distr_type(throw=False)
    REPRESENTATIVENESS[sds].test()
    # rep.performance
    MPOI_df_PValues = {p: None for p in MPOI_names}
    MPOI_df_DValues = {p: None for p in MPOI_names}
    #
    RQM1_DICT = {'TARGET': {rq1metric: {p: REPRESENTATIVENESS[sds].distr_type['target'][p]['kurtosis']
                                        for p in MPOI_names} for rq1metric in RQM1_metrics}}
    sample_names = ["sample"+str(sample_set_n)+"_slice"+str(sts) for i, sts in enumerate(sample_TARGETtslices)]
    print(sample_names)

    for sn in sample_names:
        RQM1_DICT[sn] = {rq1metric: {p: REPRESENTATIVENESS[sds].distr_type[sn][p]['kurtosis']
                                     for p in MPOI_names} for rq1metric in RQM1_metrics}

    RQM1[sds] = RQM1_DICT
    # =====================================================
    for MPOI_name in MPOI_names:
        MW_P = [REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['mannwhitneyu'][1] for sample_name in sample_names]
        KW_P = [REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['kruskalwallis'][1] for sample_name in sample_names]
        KS_P = [REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['ks'][1] for sample_name in sample_names]
        # --------------------------------------
        data = []
        for test, values in zip(Y, [MW_P, KW_P, KS_P]):
            for i, value in enumerate(values, 1):
                data.append({"test": test, "sample_names": sample_names[i-1], "pvalues": value})
        MPOI_df_PValues[MPOI_name] = pd.DataFrame(data)
        # --------------------------------------
        MW_D = [np.log(REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['mannwhitneyu'][0]) for sample_name in sample_names]
        KW_D = [np.log(REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['kruskalwallis'][0]) for sample_name in sample_names]
        KS_D = [np.log(REPRESENTATIVENESS[sds].performance[sample_name][MPOI_name]['ks'][0]) for sample_name in sample_names]
        # --------------------------------------
        data = []
        for test, values in zip(Y, [MW_D, KW_D, KS_D]):
            for i, value in enumerate(values, 1):
                data.append({"test": test, "sample_names": sample_names[i-1], "dvalues": value})
        MPOI_df_DValues[MPOI_name] = pd.DataFrame(data)
    RQM2_P[sds] = MPOI_df_PValues
    RQM2_D[sds] = MPOI_df_DValues

# ===================================================================
for sds, sample_set_n in zip(SAMPLE_DATASET, SAMPLE_SETS_LIST):
    SUBPLOT_ARRANGEMENT = (2, 4)
    plt.figure()
    palette_colors = sns.color_palette('Paired')
    palette_dict = {sn: color for sn, color in zip(sample_names, palette_colors)}
    for MPOI_name in MPOI_names:
        ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0],
                          SUBPLOT_ARRANGEMENT[1],
                          MPOI_subplot_num[MPOI_name])
        sns.barplot(x="test",
                    y="pvalues",
                    data=RQM2_P[sds][MPOI_name],
                    hue="sample_names",
                    palette=palette_dict.values(),
                    ax=ax2)
        plt.ylim(0, 1)
        plt.title(f"RQM2: {MPOI_name}", fontsize=16)
        plt.ylabel("$P_{pvalues}$", fontsize=16)
        plt.xlabel("Statistical test", fontsize=12)
        if MPOI_subplot_num[MPOI_name] != 4:
            plt.legend([],[], frameon=False)
        else:
            sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
        plt.show()



# ===================================================================
for sds, sample_set_n in zip(SAMPLE_DATASET, SAMPLE_SETS_LIST):
    SUBPLOT_ARRANGEMENT = (2, 4)
    plt.figure()
    palette_colors = sns.color_palette('Paired')
    palette_dict = {sn: color for sn, color in zip(sample_names, palette_colors)}
    for MPOI_name in MPOI_names:
        ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0],
                          SUBPLOT_ARRANGEMENT[1],
                          MPOI_subplot_num[MPOI_name])
        sns.barplot(x="test",
                    y="dvalues",
                    data=RQM2_D[sds][MPOI_name],
                    hue="sample_names",
                    palette=palette_dict.values(),
                    ax=ax2)
        plt.ylim(-12, 15)
        plt.title(f"RQM2: {MPOI_name}", fontsize=16)
        plt.ylabel("$ln(D_{dvalues})$", fontsize=16)
        plt.xlabel("Statistical test", fontsize=12)
        if MPOI_subplot_num[MPOI_name] != 4:
            plt.legend([],[], frameon=False)
        else:
            sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
        plt.show()
# ===================================================================
plt.imshow(REPRESENTATIVENESS[sds].target.s)
plt.title(f"Target grain structure at T.Slice = {TARGETtslice}")
plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
plt.colorbar()
SUBPLOT_ARRANGEMENT = (2, 5)
for sds in SAMPLE_DATASET:
    plt.figure()
    for i, sample in enumerate(REPRESENTATIVENESS[sds].samples, start = 1):
        ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0], SUBPLOT_ARRANGEMENT[1], i)
        plt.imshow(REPRESENTATIVENESS[sds].samples[sample].s)
        plt.title(f"{sample}")
        plt.xlabel(r"X-axis, $\mu m$", fontsize=10)
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=10)
# ===================================================================
MPOI_names = ["area",
              "perimeter",
              "aspect_ratio",
              "major_axis_length",
              "minor_axis_length",
              "compactness",
              "solidity",
              "eccentricity"]

HISTOGRAM_FOR_MORPH_PROPERTY = 'eccentricity'
UNITS = MPOI_units[HISTOGRAM_FOR_MORPH_PROPERTY]
binwidth = 0.5
legend_font_size = 7
label_font_size = 8
ticklabel_font_size = 8
SUBPLOT_ARRANGEMENT = (1, NUM_SAMPLE_SETS+1)
fig, ax = plt.subplots(figsize=(5, 3), dpi=140)
ax_target = plt.subplot(SUBPLOT_ARRANGEMENT[0], SUBPLOT_ARRANGEMENT[1], 1)
kde_target = sns.kdeplot(REPRESENTATIVENESS[sds].target.prop[HISTOGRAM_FOR_MORPH_PROPERTY],
                         bw_adjust=binwidth,
                         label="TARGET"
                         )
ax_samplesets = []
ax_target.legend(fontsize = legend_font_size)
ax_target.set_xlabel(HISTOGRAM_FOR_MORPH_PROPERTY + " " + UNITS, fontsize = label_font_size)
ax_target.set_ylabel("Kernel Density Estimate (K.D.E)", fontsize = label_font_size)
ax_target.set_aspect('auto')
ax_target.tick_params(labelsize=ticklabel_font_size)

xmin_target = kde_target.dataLim.xmin
xmax_target = kde_target.dataLim.xmax
ymin_target = kde_target.dataLim.ymin
ymax_target = kde_target.dataLim.ymax
kde_samples = []
xmins_samples = []
xmaxs_samples = []
ymins_samples = []
ymaxs_samples = []
for i, sds in enumerate(SAMPLE_DATASET, start=2):
    print(i)
    ax_samplesets.append(plt.subplot(SUBPLOT_ARRANGEMENT[0], SUBPLOT_ARRANGEMENT[1], i))
    for sample in REPRESENTATIVENESS[sds].samples:
        kde_samples.append(sns.kdeplot(REPRESENTATIVENESS[sds].samples[sample].prop[HISTOGRAM_FOR_MORPH_PROPERTY],
                                       bw_adjust=binwidth,
                                       label=sample
                                       )
                           )
for ax_sampleset in ax_samplesets:
    ax_sampleset.legend(fontsize = legend_font_size)
    ax_sampleset.set_xlabel(HISTOGRAM_FOR_MORPH_PROPERTY + " " + UNITS, fontsize = label_font_size)
    ax_sampleset.set_ylabel("Kernel Density Estimate (K.D.E)", fontsize = label_font_size)
    ax_sampleset.tick_params(labelsize=ticklabel_font_size)

for kde_sample in kde_samples:
    xmins_samples.append(kde_sample.dataLim.xmin)
    xmaxs_samples.append(kde_sample.dataLim.xmax)
    ymins_samples.append(kde_sample.dataLim.ymin)
    ymaxs_samples.append(kde_sample.dataLim.ymax)

xmin = min(xmin_target, min(xmins_samples))
xmax = min(xmax_target, min(xmaxs_samples))
ymin = min(ymin_target, min(ymins_samples))
ymax = max(ymax_target, max(ymaxs_samples))

ax_target.set_xlim(left=xmin, right=xmax)
for kde_sample in kde_samples:
    kde_sample.set_xlim(left=xmin, right=xmax)
ax_target.set_ylim(bottom=ymin, top=ymax)
for kde_sample in kde_samples:
    kde_sample.set_ylim(bottom=ymin, top=ymax)
fig.suptitle(HISTOGRAM_FOR_MORPH_PROPERTY, fontdict={'family': 'serif',
          'color':  'darkred',
          'weight': 'normal',
          'size': 16,})
# ===================================================================
SAMPLESET_NAME = 'SAMPLESET_5'

MWU_threshold = 0.6
KW_threshold = 0.6
KS_threshold = 0.6

MPOI_RQM2_accepted_ALLTESTS_ANY = {morph_prop: [] for morph_prop in MPOI_names}

MPOI_RQM2_accepted_MWU = {morph_prop: [] for morph_prop in MPOI_names}
MPOI_RQM2_accepted_KW = {morph_prop: [] for morph_prop in MPOI_names}
MPOI_RQM2_accepted_KS = {morph_prop: [] for morph_prop in MPOI_names}
MPOI_RQM2_accepted_ALLTESTS_AND = {morph_prop: [] for morph_prop in MPOI_names}

for morph_prop in MPOI_names:
    MWU_P_locations = RQM2_P[SAMPLESET_NAME][morph_prop]['test'] == 'MWU'
    KW_P_locations = RQM2_P[SAMPLESET_NAME][morph_prop]['test'] == 'KW'
    KS_P_locations = RQM2_P[SAMPLESET_NAME][morph_prop]['test'] == 'KS'

    MWU_sample_names = RQM2_P[SAMPLESET_NAME][morph_prop]['sample_names'][MWU_P_locations]
    KW_sample_names = RQM2_P[SAMPLESET_NAME][morph_prop]['sample_names'][KW_P_locations]
    KS_sample_names = RQM2_P[SAMPLESET_NAME][morph_prop]['sample_names'][KS_P_locations]

    MWU_P_values = RQM2_P[SAMPLESET_NAME][morph_prop]['pvalues'][MWU_P_locations]
    KW_P_values = RQM2_P[SAMPLESET_NAME][morph_prop]['pvalues'][KW_P_locations]
    KS_P_values = RQM2_P[SAMPLESET_NAME][morph_prop]['pvalues'][KS_P_locations]

    MWU_acceptable_p_values_locs = MWU_P_values > MWU_threshold
    KW_acceptable_p_values_locs = KW_P_values > KW_threshold
    KS_acceptable_p_values_locs = KS_P_values > KS_threshold

    MWU_acceptable_p_values = MWU_P_values[MWU_acceptable_p_values_locs]
    KW_acceptable_p_values = KW_P_values[KW_acceptable_p_values_locs]
    KS_acceptable_p_values = KS_P_values[KS_acceptable_p_values_locs]

    MWU_acceptable_p_sample_names = MWU_sample_names[MWU_acceptable_p_values_locs].tolist()
    KW_acceptable_p_sample_names = KW_sample_names[KW_acceptable_p_values_locs].tolist()
    KS_acceptable_p_sample_names = KS_sample_names[KS_acceptable_p_values_locs].tolist()

    for _ in MWU_acceptable_p_sample_names:
        MPOI_RQM2_accepted_ALLTESTS_ANY[morph_prop].append(_)
        MPOI_RQM2_accepted_MWU[morph_prop].append(_)
    for _ in KW_acceptable_p_sample_names:
        MPOI_RQM2_accepted_ALLTESTS_ANY[morph_prop].append(_)
        MPOI_RQM2_accepted_KW[morph_prop].append(_)
    for _ in KS_acceptable_p_sample_names:
        MPOI_RQM2_accepted_ALLTESTS_ANY[morph_prop].append(_)
        MPOI_RQM2_accepted_KS[morph_prop].append(_)

    _MWU_ = set(MPOI_RQM2_accepted_MWU[morph_prop])
    _KW_ = set(MPOI_RQM2_accepted_KW[morph_prop])
    _KS_ = set(MPOI_RQM2_accepted_KS[morph_prop])
    _MWU_KW_ = _MWU_.intersection(_KW_)
    _MWU_KW_KS_ = _MWU_KW_.intersection(_KS_)
    MPOI_RQM2_accepted_ALLTESTS_AND[morph_prop] = list(_MWU_KW_KS_)

    MPOI_RQM2_accepted_ALLTESTS_ANY[morph_prop] = list(set(MPOI_RQM2_accepted_ALLTESTS_ANY[morph_prop]))
    print(morph_prop+': '+f"MWU ({MWU_threshold}): "+SAMPLESET_NAME+': '+'ACCEPT: '+str(MWU_acceptable_p_sample_names))
    print(morph_prop+': '+f"KW: ({KW_threshold}): "+SAMPLESET_NAME+': '+'ACCEPT: '+str(KW_acceptable_p_sample_names))
    print(morph_prop+': '+f"KS: ({KS_threshold}): "+SAMPLESET_NAME+': '+'ACCEPT: '+str(KS_acceptable_p_sample_names))

MPOI_RQM2_accepted_ALLTESTS_ANY
MPOI_RQM2_accepted_ALLTESTS_AND
#MPOI_RQM2_accepted_ALLTESTS_ANY = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in MPOI_RQM2_accepted_ALLTESTS_ANY.items()]))

#MPOI_RQM2_accepted_ALLTESTS_AND = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in MPOI_RQM2_accepted_ALLTESTS_AND.items()]))

# ===================================================================
TARGET_ngrains = [TARGET_GRAIN_STRUCTURE.gs[i].n for i in TARGET_GRAIN_STRUCTURE.m]
TARGET_slices, sampleSET_names = TARGET_GRAIN_STRUCTURE.m, [k for k in SAMPLE_DATASET]

SAMPLESETS_ngrains = {}
SAMPLESETS_slices = sample_TARGETtslices

for ssn in sampleSET_names:
    gsslices = SAMPLE_DATASET[ssn]
    ngrains = [gsslice.n for gsslice in gsslices.values()]
    SAMPLESETS_ngrains[ssn] = ngrains
# ===================================================================
fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
plt.axis([0, max(TARGET_slices),
          min(TARGET_ngrains), 1.1*max(TARGET_ngrains)])
plt.plot(TARGET_slices, TARGET_ngrains, '--k', label='TARGET')

for ssn in sampleSET_names:
    plt.plot(SAMPLESETS_slices, SAMPLESETS_ngrains[ssn], label = ssn)
plt.legend()
plt.xlabel("Temporal slice number")
plt.ylabel(r"Average grain size $\mu m^2$")
# -----------------
axis_label_font_size = 14
ticklabel_font_size = 12
legend_font_size = 12
fig, ax = plt.subplots()
plt.plot(TARGET_slices, TARGET_ngrains, 'co', label='TARGET', markersize=8)
markers_set = ["+", "1", "2", "3", "4", "|", "_"]
for ssn, marker in zip(sampleSET_names, markers_set):
    plt.plot(SAMPLESETS_slices, SAMPLESETS_ngrains[ssn], marker, label=ssn, markersize=12)
plt.legend(fontsize=legend_font_size)
plt.xlabel("m", fontsize=axis_label_font_size)
plt.ylabel(r"AGS $\mu m^2$", fontsize=axis_label_font_size)
ax.tick_params(labelsize=ticklabel_font_size)
ax.set_xlim(left=min(SAMPLESETS_slices), right=max(SAMPLESETS_slices))
ax.set_ylim(bottom=100, top=500)
ax.set_facecolor('white')
# ===================================================================
# TEST GRAIN AREA DISTRIBUTION FOR LOGNORMALITY
from scipy import stats as st
# the data is lognormal if np.log(data) is normal


# FOR TARGET
nbins = 50
histdata = np.histogram(REPRESENTATIVENESS['SAMPLESET_1'].target.prop['area'].to_list(), bins = nbins)[0]
histdata = list(histdata)
histdata = histdata[:histdata.index(0)]
# plt.plot(histdata)
pvalx = st.shapiro(np.log( histdata  ))[-1]
print(pvalx )



# FOR SAMPLES
nbins = 50
SAMPLESET_NAME = 'SAMPLESET_5'
for samplesliceNAME in SAMPLE_DATASET[SAMPLESET_NAME]:
    histdata = np.histogram(REPRESENTATIVENESS[SAMPLESET_NAME].samples[samplesliceNAME].prop['perimeter'].to_list(), bins = nbins)[0]
    histdata = list(histdata)
    if 0 in histdata:
        histdata = histdata[:histdata.index(0)]
    # plt.plot(histdata)
    pvalx = st.shapiro(np.log( histdata  ))[-1]
    print(round(pvalx*100)/100)


# ===================================================================
SAMPLESET_NAME = 'SAMPLESET_1'
PROP_NAME = "solidity"
prop_min = 0.99
prop_max = 1.01
Number_grains = {}
AVG = {}
for i, samplesliceNAME in enumerate(SAMPLE_DATASET[SAMPLESET_NAME], start=7):
    gs = REPRESENTATIVENESS[SAMPLESET_NAME].samples[samplesliceNAME]
    lco = prop_min
    uco = prop_max
    A_MAX = gs.prop[PROP_NAME][gs.prop[PROP_NAME].index[gs.prop[PROP_NAME] >= lco]]
    A_B_indices = A_MAX.index[A_MAX <= uco]
    A_B_values = A_MAX[A_B_indices].to_numpy()
    gids = A_B_indices+1
    Number_grains[i] = len(gids.tolist())
    AVG[i] = gs.prop['area'][A_B_indices].mean()
    plt.figure()
    gs.plot_grains_gids(gids,
                        title = samplesliceNAME + ":  " + PROP_NAME + " [" + str(prop_min) + ':' + str(prop_max) + ']'
                        )



gids = a
prop_lgi = np.zeros_like(gs.lgi)
for gid in gids:
    prop_lgi[gs.lgi == gid] = prop_values[gid+1]

plt.figure()
# Plot the grains
cbar_min, cbar_max = min(prop_values), max(prop_values)
cbar_min, cbar_max = prop_min, prop_max
# plt.imshow(area_lgi, cmap="bwr", vmin=min(areas), vmax=max(areas))
plt.imshow(prop_lgi, cmap="bwr", vmin=prop_min, vmax=prop_max)
plt.title(f"{cbar_min} <= Grain {PROP_NAME} <= {cbar_max}")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(
    drawedges=False,
    label=PROP_NAME,
    ticklocation="auto",
    spacing="proportional",
    format="%4.1f",
    location="right",
    ticks=np.linspace(cbar_min, cbar_max, 5),
)
plt.show()






SAMPLESET_NAME = 'SAMPLESET_1'
PROP_NAME = "perimeter"
prop_min = 10
prop_max = 15
for samplesliceNAME in SAMPLE_DATASET[SAMPLESET_NAME]:
    gs = REPRESENTATIVENESS[SAMPLESET_NAME].samples[samplesliceNAME]
    prop_values = gs.prop['perimeter'].values
    # ----------------------------------------------------
    cmap = plt.cm.get_cmap("viridis")
    # define the color for each grain
    colors = cmap(prop_values / prop_values.max())
    # ----------------------------------------------------
    # Plot only those grains which have area between
    prop_lgi = np.zeros_like(gs.lgi)
    for gid in gs.gid:
        prop_lgi[gs.lgi == gid] = prop_values[gid - 1]
    # ----------------------------------------------------
    plt.figure()
    # Plot the grains
    cbar_min, cbar_max = min(prop_values), max(prop_values)
    cbar_min, cbar_max = prop_min, prop_max
    # plt.imshow(area_lgi, cmap="bwr", vmin=min(areas), vmax=max(areas))
    plt.imshow(prop_lgi, cmap="bwr", vmin=prop_min, vmax=prop_max)
    plt.title(f"{cbar_min} <= Grain {PROP_NAME} <= {cbar_max}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(
        drawedges=False,
        label=PROP_NAME,
        ticklocation="auto",
        spacing="proportional",
        format="%4.1f",
        location="right",
        ticks=np.linspace(cbar_min, cbar_max, 5),
    )
    plt.show()

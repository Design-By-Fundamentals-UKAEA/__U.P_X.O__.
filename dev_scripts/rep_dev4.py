from mcgsv2 import monte_carlo_grain_structure as mcgs
from mcgsv2 import mcrepr

TARGETS = mcgs()
TARGETS.simulate()
TARGETS.detect_grains()
tslice = 10
TARGETS.gs[tslice].char_morph_2d()

# #############################################################################
NUM_SAMPLE_SETS = 2
SAMPLE_SETS_LIST = [sset+1 for sset in range(NUM_SAMPLE_SETS)]
sample_tslices = [14, 15]
SAMPLE_DATASET = {}
# PREPARE SAMPLE DATA SETS
for sample_set_n in SAMPLE_SETS_LIST:
    SAMPLE_GS = mcgs()
    SAMPLE_GS.simulate()
    SAMPLE_GS.detect_grains()
    sample_names = ["sample"+str(sample_set_n)+"_slice"+str(sts) for i, sts in enumerate(sample_tslices)]
    samples = {sk: SAMPLE_GS.gs[sts] for sk, sts, in zip(sample_names, sample_tslices)}
    for sample_name in sample_names:
        print(f"Characterising SAMPLESET_: {sample_set_n} :: {sample_name}")
        samples[sample_name].char_morph_2d()
    SAMPLE_DATASET['SAMPLESET_'+str(sample_set_n)] = samples
# #############################################################################
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
MPOI_subplot_num = {"area": 1,
                    "perimeter": 2,
                    "aspect_ratio": 3,
                    "major_axis_length": 4,
                    "minor_axis_length": 5,
                    "compactness": 6,
                    "solidity": 7,
                    "eccentricity": 8,}

REPR = {}
for sds in SAMPLE_DATASET:
    rep = mcrepr(target_type="umc2",
                 target=TARGETS.gs[10],
                 samples=samples
                 )
    rep.parameters = ASSESSMENT_PARAMETERS
    rep.determine_distr_type(throw=False)
    rep.test()
    # rep.performance
    MPOI_df_PValues = {p: None for p in MPOI_names}
    MPOI_df_DValues = {p: None for p in MPOI_names}

# #############################################################################

print("Writing prop of TARGET} to Excel file: {target.xlsx}")
rep.target.prop.to_excel("target.xlsx", index=False)
for sample_name in sample_names:
    filename = sample_name + ".xlsx"
    print(f"Writing prop of {sample_name} to Excel file: {filename}")
    samples[sample_name].prop.to_excel(filename, index=False)

rep.prop_to_excel()

import seaborn as sns
# =====================================================
# =====================================================
# =====================================================
SUBPLOT_ARRANGEMENT = (2, 4)
Y = ["MWU", "KW", "KS"]
import pandas as pd
# =====================================================
RQM1_metrics = ["kurtosis", "skewness"]
RQM1_DICT = {'TARGET': {rq1metric: {p: rep.distr_type['target'][p]['kurtosis']
                                    for p in MPOI_names} for rq1metric in RQM1_metrics}}
for sn in sample_names:
    RQM1_DICT[sn] = {rq1metric: {p: rep.distr_type[sn][p]['kurtosis']
                                 for p in MPOI_names} for rq1metric in RQM1_metrics}
RQM1_DICT['TARGET']['skewness']
RQM1_DICT['sample1_slice10']['skewness']
# ===================================================================
for MPOI_name in MPOI_names:
    MW_P = [rep.performance[sample_name][MPOI_name]['mannwhitneyu'][1] for sample_name in sample_names]
    KW_P = [rep.performance[sample_name][MPOI_name]['kruskalwallis'][1] for sample_name in sample_names]
    KS_P = [rep.performance[sample_name][MPOI_name]['ks'][1] for sample_name in sample_names]
    # --------------------------------------
    data = []
    for test, values in zip(Y, [MW_P, KW_P, KS_P]):
        for i, value in enumerate(values, 1):
            data.append({"test": test, "index": sample_names[i-1], "value": value})
    MPOI_df_PValues[MPOI_name] = pd.DataFrame(data)
    # --------------------------------------
    MW_D = [np.log(rep.performance[sample_name][MPOI_name]['mannwhitneyu'][0]) for sample_name in sample_names]
    KW_D = [np.log(rep.performance[sample_name][MPOI_name]['kruskalwallis'][0]) for sample_name in sample_names]
    KS_D = [np.log(rep.performance[sample_name][MPOI_name]['ks'][0]) for sample_name in sample_names]
    # --------------------------------------
    data = []
    for test, values in zip(Y, [MW_D, KW_D, KS_D]):
        for i, value in enumerate(values, 1):
            data.append({"test": test, "index": sample_names[i-1], "value": value})
    MPOI_df_DValues[MPOI_name] = pd.DataFrame(data)
# ===================================================================
palette_colors = sns.color_palette('Paired')
palette_dict = {sn: color for sn, color in zip(sample_names, palette_colors)}
# ===================================================================
for MPOI_name in MPOI_names:
    ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0],
                      SUBPLOT_ARRANGEMENT[1],
                      MPOI_subplot_num[MPOI_name])
    sns.barplot(x="test",
                y="value",
                data=MPOI_df_PValues[MPOI_name],
                hue="index",
                palette=palette_dict.values(),
                ax=ax2)
    plt.ylim(0, 1)
    plt.title(f"RQM2: {MPOI_name}", fontsize=16)
    plt.ylabel("$P_{value}$", fontsize=16)
    plt.xlabel("Statistical test", fontsize=12)
    if MPOI_subplot_num[MPOI_name] != 4:
        plt.legend([],[], frameon=False)
    else:
        sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    plt.show()
# ===================================================================
for MPOI_name in MPOI_names:
    ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0],
                      SUBPLOT_ARRANGEMENT[1],
                      MPOI_subplot_num[MPOI_name])
    sns.barplot(x="test",
                y="value",
                data=MPOI_df_DValues[MPOI_name],
                hue="index",
                palette=palette_dict.values(),
                ax=ax2)
    plt.ylim(-12, 15)
    plt.title(f"RQM2: {MPOI_name}", fontsize=16)
    plt.ylabel("$ln(D_{value})$", fontsize=16)
    plt.xlabel("Statistical test", fontsize=12)
    if MPOI_subplot_num[MPOI_name] != 4:
        plt.legend([],[], frameon=False)
    else:
        sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
    plt.show()
# ===================================================================
plt.imshow(TARGETS.gs[tslice].s)
plt.title(f"target_slice{tslice}")
plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)

SUBPLOT_ARRANGEMENT = (2, 4)
for i, sample_name in enumerate(samples, start = 1):
    ax2 = plt.subplot(SUBPLOT_ARRANGEMENT[0], SUBPLOT_ARRANGEMENT[1], i)
    plt.imshow(samples[sample_name].s)
    plt.title(f"{sample_name}")
    plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
    plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
# ===================================================================
TARGET_MEANS = {p: rep.target.prop[p].mean() for p in MPOI_names}
SAMPLE_MEANS = {}
for sample in rep.samples:
    SAMPLE_MEANS[sample] = {p: rep.samples[sample].prop[p].mean() for p in MPOI_names}

rep.target.prop['area'].describe()['count']
rep.target.prop['area'].quantile()
# ===================================================================

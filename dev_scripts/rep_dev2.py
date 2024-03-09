from mcgsv2 import monte_carlo_grain_structure as mcgs

TARGETS = mcgs()
TARGETS.simulate()
TARGETS.detect_grains()
tslice = 10
TARGETS.gs[tslice].char_morph_2d()

N_samples = 1
SAMPLE_GS = mcgs()
SAMPLE_GS.simulate()
SAMPLE_GS.detect_grains()
sample_tslices = [14, 15]
sample_names = ["sample"+str(N_samples)+"_slice"+str(sts) for i, sts in enumerate(sample_tslices)]
samples = {sk: SAMPLE_GS.gs[sts] for sk, sts, in zip(sample_names, sample_tslices)}

for sample_name in sample_names:
    print(f"Characterising {sample_name}")
    samples[sample_name].char_morph_2d()

from mcgsv2 import mcrepr
rep = mcrepr(target_type="umc2",
             target=TARGETS.gs[10],
             samples=samples)

rep.target.plotgs()
for sample in rep.samples.values():
    sample.plotgs()

print("Writing prop of TARGET} to Excel file: {target.xlsx}")
rep.target.prop.to_excel("target.xlsx", index=False)
for sample_name in sample_names:
    filename = sample_name + ".xlsx"
    print(f"Writing prop of {sample_name} to Excel file: {filename}")
    samples[sample_name].prop.to_excel(filename, index=False)

rep.prop_to_excel()

import seaborn as sns

from cv2 import line
fig, ax = plt.subplots(figsize=(5, 5))
sns.histplot(
    TARGETS.gs[10].prop["area"],
    binwidth=5,
    kde=True,
    stat="probability",
    color="black",
    linewidth=1,
    legend=True,
)
sns.histplot(
    SAMPLE_GS.gs[8].prop["area"],
    binwidth=5,
    legend=True,
    stat="probability",
    kde=True,
    element="step",
    fill=False,
    linestyle="-",
    linewidth=1,
    color="blue",
)
sns.histplot(
    SAMPLE_GS.gs[10].prop["area"],
    binwidth=5,
    legend=True,
    stat="probability",
    kde=True,
    element="step",
    fill=False,
    linestyle="-",
    linewidth=1,
    color="green",
)
sns.histplot(
    SAMPLE_GS.gs[12].prop["area"],
    binwidth=5,
    legend=True,
    stat="probability",
    kde=True,
    element="step",
    fill=False,
    linestyle="-",
    linewidth=1,
    color="teal",
)


ax.set_xlim(0, 250)
ax.set_ylim(0, 0.3)
ax.set_xlabel("Grain area, $\mu m^2$")
ax.set_xticks(range(0, 201, 25))
ax.legend(
    labels=[
        "Target 2D MCGS",
        "Sample 1 2D MCGS",
        "Sample 2 2D MCGS",
        "Sample 3 2D MCGS",
    ]
)


for sample_name in samples:
    sns.histplot(samples[sample_name].prop["area"], bins=20, legend=True)




print("----------TARGET----------")
print(rep.target.prop["area"])
print("----------SAMPLE-1----------")
print(rep.samples["sample_1"].prop["area"])
print("----------SAMPLE-2----------")
print(rep.samples["sample_2"].prop["area"])
print("----------SAMPLE-3----------")
print(rep.samples["sample_3"].prop["area"])

rep.parameters = ["area", "perimeter", "npixels", "npixels_gb",
                  "eq_diameter", "perimeter_crofton", "compactness",
                  "gb_length_px", "aspect_ratio", "solidity", "eccentricity",
                  "feret_diameter", "major_axis_length", "minor_axis_length"
                  ]

rep.determine_distr_type(throw=False)
rep.distr_type['target']['area'].keys()

Right_Skewed = [rep.distr_type['target'][p]['right_skewed'] for p in rep.parameters]
Left_Skewed = [rep.distr_type['target'][p]['left_skewed'] for p in rep.parameters]
Leptokurtic = [rep.distr_type['target'][p]['leptokurtic'] for p in rep.parameters]
Platykurtic = [rep.distr_type['target'][p]['platykurtic'] for p in rep.parameters]
Normal = [rep.distr_type['target'][p]['normal'] for p in rep.parameters]
Kurtosis = [rep.distr_type['target'][p]['kurtosis'] for p in rep.parameters]
Skewness = [rep.distr_type['target'][p]['skewness'] for p in rep.parameters]

RQM1_metrics = ["kurtosis", "skewness"]

RQM1_DICT = {'TARGET': {rq1metric: {p: rep.distr_type['target'][p]['kurtosis']
                                    for p in rep.parameters} for rq1metric in RQM1_metrics}}

for sn in sample_names:
    RQM1_DICT[sn] = {rq1metric: {p: rep.distr_type[sn][p]['kurtosis']
                                 for p in rep.parameters} for rq1metric in RQM1_metrics}

rep.test()
rep.performance

rep.performance.keys()

rep.performance[sample_names[0]].keys()



rep.performance[sample_names[0]]['area']['correlation']
# =====================================================
rep.performance[sample_names[0]]['area']['mannwhitneyu']
rep.performance[sample_names[1]]['area']['mannwhitneyu']
rep.performance[sample_names[2]]['area']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['area']['kruskalwallis']
rep.performance[sample_names[1]]['area']['kruskalwallis']
rep.performance[sample_names[2]]['area']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['area']['ks']
rep.performance[sample_names[1]]['area']['ks']
rep.performance[sample_names[2]]['area']['ks']
# =====================================================
rep.performance[sample_names[0]]['perimeter']['mannwhitneyu']
rep.performance[sample_names[1]]['perimeter']['mannwhitneyu']
rep.performance[sample_names[2]]['perimeter']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['perimeter']['kruskalwallis']
rep.performance[sample_names[1]]['perimeter']['kruskalwallis']
rep.performance[sample_names[2]]['perimeter']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['perimeter']['ks']
rep.performance[sample_names[1]]['perimeter']['ks']
rep.performance[sample_names[2]]['perimeter']['ks']
# =====================================================
rep.performance[sample_names[0]]['aspect_ratio']['mannwhitneyu']
rep.performance[sample_names[1]]['aspect_ratio']['mannwhitneyu']
rep.performance[sample_names[2]]['aspect_ratio']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['aspect_ratio']['kruskalwallis']
rep.performance[sample_names[1]]['aspect_ratio']['kruskalwallis']
rep.performance[sample_names[2]]['aspect_ratio']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['aspect_ratio']['ks']
rep.performance[sample_names[1]]['aspect_ratio']['ks']
rep.performance[sample_names[2]]['aspect_ratio']['ks']
# =====================================================
rep.performance[sample_names[0]]['eq_diameter']['mannwhitneyu']
rep.performance[sample_names[1]]['eq_diameter']['mannwhitneyu']
rep.performance[sample_names[2]]['eq_diameter']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['eq_diameter']['kruskalwallis']
rep.performance[sample_names[1]]['eq_diameter']['kruskalwallis']
rep.performance[sample_names[2]]['eq_diameter']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['eq_diameter']['ks']
rep.performance[sample_names[1]]['eq_diameter']['ks']
rep.performance[sample_names[2]]['eq_diameter']['ks']
# =====================================================
rep.performance[sample_names[0]]['perimeter_crofton']['mannwhitneyu']
rep.performance[sample_names[1]]['perimeter_crofton']['mannwhitneyu']
rep.performance[sample_names[2]]['perimeter_crofton']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['perimeter_crofton']['kruskalwallis']
rep.performance[sample_names[1]]['perimeter_crofton']['kruskalwallis']
rep.performance[sample_names[2]]['perimeter_crofton']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['perimeter_crofton']['ks']
rep.performance[sample_names[1]]['perimeter_crofton']['ks']
rep.performance[sample_names[2]]['perimeter_crofton']['ks']
# =====================================================
rep.performance[sample_names[0]]['compactness']['mannwhitneyu']
rep.performance[sample_names[1]]['compactness']['mannwhitneyu']
rep.performance[sample_names[2]]['compactness']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['compactness']['kruskalwallis']
rep.performance[sample_names[1]]['compactness']['kruskalwallis']
rep.performance[sample_names[2]]['compactness']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['compactness']['ks']
rep.performance[sample_names[1]]['compactness']['ks']
rep.performance[sample_names[2]]['compactness']['ks']
# =====================================================
rep.performance[sample_names[0]]['solidity']['mannwhitneyu']
rep.performance[sample_names[1]]['solidity']['mannwhitneyu']
rep.performance[sample_names[2]]['solidity']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['solidity']['kruskalwallis']
rep.performance[sample_names[1]]['solidity']['kruskalwallis']
rep.performance[sample_names[2]]['solidity']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['solidity']['ks']
rep.performance[sample_names[1]]['solidity']['ks']
rep.performance[sample_names[2]]['solidity']['ks']
# =====================================================
rep.performance[sample_names[0]]['feret_diameter']['mannwhitneyu']
rep.performance[sample_names[1]]['feret_diameter']['mannwhitneyu']
rep.performance[sample_names[2]]['feret_diameter']['mannwhitneyu']
# ----------------
rep.performance[sample_names[0]]['feret_diameter']['kruskalwallis']
rep.performance[sample_names[1]]['feret_diameter']['kruskalwallis']
rep.performance[sample_names[2]]['feret_diameter']['kruskalwallis']
# ----------------
rep.performance[sample_names[0]]['feret_diameter']['ks']
rep.performance[sample_names[1]]['feret_diameter']['ks']
rep.performance[sample_names[2]]['feret_diameter']['ks']
# =====================================================
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

MPOI_df_PValues = {"area": None,
                   "perimeter": None,
                   "aspect_ratio": None,
                   "major_axis_length": None,
                   "minor_axis_length": None,
                   "compactness": None,
                   "solidity": None,
                   "eccentricity": None,}
MPOI_df_DValues = {"area": None,
                   "perimeter": None,
                   "aspect_ratio": None,
                   "major_axis_length": None,
                   "minor_axis_length": None,
                   "compactness": None,
                   "solidity": None,
                   "eccentricity": None,}

SUBPLOT_ARRANGEMENT = (2, 4)

Y = ["MWU", "KW", "KS"]
import pandas as pd
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

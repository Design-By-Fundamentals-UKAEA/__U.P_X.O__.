import pandas as pd
import numpy as np
import defdap.ebsd as defDap_ebsd
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import matplotlib.pyplot as plt
# ------------------------------------------
'''
fileName = "D://EBSD Datasets//OFHC_Cu_dogbone_MAPEXTRACTS_DRF1//Full_map__DRF1__LATh4-map.ctf"
fileName = "D://EBSD Datasets//OFHC_Cu_Texture_dogbone_2um_step.crc"
target = defDap_ebsd.Map(str(fileName)[:-4])
target.buildQuatArray()
target.findBoundaries()
target.findGrains(minGrainSize=1)
target.calcGrainMisOri()
# target.filterData()
target_areas = np.array([len(grain.quatList) for grain in target.grainList])
target_areas = target_areas[np.argwhere(target_areas <= 10)].T.squeeze()
target_areas.size
'''
# ------------------------------------------
filename = "D://Admin//Task specification 24-25//repr - proof of use//areas.txt"
target_areas = pd.read_csv(filename, delimiter='\t', header=None).to_numpy().T.squeeze()
target_areas.size
target_areas = np.sort(target_areas)
target_areas.max()
target_areas.min()
# ------------------------------------------
fileName = "C://Development//M2MatMod//UPXO-CPFEM-Preperation//mcgs 100x100x100 S16//texture instance 1 mctime step 5//slice_0.ctf"
sample = defDap_ebsd.Map(str(fileName)[:-4],dataType="OxfordText")
sample.buildQuatArray()
sample.findBoundaries()
sample.findGrains(minGrainSize=16)
sample.calcGrainMisOri()
sample.plotEulerMap()
# sample.filterData()
sample_areas = np.array([len(grain.quatList) for grain in sample.grainList])
sample_areas.size
sample_areas.max()
sample_areas.min()
# ------------------------------------------
target_areas = target_areas[np.argwhere(target_areas <= sample_areas.max())].squeeze().T
target_areas.mean()
scale = sample_areas.mean()/target_areas.mean()
# ------------------------------------------
sns.histplot(target_areas, color='red', alpha=0.5, label='Target', kde=True, stat='probability', binwidth=2.5)
sns.histplot(sample_areas, color='blue', alpha=0.5, label='Sample', kde=True, stat='probability', binwidth=2.5)
plt.xlabel('Grain area, um^2')
plt.ylabel('Probability')
plt.show()

sns.histplot(target_areas, color='red', alpha=0.5, label='Target', kde=True)
plt.xlabel('Grain area, um^2')
plt.ylabel('Count')
plt.show()

sns.histplot(sample_areas, color='blue', alpha=0.5, label='Sample', kde=True)
plt.xlabel('Grain area, um^2')
plt.ylabel('Count')
plt.show()
# ------------------------------------------
# mwu_D, mwu_P = mannwhitneyu(areas, areas[:,np.newaxis][:120:-1,:].squeeze())
# print(mwu_P)
# ------------------------------------------
mwu_D, mwu_P = mannwhitneyu(target_areas, sample_areas/scale)
print(mwu_P)

mwu_D, mwu_P = mannwhitneyu(np.random.choice(target_areas, sample_areas.size), sample_areas/scale)
print(mwu_P)

kw_D, kw_P = kruskal(np.random.choice(target_areas, sample_areas.size), sample_areas/scale)
print(kw_P)
# ------------------------------------------

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:33:02 2024

@author: Dr. Sunil Anandatheertha
"""
from pathlib import Path
import matplotlib.pyplot as plt
from upxo.interfaces.user_inputs.excel_commons import read_excel_range

data_dir = Path(r"C:\Development\M2MatMod\upxo_packaged\upxo_private\data")
FileNames = ["repr01.xlsx", "repr02.xlsx", "repr03.xlsx", "repr04.xlsx"]
file_names = [str(data_dir / filename) for filename in FileNames]

sheet_names = ["Sheet1a", "Sheet1b", "Sheet1c", "Sheet1d"]
cell_starts = ['B10', 'B27', 'B44', 'B61', 'B78', 'B95', 'B112', 'B129']
cell_ends = ['L22', 'L39', 'L56', 'L73', 'L90', 'L107', 'L124', 'L141']
cell_ranges = [a+':'+b for a, b in zip(cell_starts, cell_ends)]

df_cp0_ngset0 = read_excel_range(file_names[0], sheet_names[0], cell_ranges[0])
df_cp1_ngset0 = read_excel_range(file_names[0], sheet_names[1], cell_ranges[0])
df_cp2_ngset0 = read_excel_range(file_names[0], sheet_names[2], cell_ranges[0])
df_cp3_ngset0 = read_excel_range(file_names[0], sheet_names[3], cell_ranges[0])

dframes = [df_cp0_ngset0, df_cp1_ngset0, df_cp2_ngset0, df_cp3_ngset0]
markers = ['o', 's', 'x', 'd', '<', 'h', '*', '^']
labels = ['Lath = 1.0, Outliers removed.',
          'Lath = 1.0, Outliers not removed.',
          'Lath = 0.0, Outliers removed.',
          'Lath = 0.0, Outliers not removed.']
"""
I am now looking at the effects of characterization parameters.
So, we will only focus on 100x100 data from repr01.xlsx and nothing else.
"""
fig = plt.figure(figsize=(5, 5), dpi=125)
axs = plt.gca()
plt.plot(df_cp0_ngset0['ngrains/Ng'], df_cp0_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='Lath = 1.0, Outliers removed.')
plt.plot(df_cp1_ngset0['ngrains/Ng'], df_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='Lath = 1.0, Outliers removed.')
plt.plot(df_cp2_ngset0['ngrains/Ng'], df_cp2_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='Lath = 1.0, Outliers removed.')
plt.plot(df_cp3_ngset0['ngrains/Ng'], df_cp3_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='Lath = 1.0, Outliers removed.')
plt.legend()
axs.set_xlabel('<Ng of subsets> / Ng of parent', fontsize=12)
axs.set_ylabel('Representativeness, R: KL-re T|S', fontsize=12)

"""
I am now looking at single pixel grains removed and outliers not removed.

This is case 2, i.e. characterisaion parameters 2, i.e cp1
So, we will focus on 100x100 data from repr01.xlsx, 200x200 data from repr02.xlsx
amd 500x500 data from repr03.xlsx. For all datasets, we will use the parent
set with the largewst number of grains, which is the first parent.

In the below codes, p0, p1 and p2 are parent 0, 1 and 2 respectively.

A general guide:
    * To change parent, change the excel file
    * To change cp (char. Parameter) change the sheet
    * To change the tslice of parent, i.e. to change the parent, change the
    cell_ranges
"""
df_p0_cp1_ngset0 = read_excel_range(file_names[0], sheet_names[0], cell_ranges[0])
df_p1_cp1_ngset0 = read_excel_range(file_names[1], sheet_names[0], cell_ranges[0])
df_p2_cp1_ngset0 = read_excel_range(file_names[2], sheet_names[0], cell_ranges[0])

fig = plt.figure(figsize=(5, 5), dpi=125)
axs = plt.gca()
plt.plot(df_p0_cp1_ngset0['ngrains/Ng'], df_p0_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='100x100 Lath = 1.0, Outliers not removed.')
plt.plot(df_p1_cp1_ngset0['ngrains/Ng'], df_p1_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='200x200 Lath = 1.0, Outliers not removed.')
plt.plot(df_p2_cp1_ngset0['ngrains/Ng'], df_p2_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='500x500 Lath = 1.0, Outliers not removed.')
plt.legend()
axs.set_xlabel('<Ng of subsets> / Ng of parent', fontsize=12)
axs.set_ylabel('Representativeness, R: KL-re T|S', fontsize=12)


fig = plt.figure(figsize=(7, 5), dpi=125)
axs = plt.gca()
plt.plot(df_p0_cp1_ngset0['n grains'], 1/ df_p0_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='100x100 Lath = 1.0, Outliers not removed.')
plt.plot(df_p1_cp1_ngset0['n grains'], 1/ df_p1_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='200x200 Lath = 1.0, Outliers not removed.')
plt.plot(df_p2_cp1_ngset0['n grains'], 1/ df_p2_cp1_ngset0['R density'],
         marker='o', mfc='None', ms=5, lw=1, alpha=1, label='500x500 Lath = 1.0, Outliers not removed.')
plt.legend()
axs.set_xlabel('<Ng of subsets> / Ng of parent', fontsize=12)
axs.set_ylabel('Representativeness, R: KL-re T|S', fontsize=12)
axs.set_xlim([0, 250])

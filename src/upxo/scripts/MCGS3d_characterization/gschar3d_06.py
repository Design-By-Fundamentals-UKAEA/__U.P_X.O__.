"""
Created on Tue Sep  3 14:25:14 2024

@author: Dr. Sunil Anandatheertha
"""
from upxo.ggrowth.mcgs import mcgs
# -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
pxt = mcgs()
pxt.simulate(verbose=False)
# ===================================================
tslice = 12
gstslice = pxt.gs[tslice]
# ===================================================
gstslice.char_morphology_of_grains()
gstslice.set_mprops()
# ===================================================
gstslice.sss_rel_morpho(slice_plane='xy', loc=25, reset_lgi=True,
                        slice_gschar_kernel_order=4,
                        mprop_names_2d=['eqdia', 'arbbox', 'solidity'],
                        mprop_names_3d=['eqdia', 'arbbox', 'solidity'],
                        ignore_border_grains_2d=True,
                        ignore_border_grains_3d=True,
                        reset_mprops=False, kdeplot=False,
                        save_plot3d_grains=True, save_plot2d_grains=True)

gstslice.sss_rel_morpho_multiple(slice_planes=['xy', 'yz', 'xz'],
                                 loc_starts=[0, 0, 0],
                                 loc_ends=[50, 50, 50],
                                 loc_incrs=[8, 8, 8],
                                 reset_lgi=True,
                                 slice_gschar_kernel_order=4,
                                 mprop_names_2d=['eqdia', 'arbbox', 'solidity'],
                                 mprop_names_3d=['eqdia', 'arbbox', 'solidity'],
                                 ignore_border_grains_2d=False,
                                 ignore_border_grains_3d=False,
                                 save_plot3d_grains=False,
                                 save_plot2d_grains=False,
                                 show_legends=False,
                                 identify_peaks=True,
                                 show_peak_location=True,
                                 cmp_peak_locations=True,
                                 cmp_distributions=True,
                                 plot_distribution_cmp=True,
                                 )

gstslice.sssr['viz3d'].show()
gstslice.sssr['viz2d'].show()

gstslice.clean_gs_GMD_by_source_erosion_v1(prop='volnv', threshold=1)

gstslice.sss_rel_morpho(slice_plane='xy', loc=0, reset_lgi=True,
                        slice_gschar_kernel_order=4,
                        mprop_names_2d=['eqdia', 'arbbox', 'solidity'],
                        mprop_names_3d=['eqdia', 'arbbox', 'solidity'],
                        ignore_border_grains_2d=True,
                        ignore_border_grains_3d=True,
                        reset_mprops=False, kdeplot=True,
                        save_plot3d_grains=True, save_plot2d_grains=True)

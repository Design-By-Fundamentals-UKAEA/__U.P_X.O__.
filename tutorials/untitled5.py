# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 03:50:09 2022

@author: rg5749
"""

############################################################
LEVEL 0 grain struture

High level morphological features:
	1. Grains          : FEATURE F0A
	2. Grain boundaries: FEATURE F0B
	3. Grain vertices  : 

Morphological feature related morphological base data:
	1. Number of external edges
	2. Number of internal edges

Morphological feature related coordinate data:
	1. Grain vertices cooridnates
	2. Grain centroid coordinates

Morphological feature related geometric-parameter data:
	1. Grain area
	2. External perimeter
	3. Internal perimeter
	4. Minimum diagonal
	5. Maximum diagonal
	6. Mean diagonal
	7. External vertices internal angle
	8. Centroid to external vertices distances
	9. Centroid to internal vertices distances - TO DO
	10. Triple point junctions - TO DO

Triple point junctions: FEATURE F0D - TODO

Statistical information for F0 series
	1. Area
	2. Internal perimeter
	3. External perimeter
	4. Number of external edges
	5. Number of internal edges
	6. Minimum diagonal
	7. Maximum diagonal
	8. Mean diagonal
	9. Standard deviation diagonal
	10. Internal angles on the vertices of the external boundary
	11. Centroid to external vertices distances
############################################################
DONE: gen_RAND_Lattice_Coord -- TODO: operate_HEX_LatCoord_apply_gradient
DONE: gen_REC_Lattice_Coord -- DONE: operate_REC_LatCoord_apply_gradient
DONE: gen_HEX_Lattice_Coord -- TODO: operate_HEX_LatCoord_apply_gradient
DONE: gen_TRI_REC_mixed_Lattice_Coord -- TODO: operate_TRI_LatCoord_apply_gradient
DONE: gen_TRI_HEX_mixed_Lattice_Coord -- TODO: operate_TRI_HEX_LatCoord_apply_gradient
DONE: gen_mixed_Lattice_Coord -- TODO: operate_mixed_LatCoord_apply_gradient
DONE: gen_TRI_Lattice_Coord -- TODO: operate_TRI_LatCoord_apply_gradient
DONE: add_Perturb_coordinates -- TODO: Add Gaussian noise sub-definition
DONE: vis_coords
DONE: form_2D_Coord_Array
DONE: form_2D_VorTess_Seeds
DONE: form_2D_VorTess_Object
DONE: vis_Vor
DONE: voronoi_finite_polygons_2d
DONE: ini_Grain_param
DONE: make_Super_Bounding_Polygon
DONE: L0GS_calc_num_VorGrains
DONE: build_GSL0_idname
DONE: build_GSL0_vid
DONE: build_GSL0_edgeid
DONE: build_GSL0_ccoord
DONE: build_GSL0_area
DONE: build_GSL0_extperim
DONE: build_GSL0_intperim
DONE: build_GSL0_nextedges
DONE: build_GSL0_nedgesextv
DONE: build_GSL0_mindiag
DONE: build_GSL0_maxdiag
DONE: build_GSL0_meandiag
DONE: build_GSL0_stddiag
DONE: build_GSL0_extvintangle
DONE: build_GSL0_ctovdist
DONE: build_hist_data
DONE: build_GSL0_pdstr_area
DONE: build_GSL0_pdstr_extperim
DONE: build_GSL0_pdstr_intperim
DONE: build_GSL0_pdstr_nextedges
DONE: build_GSL0_pdstr_nedgesextv
DONE: build_GSL0_pdstr_mindiag
DONE: build_GSL0_pdstr_maxdiag
DONE: build_GSL0_pdstr_extvintangle
DONE: build_GSL0_pdstr_ctovdist
DONE: pop_GSL0_idname
DONE: pop_GSL0_edgeid
DONE: pop_GSL0_vid
DONE: pop_GSL0_vcoord
DONE: pop_GSL0_ccoord
DONE: pop_GSL0_area
DONE: pop_GSL0_extperim
DONE: pop_GSL0_intperim
DONE: pop_GSL0_nextedges
DONE: pop_GSL0_nedgesextv
DONE: pop_GSL0_mindiag
DONE: pop_GSL0_maxdiag
DONE: pop_GSL0_meandiag
DONE: pop_GSL0_stddiag
DONE: pop_GSL0_extvintangle
DONE: pop_GSL0_ctovdist
DONE: extract_vcoord_POU4
DONE: extract_ccoord_POU
DONE: extract_area_POU
DONE: extract_extperim_POU
DONE: extract_edges_POU
DONE: calc_distance_matrix
DONE: calc_edgelengths_POU
DONE: calc_diaglengths_POU
DONE: calc_maxdiag_POU
DONE: calc_mindiag_POU
DONE: calc_meandiag_POU
DONE: calc_stddiag_POU
DONE: calc_diaglength_stats
DONE: calc_L0GS_histogram_data_1d
DONE: build_GSL0_STAT_KDEdata
DONE: build_GSL0_STAT_Peaks_paramDistr
DONE: build_GSL0_STAT_SkewKurt_paramDistr
DONE: stat_GSL0_estimate_SciPi_KDE_param
DONE: stat_GSL0_identify_PEAK_KDEparam
DONE: stat_GSL0_estimate_skew_kurt_param
DONE: vis_stat_hist_kde
DONE: vis_stat_kde_Peaks
DONE: vis_annotation_stat_kde_Peaks
DONE: vis_L0GS_GrainStructure
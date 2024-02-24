import numpy as np
from shapely.ops import voronoi_diagram
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint
from shapely.ops import polygonize
from shapely.ops import split, SplitOp
from shapely.ops import voronoi_diagram
from shapely import affinity
import matplotlib.pyplot as plt
import itertools
def plot_pxtal(ax_obj, **kwargs):
#               bool_xtal           = True,
#               obj_xtal_list       = None,
    #--------------------------------------------------------
    # list of all xtal objects
    if 'xtal_list' in kwargs.keys():
        bool_xtal == kwargs['bool_xtal']
        if bool_xtal:
            if 'obj_xtal_list' not in kwargs.keys():
                bool_xtal == False
    else:
        bool_xtal == False
    #--------------------------------------------------------
    # list of xtals to highlight
    if 'xtals_to_highlight' in kwargs.keys():
        bool_xtal == kwargs['bool_xtal']
        if bool_xtal:
            if 'obj_xtal_list' not in kwargs.keys():
                bool_xtal == False
    else:
        bool_xtal == False
    #--------------------------------------------------------
    # list of boundary edge objects of all xtals
    if 'xtal_be_list' in kwargs.keys():
        bool_gbe_edges = kwargs['bool_gbe_edges']
        if bool_gbe_edges:
            if 'obj_xtal_list' not in kwargs.keys():
                bool_xtal == False
    else:
        bool_gbe_edges = False
    #--------------------------------------------------------
    # list of centroid point objects of all xtals
    if 'xtal_centroid_list' in kwargs.keys():
        bool_xtal_centroids = kwargs['bool_xtal_centroids']
    else:
        bool_xtal_centroids = False
    #--------------------------------------------------------
    # list of representative points, each for an xtal
    if 'xtal_rep_point' in kwargs.keys():
        bool_rep_point = kwargs['bool_rep_point']
    else:
        bool_rep_point = False
    #--------------------------------------------------------
    # 
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------
    #--------------------------------------------------------

#               bool_xtal_centroids    = True,
#               obj_xtal_centroid_list = None,

#               bool_xtal_rep_point = True,
#               obj_xtal_rep_point  = None,

#               bool_xbe_edges  = False,
#               obj_xbe_list    = None,

#               xtals_to_highlight  = None,
#               pxtal_level         = 0,

#               bool_gbe_vertices   = True,

#               bool_overlay_grid   = True,
#               fig_size            = (3.5, 3.5),
#               fig_dpi             = 100,
#               clr_L0_xtal         = 'white',
#               clr_L0_gbe_L0       = 'black',
#               clr_L1_xtal_gbz     = 'lightgrey',
#               clr_L1_xtal_gcz     = 'darkgray',
#               clr_L1_xtal_twi     = 'yellow',
#               clr_L1_xtal_pap     = 'lime',
#               clr_L2_xtal_block   = 'cyan',
#               clr_L3_xtal_lath    = 'blue',
#               alpha_xtal_face     = 1.0,
#               lw_viz_gbe          = 1,
#               ann_text_xtal_n     = True,
#               file                = ('Grain_Structure.png')

    colors = ['chocolate', 'gold', 'blue', 'crimson', 'khaki',
              'orange', 'grey', 'darkgrey', 'lavender', 'lightblue',
              'darkblue', 'lightgreen', 'indigo', 'lime', 'magenta',
              'orangered', 'tomato', 'tan', 'teal', 'violet',
              'yellowgreen']
    
        
    if isinstance(xtals_to_highlight, list) and len(xtals_to_highlight) > 0 and xtals_to_highlight != None:
        xtal_fill_color = []
        xtal_count = 0
        for xtal in xtal_list:
            if xtal_count in xtals_to_highlight:
                xtal_fill_color.append('lime')
            else:
                xtal_fill_color.append(vis_parameters['vis_xtal_clr'])
            xtal_count += 1
    else:
        xtal_fill_color = []
        for xtal in xtal_list:
            xtal_fill_color.append(vis_parameters['vis_xtal_clr'])
        
    xtal_count = 0
    for xtal in xtal_list:
        if vis_pxtal_controls['vis_xtal_face_bool']:
            gb_xy = xtal.boundary.xy
            ax_obj.fill(gb_xy[0], gb_xy[1],
                        facecolor = xtal_fill_color[xtal_count],
                        edgecolor = vis_parameters['vis_gbe_clr'],
                        alpha = vis_parameters['vis_xtal_alpha'],
                        linewidth = vis_parameters['vis_gbe_lw']
                        )
        if vis_pxtal_controls['vis_xtal_centroids_bool']:
            ax_obj.text(xtal_centroid_list[xtal_count].x,
                        xtal_centroid_list[xtal_count].y,
                        str(xtal_count),
                        fontsize = 10
                        )
        xtal_count += 1

def plot_grid_overlay(x, y, ax_obj):
    ax_obj.scatter(x, y, s = 1, marker = '.', color = 'blue')
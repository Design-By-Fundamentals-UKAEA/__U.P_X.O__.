import numpy as np
from shapely.ops import voronoi_diagram
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint
from shapely.ops import polygonize
from shapely.ops import split, SplitOp
from shapely.ops import voronoi_diagram
from shapely import affinity
import matplotlib.pyplot as plt
import itertools
# =============================================================================
# def plot_pxtal(ax_obj,
#                xtals_to_highlight,
#                vis_pxtal_controls,
#                vis_parameters):
# =============================================================================
def plot_pxtal(xtal_list_xtal,
               xtals_to_highlight,
               xtal_centroid_list,
               vis_pxtal_controls,
               vis_parameters,
               ax_obj):

    colors = ['chocolate', 'gold', 'blue', 'crimson', 'khaki',
              'orange', 'grey', 'darkgrey', 'lavender', 'lightblue',
              'darkblue', 'lightgreen', 'indigo', 'lime', 'magenta',
              'orangered', 'tomato', 'tan', 'teal', 'violet',
              'yellowgreen']
    
        
    if isinstance(xtals_to_highlight, list) and len(xtals_to_highlight) > 0 and xtals_to_highlight != None:
        xtal_fill_color = []
        xtal_count = 0
        for xtal in xtal_list_xtal:
            if xtal_count in xtals_to_highlight:
                xtal_fill_color.append('lime')
            else:
                xtal_fill_color.append(vis_parameters['xtal_clr'])
            xtal_count += 1
    else:
        xtal_fill_color = []
        for xtal in xtal_list_xtal:
            xtal_fill_color.append(vis_parameters['xtal_clr'])
        
    xtal_count = 0
    for xtal in xtal_list_xtal:
        if vis_pxtal_controls['xtal_face_bool']:
            gb_xy = xtal.boundary.xy
            ax_obj.fill(gb_xy[0], gb_xy[1],
                        facecolor = xtal_fill_color[xtal_count],
                        edgecolor = vis_parameters['gbe_clr'],
                        alpha = vis_parameters['xtal_face_alpha'],
                        linewidth = vis_parameters['gbe_lw']
                        )
        if vis_pxtal_controls['xtal_centroids_bool']:
            ax_obj.text(xtal_centroid_list[xtal_count].x,
                        xtal_centroid_list[xtal_count].y,
                        str(xtal_count),
                        fontsize = 10
                        )
        xtal_count += 1

def plot_grid_overlay(x, y, ax_obj):
    ax_obj.scatter(x, y, s = 1, marker = '.', color = 'blue')
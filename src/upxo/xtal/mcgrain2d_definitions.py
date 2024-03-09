import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label as skim_label

class grain2d():
    """
    brec: bounding rectangle of the grain
    bbox: bounding box
    bbox_ex: bbox expanded (by unit pixel along permitted direction)

    gid: This grain ID. Same as the lgi number of this grain
    gbid: Grain boundary ID
    gind: Indices of grain pixel in the parent state matrix
    gbind: Indices of grain boundary pixel in the parent state matrix
    gbsegs_pre: Partially processed grain boundary segments

    s: State value of the grain
    sn: Number (n) of this grain in the list of 's' stated grains

    neigh: list of neighbouring grain ids --> User calculated

    grep_cen: Geometric repr: centroid point --> User calculated
    grep_mp: Geometric repr: UPXO multi-point object of grain boundary --> User calculated
    grep_me: Geometric repr: UPXO multi-edge object of grain boundary --> User calculated
    grep_ring: Geometric representation: UPXO ring object of grain boundary --> User calculated

    feat_bbox: Feature: Bounding box <<-- STORED UPON CREATION
    feat_islands: Feature: Island grains
    feat_gc: Feature: Grain core
    feat_gbz: Feature: Grain boundary zone
    feat_ell: Feature: Ellipse --> User calculated
    feat_rec: Feature: Rectangle --> User calculated

    x_xsys: Crystal system: 'fcc', 'bcc', 'hcp': USER INPUT / EXTRACTED
    x_phase: Phase ID: USER INPUT / EXTRACTED
    x_mean: Mean crystallographic orientation: Bunge's Euler angle: USER INPUT / EXTRACTED

    gbvert: Grain boundary vertices <<-- STORED UPON CREATION
    gbseg: Grain boundary segments --> User calculated

    _xgr_min_: Minimum value of the xgr of the parent grain structure
    _xgr_max_: Maximum value of the xgr of the parent grain structure
    _xgr_incr_: Increment value of the xgr of the parent grain structure

    _ygr_min_: Minimum value of the ygr of the parent grain structure
    _ygr_max_: Maximum value of the ygr of the parent grain structure
    _xgr_incr_: Increment value of the ygr of the parent grain structure
    """
    __slots__ = ('loc',
                 'position',
                 'coords',
                 'gbloc',
                 'brec',
                 'bbox_bounds',
                 'bbox_ex_bounds',
                 'bbox',
                 'bbox_ex',
                 'px_area',
                 'skprop',
                 'gid',
                 'gbid',
                 'gind',
                 'gbind',
                 'gbsegs_pre',
                 's',
                 'sn',
                 'neigh',
                 'grep_cen',
                 'grep_mp',
                 'grep_me',
                 'grep_ring',
                 'feat_bbox',
                 'feat_islands',
                 'feat_gc',
                 'feat_gbz',
                 'feat_ell',
                 'feat_rec',
                 'x_xsys',
                 'x_phase',
                 'x_mean',
                 'gbvert',
                 'gbseg',
                 'xmin',
                 'xmax',
                 'ymin',
                 'ymax'
                 )

    def __init__(self):
        self.loc = None
        self.gbloc = None
        self.brec = None
        self.bbox_bounds = None
        self.bbox_ex_bounds = None
        self.bbox = None
        self.bbox_ex = None
        self.px_area = None
        self.skprop = None
        self.gid = None
        self.gind = None
        self.s = None
        self.sn = None
        self.position = None
        self.coords = None
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None

    def __str__(self):
        return f's{self.s}, sn{self.sn}'

    def __len__(self):
        return len(self.loc)

    def __mul__(self, k):
        self.px_area *= k

    def __att__(self):
        return gops.att(self)

    def make_prop(self, generator, skprop=True):
        if skprop:
            self.skprop = generator(self.bbox_ex, cache=False)[0]

    @property
    def centroid(self):
        coords = self.coords.T
        return (coords[0].mean(), coords[1].mean())

    def plot(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.bbox_ex)
        plt.title(f"Grain plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$")
        if not hold_on:
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgb(self, hold_on=False):
        z = np.zeros_like(self.bbox_ex)
        rmin = self.bbox_ex_bounds[0]
        cmin = self.bbox_ex_bounds[2]
        for rc in self.gbloc:
            z[rc[0]-rmin, rc[1]-cmin] = 1
        # ------------------------------------
        if not hold_on:
            plt.figure()
        plt.imshow(z)
        if not hold_on:
            plt.title(f"Grain boundary plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$. Perimeter: {round(self.skprop.perimeter*10000)/10000} $\mu m$")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgbseg(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.gbsegs_pre)
        if not hold_on:
            plt.title(f"Grain boundary segment plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$. Perimeter: {round(self.skprop.perimeter*10000)/10000} $\mu m$")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

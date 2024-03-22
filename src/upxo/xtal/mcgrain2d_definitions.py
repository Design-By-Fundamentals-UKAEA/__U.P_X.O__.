import numpy as np
import matplotlib.pyplot as plt
# import cv2
# from skimage.measure import label as skim_label
import upxo._sup.gops as gops


class grain2d():
    """
    ATTRIBUTES AND THEIR ACCESS
    ------------------------------
    loc:
        Grain pixel loca5tions.
        pxt.gs[t].g[gid]['grain'].loc, pxt.gs[t].g[gid]['grain'].coords

    npixels:
        Number of pixels in the grain
        pxt.gs[t].g[gid]['grain'].npixels

    position:
        Grain relative poistioning in the poly-crystal.
        pxt.gs[t].g[gid]['grain'].position
        initial 2 values provide x and y coord of the centroid
        last string value provides the relative positioning of the grain

    coords:
        Explanation here

    gbloc:
        grain boundary pixel locations

    bbox_bounds:
        Bounds of the bounding box
        pxt.gs[t].g[gid]['grain'].bbox_bounds

    bbox_ex_bounds:
        Bounds of the extended bounding box. Extended by unit pixels on all
        sides. If grain is a boundary grain, only possible directions  will
        be extended by unit pixel.
        pxt.gs[t].g[gid]['grain'].bbox_ex_bounds
    bbox:
        Mask on bounding box.
        pxt.gs[t].g[gid]['grain'].bbox
    bbox_ex:
        Mask on extended bouning box.
        pxt.gs[t].g[gid]['grain'].bbox_ex

    skprop:
        Scikit image property generator
        pxt.gs[t].g[gid]['grain'].skprop

    _px_area:
        Area of a single pixel
        pxt.gs[t].g[gid]['grain']._px_area

    gid:
        This grain ID. Same as the lgi number of this grain.
        pxt.gs[t].g[gid]['grain'].gid

    gind:
        Indices of grain pixel in the parent state matrix
        pxt.gs[t].g[gid]['grain'].gind

    gbid:
        Grain boundary ID
        pxt.gs[t].g[gid]['grain'].gbid

    gbind:
        Indices of grain boundary pixel in the parent state matrix
        pxt.gs[t].g[gid]['grain'].gbind

    gbvert:
        Grain boundary vertices
        pxt.gs[t].g[gid]['grain'].gbvert

    gbsegs:
        grain boundary segments
        pxt.gs[t].g[gid]['grain'].gbsegs

    gbsegs_geo:
        grain boundary segments: dict with below keys:
        'info': OPTIONS:
            'spg.menp': Single Pixel Grain. Multi-edge not possible
            'slg.menp': Straight line Grain. Multi-edge not possible
        'me': Multi-edge
        pxt.gs[t].g[gid]['grain'].gbsegs_geo

    s:
        Monte-Car;o State value of the current grain
        pxt.gs[t].g[gid]['grain'].s

    sn:
        pxt.gs[t].g[gid]['grain'].sn

    neigh:
        GIDs of the neighbouring grains
        pxt.gs[t].g[gid]['grain'].neigh

    precipitates:
        Precipitates inside the present grain
        pxt.gs[t].g[gid]['grain'].precipitates

    grain_core:
        Grain core region
        pxt.gs[t].g[gid]['grain'].grain_core

    gb_zone:
        Grain boundary zone
        pxt.gs[t].g[gid]['grain'].gb_zone

    subgrains:
        sub-grains inside grain. It shall also encompass any
        island_grains
        pxt.gs[t].g[gid]['grain'].subgrains

    paps:
        prior-austenite packats
        pxt.gs[t].g[gid]['grain'].paps

    blocks:
        block structuers in paps
        pxt.gs[t].g[gid]['grain'].blocks

    laths:
        Lath structures inside blocks
        pxt.gs[t].g[gid]['grain'].laths

    xstruc: Crystal Structure: 'fcc', 'bcc', 'hcp'

    xmin, xmax, ymin, ymax:
        pxt.gs[t].g[gid]['grain'].(xmin, xmax, ymin, ymax)

    control_points_mesh:
        Points places at strategic locations inside the grain to control
        mesh density of a conformal mesh
        pxt.gs[t].g[gid]['grain'].control_points_mesh

    ea_pixels:
        Euler snagles of all pixels

    quats_pixels:
        Quaternions of all the pixesl int he grain

    ref_quat:
        Reference quaternion value of the all pixes in the grain

    ref_ea:
        Referecne Euler angle of all the pixesl in the grain

    texcomp:
        T3exture component name to which the pixel would belong to.
        Defined at each of the pixesl in the grain

    glb_pert_min_ea1,  glb_pert_max_ea1:
        Global minimum and maximum allowed perturbation to euler angle 1
    glb_pert_min_ea2,  glb_pert_max_ea2:
        Global minimum and maximum allowed perturbation to euler angle 2
    glb_pert_min_ea3,  glb_pert_max_ea3
        Global minimum and maximum allowed perturbation to euler angle 3
    lcl_pert_min_ea1, lcl_pert_max_ea1:
        Local minimum and maximum allowed perturbation to euler angle 1
    lcl_pert_min_ea2, lcl_pert_max_ea2:
        Local minimum and maximum allowed perturbation to euler angle 2
    lcl_pert_min_ea3, lcl_pert_max_ea3:
        Local minimum and maximum allowed perturbation to euler angle 3
    --------------------------------------------------------------------------
    _xgr_min_: Minimum value of the xgr of the parent grain structure
    _xgr_max_: Maximum value of the xgr of the parent grain structure
    _xgr_incr_: Increment value of the xgr of the parent grain structure

    _ygr_min_: Minimum value of the ygr of the parent grain structure
    _ygr_max_: Maximum value of the ygr of the parent grain structure
    _xgr_incr_: Increment value of the ygr of the parent grain structure
    """

    __slots__ = ('loc', 'npixels', 'position', 'coords', 'gbloc', 'brec',
                 'bbox_bounds', 'bbox_ex_bounds', 'bbox', 'bbox_ex', 'skprop',
                 '_px_area', 'gid', 'gind', 'gbid', 'gbind', 'gbvert',
                 'gbsegs', 'gbsegs_geo', 's', 'sn', 'neigh',
                 'precipitates', 'grain_core', 'gb_zone', 'subgrains', 'paps',
                 'blocks', 'laths',
                 'xstruc', 'xmin', 'xmax', 'ymin', 'ymax',
                 'control_points_mesh',
                 'ea_pixels', 'quats_pixels', 'ref_quat', 'ref_ea', 'texcomp',
                 'euler1', 'euler2', 'euler3',
                 'glb_pert_min_ea1', 'glb_pert_max_ea1', 'glb_pert_min_ea2',
                 'glb_pert_max_ea2', 'glb_pert_min_ea3', 'glb_pert_max_ea3',
                 'lcl_pert_min_ea1', 'lcl_pert_max_ea1', 'lcl_pert_min_ea2',
                 'lcl_pert_max_ea2', 'lcl_pert_min_ea3', 'lcl_pert_max_ea3',
                 )

    def __init__(self):
        # Set position/location related slots
        self.loc, self.position = None, None
        self.coords, self.gbloc = None, None
        # set bounds related
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.brec, self.bbox_bounds, self.bbox_ex_bounds = None, None, None
        # set masks
        self.bbox, self.bbox_ex = None, None
        # Set local neighbourhood related slots
        self.neigh = None
        # set properties
        self.npixels, self._px_area, self.skprop = None, None, None
        # set state related slots
        self.s, self.sn = None, None
        # set grain indices related slots
        self.gid, self.gind = None, None
        # Set grain boundary indices related slots
        self.gbid, self.gbind = None, None
        # Set grain boundaryt points
        self.gbvert = None
        # Set grain bounadryu segfments
        self.gbsegs, self.gbsegs_geo = None, None
        # FEATURES
        self.precipitates, self.grain_core, self.gb_zone = None, None, None
        self.subgrains = None
        self.paps, self.blocks, self.laths = None, None, None
        # MESHING RELATED DATA
        self.control_points_mesh = None
        # PHASE RELATED
        self.xstruc = 'fcc'
        # ORIENTATION RELATED
        self.ea_pixels, self.quats_pixels = None, None
        self.euler1, self.euler2, self.euler3 = None, None, None
        self.ref_quat, self.ref_ea = None, None
        self.glb_pert_min_ea1, self.glb_pert_max_ea1 = None, None
        self.glb_pert_min_ea2, self.glb_pert_max_ea2 = None, None
        self.glb_pert_min_ea3, self.glb_pert_max_ea3 = None, None
        self.lcl_pert_min_ea1, self.lcl_pert_max_ea1 = None, None
        self.lcl_pert_min_ea2, self.lcl_pert_max_ea2 = None, None
        self.lcl_pert_min_ea3, self.lcl_pert_max_ea3 = None, None
        self.texcomp = 'unknown'

    def __repr__(self):
        _repr_ = f'UPXO [{self.position[2]}] grain2d. GID:{self.gid}'
        _repr_ += f'-S:{self.s}'
        _repr_ += f'-Sn:{self.sn}'
        _repr_ += f'-Centroid:[{self.position[0]:.4f},{self.position[1]:.4f}]'
        return _repr_

    def __len__(self):
        return len(self.loc)

    def __lt__(self, _g):
        return self.npixels < self._g.npixels

    def __le__(self, _grain):
        return self.npixels <= self._g.npixels

    def __gt__(self, _grain):
        return self.npixels > self._g.npixels

    def __ge__(self, _grain):
        return self.npixels >= self._g.npixels

    def __mul__(self, k):
        self._px_area *= k

    def __att__(self):
        return gops.att(self)

    def set_glb_ea_pert(self, pert_ea1, pert_ea2, pert_ea3):
        self.glb_pert_min_ea1, self.glb_pert_max_ea1 = pert_ea1[0], pert_ea1[1]
        self.glb_pert_min_ea2, self.glb_pert_max_ea2 = pert_ea2[0], pert_ea2[1]
        self.glb_pert_min_ea3, self.glb_pert_max_ea3 = pert_ea3[0], pert_ea3[1]

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

from copy import deepcopy
import numpy as np

class ebsd_data():
    __slots__ = ('map_raw',
                 'map',
                 'gid',
                 'ea_avg',
                 'prop',
                 'n',
                 'quat_avg',
                 'filename'
                 )
    def __init__(self,
                 filename=None,
                 ):
        from defdap.quat import Quat
        self.filename = filename  # Cugrid_after 2nd_15kv_2kx_2
        self.load_ctf()

    def load_ctf(self):
        import defdap.ebsd as DEFDAP_EBSD
        self.map_raw = DEFDAP_EBSD.Map(self.filename,
                                       dataType="OxfordText")
        self.map = deepcopy(self.map_raw)

    def build_quatarray(self):
        self.map.buildQuatArray()

    def detect_grains(self, size_min=10):
        self.map.findGrains(minGrainSize=size_min)

    def calc_grain_ori_avg(self):
        self.map.calcGrainMisOri()
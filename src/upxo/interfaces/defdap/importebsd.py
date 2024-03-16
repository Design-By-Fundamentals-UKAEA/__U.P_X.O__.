from copy import deepcopy
import numpy as np
import defdap.ebsd as defDap_ebsd
from defdap.quat import Quat

class ebsd_data():
    """

    """
    __slots__ = ('map_raw',
                 'map',
                 'gid',
                 'ea_avg',
                 'prop',
                 'n',
                 'quat_avg',
                 'fileName')
    def __init__(self,
                 fileName=None,
                 ):
        self.fileName = fileName  # Cugrid_after 2nd_15kv_2kx_2
        self.load_ctf()

    def load_ctf(self):
        self.map_raw = defDap_ebsd.Map(self.fileName,
                                       dataType="OxfordText")
        self.map = deepcopy(self.map_raw)

    def build_quatarray(self):
        self.map.buildQuatArray()

    def detect_grains(self, size_min=10):
        self.map.findGrains(minGrainSize=size_min)

    def calc_grain_ori_avg(self):
        self.map.calcGrainMisOri()

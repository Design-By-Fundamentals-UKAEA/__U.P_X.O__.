from copy import deepcopy
import numpy as np
import defdap.ebsd as defDap_ebsd
# from defdap.quat import Quat
from scipy.ndimage import generic_filter
from upxo._sup.validation_values import _validation

class ebsd_data():
    """
    """
    __slots__ = ('map_raw',
                 'map', 'gbjp',
                 'gid',
                 'ea_avg',
                 'prop',
                 'n',
                 'quat_avg',
                 'fileName',
                 'val')

    def __init__(self,
                 fileName=None,
                 ):
        self.val = _validation()
        # VALIDATE WHETHER FILENAME HAS EXTENSION
        # VALIDATE WHETHER FILE EXTENSION IS A PERMITTED ONE
        print('-----------------------')
        print(fileName)
        print('-----------------------')
        if str(fileName)[-4:] == '.ctf':
            self.fileName = fileName
            self.load_ctf()
        self.gbjp = None

    def load_ctf(self):
        # VALIDATE IF FILE EXISTS
        self.map_raw = defDap_ebsd.Map(str(self.fileName)[:-4],
                                       dataType="OxfordText")
        self.map = deepcopy(self.map_raw)

    def _port_defdap_to_upxo(self):
        # FIRST UNPACK THE DEFDAP DATASETS TO UPXO NATIVE
        self._unpack_defdap()

    def _unpack_defdap(self):
        '''
        Following defdap data-sets will be unpacked to UPXO
        ---------------------------------------------------
        defdep_map.grains
        defdep_map.eulerAngleArray
        defdep_map.quatArray
        defdep_grainList[0].coordList
        '''
        self.lgi = self.map.grains
        self.ea = self.map.eulerAngleArray
        self.quats = self.quatArray
        self.g_ref_orientation

    def build_quatarray(self):
        self.map.buildQuatArray()

    def detect_grains(self, size_min=10):
        self.map.findGrains(minGrainSize=size_min)

    def calc_grain_ori_avg(self):
        self.map.calcGrainMisOri()

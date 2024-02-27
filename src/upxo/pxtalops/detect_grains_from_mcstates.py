import numpy as np
from upxo._sup import dataTypeHandlers as dth
from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as GS2d


def identify_grains_upxo_2d(S):
    """
    Deprecated.
    """

    # NOT OPERATIONAL WITH NEW MODIFICATIONS
    Nx = S.shape[0]
    Ny = S.shape[1]
    grains = []
    visited = np.zeros_like(S, dtype=bool)
    while np.sum(visited) < Nx*Ny:
        # Get the indices of the first unvisited cell
        i, j = np.unravel_index(np.argmax(~visited), (Nx, Ny))
        # Flood-fill to find the grain
        grain_label, grain, stack = S[i, j], [], [(i, j)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= Nx or y >= Ny or visited[x, y] or S[x, y] != grain_label:
                continue
            visited[x, y] = True
            grain.append((x, y))
            stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
        grains.append(grain)
    return grains

def find_grains2d_mcgs(library=None,
                       all_temporal_slices=None,
                       mlist=None,
                       ):
    if library == 'upxo':
        print('TODO: INclude call to def: identify_grains_upxo_2d')
    elif library in ('opencv', 'scikit-image'):
        _S_ = self.gs[m].s
        for i, _s_ in enumerate(np.unique(_S_)):
            # Mark the presence of this state
            self.gs[m].spart_flag[_s_] = True
            # Recognize the grains belonging to this state
            image = (_S_ == _s_).astype(np.uint8) * 255
            _, labels = cv2.connectedComponents(image)
            if i == 0:
                self.gs[m].lgi = labels
            else:
                labels[labels > 0] += self.gs[m].lgi.max()
                self.gs[m].lgi = self.gs[m].lgi + labels
            self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
        # Get the total number of grains
        self.gs[m].n = np.unique(self.gs[m].lgi).size
        # Generate and store the gid-s mapping
        self.gs[m].gid = list(range(1, self.gs[m].n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                               self.gs[m].s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                _gid_s_.append(0)
        self.gs[m].gid_s = _gid_s_
        # Make the output string to print on promnt
        optput_string_01 = f'Temporal slice number = {m}.'
        optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
        print(optput_string_01 + optput_string_02)


def find_grains_opencv_2d(self,
                          m,
                          ):
    """
    Detect grains using the library 'opencv'.
    m is a single temporal slice number.
    Note:
        This method is for internal call. It is recommended that the user
        uses the method 'detect_grains(M)' instead.

    Parameters
    ----------
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    _S_ = self.gs[m].s
    for i, _s_ in enumerate(np.unique(_S_)):
        # Mark the presence of this state
        self.gs[m].spart_flag[_s_] = True
        # Recognize the grains belonging to this state
        image = (_S_ == _s_).astype(np.uint8) * 255
        _, labels = cv2.connectedComponents(image)
        if i == 0:
            self.gs[m].lgi = labels
        else:
            labels[labels > 0] += self.gs[m].lgi.max()
            self.gs[m].lgi = self.gs[m].lgi + labels
        self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
        self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
    # Get the total number of grains
    self.gs[m].n = np.unique(self.gs[m].lgi).size
    # Generate and store the gid-s mapping
    self.gs[m].gid = list(range(1, self.gs[m].n+1))
    _gid_s_ = []
    for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                           self.gs[m].s_gid.values()):
        if _gid_:
            for __gid__ in _gid_:
                _gid_s_.append(_gs_)
        else:
            _gid_s_.append(0)
    self.gs[m].gid_s = _gid_s_
    # Make the output string to print on promnt
    optput_string_01 = f'Temporal slice number = {m}.'
    optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
    print(optput_string_01 + optput_string_02)

def find_grains_scikitimage_2d(self,
                               m,
                               connectivity=2,
                               ):
    """
    Detect grains using the library 'scikit-image'.

    Parameters
    ----------
    m : TYPE
        a single temporal slice number.
    connectivity : TYPE, optional
        recommended value = 2. The default is 2.
        Note:
            This method is for internal call. It is recommended that the user
            uses the method 'detect_grains(M)' instead.
        1-connectivity     2-connectivity     diagonal connection close-up
             [ ]           [ ]  [ ]  [ ]             [ ]
              |               \  |  /                 |  <- hop 2
        [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
              |               /  |  \             hop 1
             [ ]           [ ]  [ ]  [ ]
        REF: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

    Returns
    -------
    None.

    """
    _S_ = self.gs[m].s
    for i, _s_ in enumerate(np.unique(_S_)):
        # Mark the presence of this state
        self.gs[m].spart_flag[_s_] = True
        # -----------------------------------------------
        # Recognize the grains belonging to this state
        bin_img = (_S_ == _s_).astype(np.uint8)
        labels = skim_label(bin_img, connectivity=connectivity)
        if i == 0:
            self.gs[m].lgi = labels
        else:
            labels[labels > 0] += self.gs[m].lgi.max()
            self.gs[m].lgi = self.gs[m].lgi + labels
        self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
        self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
        # -----------------------------------------------
    # Get the total number of grains
    self.gs[m].n = np.unique(self.gs[m].lgi).size
    # Generate and store the gid-s mapping
    self.gs[m].gid = list(range(1, self.gs[m].n+1))
    _gid_s_ = []
    for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                           self.gs[m].s_gid.values()):
        if _gid_:
            for __gid__ in _gid_:
                _gid_s_.append(_gs_)
        else:
            _gid_s_.append(0)
    self.gs[m].gid_s = _gid_s_
    # Make the output string to print on promnt
    optput_string_01 = f'Temporal slice number = {m}.'
    optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
    print(optput_string_01 + optput_string_02)


"""
This module provides functions for identifying and labeling grains in a 2D
microstructure.

The module includes the following functions:
- `identify_grains_upxo_2d`: Deprecated function for identifying grains in a
2D microstructure.
- `mcgs2d`: Identifies and labels grains in a 2D microstructure.
- `quick_plot_s`: Plots the state matrix for a given temporal slice of a
microstructural system.

The module also includes a deprecated function `identify_grains_upxo_2d` and a
helper function `quick_plot_s` for visualizing the state matrix.

Note: The functions in this module require the installation and configuration
of necessary image processing libraries such as OpenCV or scikit-image.

Examples:
    Import and initialize the function:

        from upxo.pxtalops import detect_grains_from_mcstates as get_grains
        gs_dict = {...}  # Pre-loaded microstructure data
        msteps = 0  # Temporal slice index to analyze

    Call the function using scikit-image for grain detection:

        gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                              gs_dict = PXGS.gs,
                                              msteps = 10,
                                              isograin_pxl_neigh_order=2,
                                              store_state_ng=True
                                              )

        gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                              gs_dict = PXGS.gs,
                                              msteps = PXGS.m,
                                              isograin_pxl_neigh_order=2,
                                              store_state_ng=True
                                              )

"""
import numpy as np
from upxo._sup import dataTypeHandlers as dth
# from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as GS2d
import matplotlib.pyplot as plt


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


def mcgs2d(library=None,
           gs_dict=None,
           msteps=None,
           isograin_pxl_neigh_order=2,
           store_state_ng=True,
           ):
    """
    Identifies and labels grains in a 2D microstructure.

    This function labels grains in a given microstructure temporal slice using
    specified image processing libraries. It supports both OpenCV and
    scikit-image for grain detection and labeling, based on the user's choice.
    The function updates the input microstructure dictionary (`gs_dict`) with
    grain identification information and optionally stores the count of grains
    per state.

    Parameters
    ----------
    library : {'opencv', 'scikit-image'}, optional
        The library to use for grain identification. If not specified, the
        function raises a NotImplementedError for 'upxo'.
    gs_dict : dict
        A dictionary containing the microstructural system's data. It must
        include the state matrix for the specified temporal slice (`m`).
    msteps : list
        The indices of the temporal slices to analyze within `gs_dict`.
    isograin_pxl_neigh_order : {1, 2}, optional
        The pixel connectivity criterion for labeling grains. Use 1 for
        4-connectivity and 2 for 8-connectivity. Defaults to 2.
    store_state_ng : bool, optional
        If True, stores the number of grains for each distinct state in the
        microstructure. Defaults to True.

    Returns
    -------
    gs_dict : dict
        The updated microstructure dictionary with added grain identification
        information for the specified temporal slice.
    state_ng : numpy.ndarray
        An array containing the number of grains for each state, if
        `store_state_ng` is True.

    Raises
    ------
    NotImplementedError
        If the specified library is not supported or 'upxo' is provided as
        the library parameter.

    Notes
    -----
    - The function modifies the input `gs_dict` in place, adding grain
      identification information for the specified temporal slice.
    - Zero is reserved and not used as a valid grain label. Grains are labeled
      starting from 1.
    - The function assumes the presence of necessary libraries (`opencv` or
      `scikit-image`). Users must ensure these dependencies are installed.

    Examples
    --------
    Import

        from upxo.pxtalops import detect_grains_from_mcstates as get_grains

    Call the function using OpenCV for grain detection:

        gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                              gs_dict = PXGS.gs,
                                              msteps = 10,
                                              isograin_pxl_neigh_order=2,
                                              store_state_ng=True
                                              )

        gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                              gs_dict = PXGS.gs,
                                              msteps = PXGS.tslices,
                                              isograin_pxl_neigh_order=2,
                                              store_state_ng=True
                                              )
    """
    ocv_options = ('opencv', 'ocv', 'cv', 'cv2')
    ski_options = ('scikit-image', 'skimg', 'ski', 'si')
    # -----------------------------------------
    # Lets import some stuffs
    if library == 'upxo':
        print('Deprecated')
    elif library in ocv_options:
        import cv2
    elif library in ski_options:
        from skimage.measure import label as skim_label
    # -----------------------------------------
    # Handling datatypes
    if type(msteps) == int:
        msteps = [msteps]
    if type(msteps) not in dth.dt.ITERABLES:
        raise TypeError(f"msteps must be an iterable type. Received: {type(msteps)}")
    else:
        state_ng = {}
        for m in msteps:
            _S_ = gs_dict[m].s
            state_ng[m] = np.zeros(np.unique(_S_).size, dtype=int)
            for i, _s_ in enumerate(np.unique(_S_)):
                # Mark the presence of this state
                gs_dict[m].spart_flag[_s_] = True
                # -----------------------------------------
                # Identify the grains belonging to this state
                if library in ocv_options:
                    image = (_S_ == _s_).astype(np.uint8) * 255
                    state_ng[m][i], labels = cv2.connectedComponents(image)
                elif library in ski_options:
                    bin_img = (_S_ == _s_).astype(np.uint8)
                    labels, state_ng[m][i] = skim_label(bin_img,
                                                        return_num=True,
                                                        connectivity=isograin_pxl_neigh_order)
                # -----------------------------------------
                if i == 0:
                    gs_dict[m].lgi = labels
                else:
                    labels[labels > 0] += gs_dict[m].lgi.max()
                    gs_dict[m].lgi = gs_dict[m].lgi + labels
                # -----------------------------------------
                gs_dict[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
                gs_dict[m].s_n[_s_-1] = len(gs_dict[m].s_gid[_s_])
                print(f"MC state = {_s_}:  Num grains = {gs_dict[m].s_n[_s_-1]}")
            # Get the total number of grains
            gs_dict[m].n = np.unique(gs_dict[m].lgi).size
            # Generate and store the gid-s mapping
            gs_dict[m].gid = list(range(1, gs_dict[m].n+1))
            _gid_s_ = []
            for _gs_, _gid_ in zip(gs_dict[m].s_gid.keys(),
                                   gs_dict[m].s_gid.values()):
                if _gid_:
                    for __gid__ in _gid_:
                        _gid_s_.append(_gs_)
                else:
                    _gid_s_.append(0)
            gs_dict[m].gid_s = _gid_s_
            # Make the output string to print on promnt
            optput_string_01 = f'Temporal slice number = {m}.'
            optput_string_02 = f' |||| No. of grains detected = {gs_dict[m].n}'
            print(optput_string_01 + optput_string_02)
    return gs_dict, state_ng


def quick_plot_s(gs_dict, mcstep):
    """
    Plot the state matrix for a given temporal slice of a microstructural
    system.

    This function visualizes the state matrix (`s`) of a microstructural
    system at a specified temporal slice (`m`). It is designed for quick
    inspection of the state configuration in the system, typically used
    in the analysis or debugging of simulations involving grain structure
    evolution.

    Parameters:
    - gs_dict (dict): A dictionary containing the microstructural system's
                      data. Each key represents a temporal slice index, and
                      its value is an object containing the state matrix
                      (`s`) among other properties.
    - mcstep (int/list): The index of the temporal slice to be visualized.
                         This corresponds to a key in `gs_dict`. Type: list
                         is preferred. m of type int will ge put inside a list
                         and used as [mcstep].

    Returns:
    None. Directly displays the plot of the state matrix using matplotlib.

    Example usage:
    ```python

    from upxo.pxtalops import detect_grains_from_mcstates as gdetops

    gdetops.quick_plot_s(gs_dict, 10)
    ```
    This plots the state matrix for the 10th temporal slice of the
    microstructural system described in `gs_dict`.

    Note:
    This function requires matplotlib for plotting. Ensure matplotlib is
    installed and configured in your environment.
    """
    mcstep_max = max(list(gs_dict.keys()))
    if mcstep > mcstep_max:
        print(f'mcstep > max value. Plot is at max mcstep of {mcstep_max}')
        mcstep = mcstep_max
    plt.imshow(gs_dict[mcstep].s.astype(np.uint8))
    plt.colorbar()
    plt.title(f"MCGS2D tslice {mcstep}")
    plt.show()

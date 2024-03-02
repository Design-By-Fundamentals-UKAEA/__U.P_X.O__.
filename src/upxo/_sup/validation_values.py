"""UPXO: Validation & Conversion Utilities (NumPy Arrays)

This module provides functions and a class for validating and manipulating
NumPy arrays within the UPXO framework:

- `_validation` (internal): Class for validation targets and type checks.
- `ensure_ndarr_depth2`: Converts elements within iterables (up to depth 2) to
ndarrays.
- `chk_obj_type`: Checks if an object belongs to a specific type.
- `contains_nparray`: Checks for array containment within a target list.

These functionalities enhance data integrity and streamline NumPy array
handling in UPXO workflows.
"""


import numpy as np
from typing import Iterable


class _validation():
    """_validation (internal UPXO): Validation & Array Functions

    Provides internal functionalities for:

    - Storing pre-defined validation data (e.g., kernels).
    - Verifying object types using `chk_obj_type`.
    - Array validation and conversion:
        - `ensure_ndarr_depth2`: Converts elements up to depth 2 into ndarrays.
        - `contains_nparray`: Checks for array containment within a target
        list (ndarray).

    Intended for internal use within the UPXO framework only.

    Import
    ------
    from upxo._sup.validation_values import validation

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    # ------------------------------------------------
    # CLASS VARIABLE DEFINITION
    """ Exaplanation of gbjp_iden_kernels """
    gbjp_kernels2d: list = np.array([np.array([[1, 1, 1],
                                               [1, 0, 1],
                                               [1, 1, 1]]),
                                     np.array([[0, 1, 0],
                                               [1, 0, 1],
                                               [0, 1, 0]]),
                                     np.array([[1, 0, 1],
                                               [0, 0, 0],
                                               [1, 0, 1]])
                                     ])
    # ------------------------------------------------

    def __init__(self):
        pass

    def ensure_ndarr_depth2(self,
                            array: Iterable,
                            var_name: str = 'VARIABLE') -> np.ndarray | list[
                                np.ndarray]:
        """
        Updates all elements within an iterable (up to depth 2) to NumPy
        ndarrays.

        Parameters
        ----------
        array : Iterable
          The input iterable containing potentially nested iterables and
          elements to be converted.
        var_name : str, optional
          A variable name used for error messages. Defaults to 'VARIABLE'.

        Returns
        -------
        ndarray or list of ndarrays
          The converted iterable with all elements (up to depth 2) as NumPy
          ndarrays.

        Raises
        -------
        ValueError
          If the input `array` has a dimension (ndim) less than 2.
        TypeError
          If the input `array` is not an iterable at depth 1 or if the elements
          after conversion at depth 2 are not valid types.

        Notes
        -----
        This function recursively iterates through an iterable, converting
        elements that are not already NumPy ndarrays to ndarrays. It supports
        up to a maximum depth of 2 (i.e., nested iterables with a maximum
        depth of 2).
        """
        # ----------------------------------
        if array.ndim < 2:
            raise ValueError(f'np.array({var_name})',
                             '.ndim must be >= 2.')
        # ----------------------------------
        if not isinstance(var_name, str):
            var_name = 'VARIABLE'
        # ----------------------------------
        # VALIDATE array: Depth 1
        if not isinstance(array, Iterable):
            raise TypeError(f'{var_name} must be an Iterable.')
        else:
            if not isinstance(array, np.ndarray):
                array = np.array(array)
        # ----------------------------------
        # VALIDATE array: Depth 2
        if all([isinstance(_, Iterable) for _ in array]):
            if not all([isinstance(_, np.array) for _ in array]):
                array = np.array([np.array(_) for _ in array])
        else:
            raise TypeError('Invalid type(s) of input field array')
        # ----------------------------------
        return array

    def chk_obj_type(self, obj, expected_type):
        """
        Checks if an object's type matches the expected type.

        Args:
            obj: The object to check the type of.
            expected_type: The expected type of the object (as a string).

        Returns:
            True if the object's type matches the expected type, False
            otherwise.

        Example
        -------
        from upxo._sup.validation_values import _validation
        val = _validation()
        val.chk_obj_type(gs, expected_type)
        """
        return obj.__class__.__name__ == expected_type

    def contains_nparray(self,
                         ttype: str = 'gbjp_kernels2d',
                         target: Iterable = None,
                         sample: Iterable = None,
                         target_depth: int = 2,
                         ) -> bool:
        """
        Checks if a sample NumPy array is contained in a list of target NumPy
        arrays.

        This function validates if a given `sample` NumPy array exists within
        a collection of `target` NumPy arrays. The `ttype` argument specifies
        the source of the `target` arrays:

        - **'user'**: Requires you to provide a list or tuple of NumPy arrays
        in the `target` argument. This function will convert these
        user-provided arrays to NumPy ndarrays if necessary.
        - **'gbjp_kernels2d' (or any other valid CLASS_VARIBLE name)**: Uses
        a predefined set of NumPy arrays stored in the `self.gbjp_kernels2d`
        class variable for comparison.

        Parameters:
        ttype (str, optional): The type of target arrays to use for validation.
                               Defaults to 'gbjp_kernels2d'.
        target (Iterable, optional): The list or tuple of target NumPy arrays
                                     for validation (only used if `ttype`
                                                     is 'user').
        sample (Iterable, optional): The NumPy array to check for containment.

        Returns:
        bool: True if the `sample` array is found in one of the `target`
        arrays, False otherwise.

        Raises:
        TypeError: If `ttype` is not a string or if `target` and `sample`
        are not iterables.
        ValueError: If `ttype` is not a valid option.

        Notes:
        - This function supports nested iterables up to depth 2
        (i.e., lists containing lists) when using the `'user'` option for
        `ttype`.

        """
        # VALIDATE ttype
        if not isinstance(ttype, str):
            raise TypeError(f'Invalid ttype. Expected {str}')
        # ----------------------------------
        # Prepare the target
        _v_, containment = False, False
        if ttype == 'user':
            if target_depth == 2:
                target, _v_ = self.ensure_ndarr_depth2(target,
                                                       var_name='target'),
            True
        elif ttype == 'gbjp_kernels2d':
            target, _v_ = self.gbjp_kernels2d, True
        # ----------------------------------
        # Check for containment of sample in target
        if _v_:
            containment = any(np.array_equal(sample, valid_kernel)
                              for valid_kernel in target)
        # ----------------------------------
        return containment

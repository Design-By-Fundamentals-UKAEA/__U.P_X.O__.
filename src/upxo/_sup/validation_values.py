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
import os
from pathlib import Path
from typing import Iterable
from upxo._sup.dataTypeHandlers import dt


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
    # Do not use any speacial characters or spaces in any elements here
    # Any special characters get stripped of the string and would fail
    # validations !!
    fileContent_options_all = ('ctfheader', 'ebsdctf', 'ctffile', 'ctf',
                               'temperatures', 'states', 'grid2d', 'grid3d',
                               'upxoinstance', 'femesh', 'orientations')
    fileConOpt_ctf_headers = ('ctfheader', 'ebsdheader')
    fileConOpt_ctf_files = ('ebsdctf', 'ctf', 'ctffile')
    # ------------------------------------------------
    valid_extensions = ('.txt', '.dat', '.ctf', '.crc', '.h5df', '.dream3d')
    # ------------------------------------------------

    def __init__(self):
        pass

    def __repr__(self):
        return 'UPXO.Validations'

    def ensure_ndarr_depth2(self, array, var_name='VARIABLE'):
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

    def isiter(self, _iter):
        if not isinstance(_iter, Iterable):
            raise TypeError('INput not iterable.')

    def valstrs(self, strings):
        if not isinstance(strings, Iterable):
            strings = (strings,)
        for string in strings:
            if isinstance(string, str) or string.__class__.__name__ == 'WindowsPath':
                pass
            else:
                raise TypeError(f'Invalid type({string}). Expected: {str}',
                                f' Receieved: {type(string)}')

    def valnums(self, numbers):
        if not isinstance(numbers, Iterable):
            numbers = (numbers,)
        if not all([type(_) in dt.NUMBERS for _ in numbers]):
            raise TypeError(f'Invalid types in({numbers})'
                            f'Expected: type in {dt.Numbers}')

    def val_data_exist(self, *args, **kwargs):
        if args:
            for arg in args:
                if arg is None:
                    raise ValueError('One of inputs is empty')
        if kwargs:
            for kwarg_key, kwarg_val in kwargs.items():
                if not kwarg_val:
                    raise ValueError(f'{kwarg_key} value is empty.')

    def valnparr_types(self, arr1, arr2):
        '''
        from upxo._sup.validation_values import _validation
        val = _validation()
        val.valnparr_types(arr1, arr2)
        '''
        if not type(arr1) == type(arr2):
            raise TypeError('The two arguments are not of same type.'
                            'Expected: both must be numpy.ndarray')

    def valnparr_shape(self, arr1, arr2):
        '''
        val.valnparr_shape(arr1, arr2)
        '''
        # Validate existance
        self.val_data_exist(array1=arr1,
                            array2=arr2)
        # Validate numpy array type
        self.valnparr_types(arr1, arr2)
        if not arr1.shape == arr2.shape:
            raise ValueError('Entered np arrays must have same shape.')

    def valnparrs_types(self, *args):
        '''
        Validate numpy arrays for same type

        from upxo._sup.validation_values import _validation
        val = _validation()
        val.valnparrs_types(*args)
        '''
        # Validate existence
        self.val_data_exist(*args)
        # Validate types
        if len(args) == 1:
            if not isinstance(args[0], np.ndarray):
                raise TypeError('arg no.1 is not a numpy array')
        elif len(args) > 1:
            for i, arg in enumerate(args[1:], start=1):
                if not isinstance(arg, np.ndarray):
                    raise TypeError(f'arg no.{i} is not a numpy array')

    def valnparrs_shapes(self, *args):
        '''
        Validate numpy arrays for same shape

        from upxo._sup.validation_values import _validation
        val = _validation()
        a = np.random.random((3, 3))
        b = np.random.random((3, 3))
        c = np.random.random((3, 3))
        d = np.random.random((3, 4))
        val.valnparrs_shapes(a, b, c, d)
        '''
        # Validate types. This also valkidates existance by default
        self.valnparrs_types(*args)
        # Validate shapes
        if len(args) > 1:
            for i, arg in enumerate(args[1:], start=1):
                if not arg.shape == args[0].shape:
                    raise TypeError(f'Arg no.{i}.shape is not same'
                                    ' as Arg no.0.shape')

    def valnparrs_nelem(self, *args):
        '''
        Validate the total number of elemnets

        from upxo._sup.validation_values import _validation
        val = _validation()
        a = np.random.random((3, 3))
        b = np.random.random((9, 1))
        c = np.random.random((1, 9))
        val.valnparrs_nelem(a, b, c)
        '''
        # Validate if all are iterables
        for ia in args:
            self.isiter(ia)
        # Validate types
        self.valnparrs_types(*args)
        # Validate nuimber of elemenrs
        if len(set([arg.size for arg in args])) > 1:
            raise ValueError('The np arrays have unequal sizes')

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

    def val_path_exists(self, path, throw_path=True):
        if not path:
            raise ValueError('Path cannot be empty')
        self.valstrs(path)
        path = Path(path)
        if path.exists():
            if throw_path:
                return path
        else:
            raise FileNotFoundError(f"Path: {path} does not exist.")

    def val_filename_has_ext(self, file_name):
        self.valstrs(file_name)
        root, ext = os.path.splitext(file_name)
        if not ext:
            raise ValueError(f"{file_name} has no extention.")

    def val_file_exists(self, path, file_name_with_ext):
        path = self.val_path_exists(path, throw_path=True)
        self.valstrs(file_name_with_ext)
        if path.__class__.__name__ != 'WindowsPath':
            path = Path(path)
        if not Path(path/file_name_with_ext).exists():
            raise FileNotFoundError(f'File: {file_name_with_ext} does not'
                                    f'exist at Path: {path}')

    def val_filename_ext_permitted(self, ext):
        self.valstrs(ext)
        if ext not in self.valid_extensions:
            raise ValueError(f'{ext} is not a permitted extensions')

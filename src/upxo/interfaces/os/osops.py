import os
# import shutil
import pandas as pd
# import h5py
# import csv
# import openpyxl
from pathlib import Path
from upxo._sup.validation_values import _validation

def get_file_path(target_filename=None):
    """
    Search for a file with a specified name within a directory and all
    its subdirectories.

    Parameters:
    - root_dir_path (str): The path to the directory where the search should
    start. It should be an absolute path.
    - target_filename (str): The name of the file to search for, including the
    extension.

    Returns:
    - str: The full path to the found file if the file exists within
    the directory tree; otherwise, None.

    This function walks through all directories and subdirectories starting
    from root_dir_path, searching for target_file. If the file is found,
    the function immediately returns the full path to the file. If the search
    completes without finding the file, the function returns None.

    Example:
        get_file_path(search_for_file=True,
                      target_filename='_ctf_header_CuCrZr_1.txt')
    """
    search_for_file = True  # Retain for later development
    if search_for_file:
        for dirpath, dirnames, filenames in os.walk(get_path_UPXOsrcDIR()):
            if target_filename in filenames:
                return os.path.join(dirpath, target_filename)
    return None


def get_path_UPXOsrcDIR():
    return get_path_UPXODIR_L1(dirname='src')


def get_path_UPXOdataDIR():
    return get_path_UPXODIR_L1(dirname='data')


def get_path_UPXODIR_L1(dirname='src'):
    """
    Permitted options for dirname are currently:
        1. data
        2. demos
        3. dev_scripts
        4. docs
        5. external_contributions
        6. gallery
        7. logs
        8. profiling
        9. src
        10. tests
        11. tutorials
    """
    _permitted_ = ('data', 'demos', 'dev_scripts', 'docs',
                   'external_contributions', 'gallery', 'logs',
                   'profiling', 'src', 'tests', 'tutorials')
    # User input validation
    val = _validation()
    val.valstrs(dirname)
    if dirname not in _permitted_:
        raise ValueError(f'dirname: {dirname} not permitted.')
    # Get root and build path
    cwd = Path(os.getcwd())
    n = find_n_to_targetDIR(cwd, 'upxo_private')
    return cwd.parents[n]/dirname


def get_path_UPXOwriterdataDIR():

    src_path = get_path_UPXOsrcDIR()
    dp = src_path/'upxo'/'_writer_data'
    if dp.exists():
        return dp
    else:
        raise FileNotFoundError(f'Directory {dp} does not exist.')


def find_n_to_targetDIR(cwd: Path, target_dir_name: str) -> int:
    """
    Finds the depth (n) to reach a specified target directory from the current
    working directory (cwd).

    Parameters:
    - cwd (Path): The current working directory as a Path object.
    - target_dir_name (str): The name of the target directory to reach.

    # Example usage
    cwd = Path('C:/Development/M2MatMod/upxo_packaged/upxo_private/src/upxo/meshing')
    target_dir_name = 'src'
    n = find_n_to_targetDIR(cwd, target_dir_name)

    # Accessing the src directory if found
    if n != -1:
        src_dir = cwd.parents[n]
        print(f"The 'src' directory path: {src_dir}")
    """
    for n, parent in enumerate(cwd.parents):
        if parent.name == target_dir_name:
            return n
    raise ValueError(f'Search dir {target_dir_name} does not exist',
                     f'PWD is {cwd}',
                     'Please make sure UPXO is installed properly',
                     'If installed properly, just path it from repo')


def error_handler(func):
    """Decorator to handle exceptions and log file operations."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print(f"Operation '{func.__name__}' completed successfully.")
            return result
        except FileNotFoundError as e:
            print(f"Error: {e}. File or directory not found in '{func.__name__}'.")
        except Exception as e:
            print(f"An unexpected error occurred in '{func.__name__}': {e}")
    return wrapper


def data_importer(func):
    """Decorator to abstract the data importing functionality."""
    @error_handler  # Use the error handling decorator
    def wrapper(file_path, *args, **kwargs):
        print(f"Importing data from {file_path}")
        return func(file_path, *args, **kwargs)
    return wrapper


def data_exporter(func):
    """Decorator to abstract the data exporting functionality."""
    @error_handler  # Use the error handling decorator
    def wrapper(data, file_path, *args, **kwargs):
        print(f"Exporting data to {file_path}")
        func(data, file_path, *args, **kwargs)
        print("Data export completed successfully.")
    return wrapper


class FileManager:
    @error_handler
    def create_folder(self, path):
        """Create a new folder at the specified path."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder created at: {path}")
        else:
            print(f"Folder already exists at: {path}")

    @error_handler
    def delete_folder(self, path):
        """Delete the folder at the specified path."""
        if os.path.exists(path) and os.path.isdir(path):
            os.rmdir(path)
            print(f"Folder deleted at: {path}")
        else:
            print(f"Folder does not exist at: {path}")

    @error_handler
    def list_folder_contents(self, path):
        """List the contents of the folder at the specified path."""
        if os.path.exists(path) and os.path.isdir(path):
            return os.listdir(path)
        else:
            print(f"Path does not exist or is not a directory: {path}")

    @error_handler
    def write_to_file(self, file_path, data):
        """Write data to a file."""
        with open(file_path, 'w') as file:
            file.write(data)
            print(f"Data written to file: {file_path}")

    @error_handler
    def read_from_file(self, file_path):
        """Read data from a file."""
        with open(file_path, 'r') as file:
            return file.read()

    @data_importer
    def import_txt(self, file_path):
        """Import data from a TXT file."""
        with open(file_path, 'r') as file:
            data = file.read()
        return data

    @data_importer
    def import_csv(self, file_path):
        """Import data from a CSV file."""
        return pd.read_csv(file_path)

    @data_exporter
    def export_csv(self, data, file_path):
        """Export data to a CSV file."""
        data.to_csv(file_path, index=False)
"""
# Example Usage
file_manager = FileManager()
file_manager.create_folder('new_folder')
file_manager.delete_folder('new_folder')
contents = file_manager.list_folder_contents('.')
file_manager.write_to_file('example.txt', 'Hello, FileManager!')
data = file_manager.read_from_file('example.txt')
print(data)
"""

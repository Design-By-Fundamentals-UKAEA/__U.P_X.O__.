import os
# import shutil
import pandas as pd
# import h5py
# import csv
# import openpyxl

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

# Example Usage
file_manager = FileManager()
file_manager.create_folder('new_folder')
file_manager.delete_folder('new_folder')
contents = file_manager.list_folder_contents('.')
file_manager.write_to_file('example.txt', 'Hello, FileManager!')
data = file_manager.read_from_file('example.txt')
print(data)
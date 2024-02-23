import unittest
import os
from file_manager import FileManager  # Ensure you import FileManager class correctly

class TestFileManager(unittest.TestCase):
    def setUp(self):
        """Set up environment before each test."""
        self.file_manager = FileManager()
        self.test_dir = 'test_dir'
        self.test_file = 'test_file.txt'
        self.test_content = 'Hello, FileManager!'

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_create_and_delete_folder(self):
        """Test folder creation and deletion."""
        self.file_manager.create_folder(self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir), "Folder was not created.")

        self.file_manager.delete_folder(self.test_dir)
        self.assertFalse(os.path.exists(self.test_dir), "Folder was not deleted.")

    def test_list_folder_contents(self):
        """Test listing folder contents."""
        self.file_manager.create_folder(self.test_dir)
        content = self.file_manager.list_folder_contents(self.test_dir)
        self.assertIsInstance(content, list, "The returned content should be a list.")
        self.assertEqual(len(content), 0, "The folder should be empty.")

    def test_write_and_read_file(self):
        """Test writing to and reading from a file."""
        self.file_manager.write_to_file(self.test_file, self.test_content)
        self.assertTrue(os.path.exists(self.test_file), "File was not created.")

        content = self.file_manager.read_from_file(self.test_file)
        self.assertEqual(content, self.test_content, "File content does not match.")

if __name__ == '__main__':
    unittest.main()

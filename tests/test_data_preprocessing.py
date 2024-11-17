import sys
import os
import unittest
from unittest.mock import patch
import pandas as pd

# Add the 'src' directory to the Python path dynamically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now, you can import the load_data function from src.data_preprocessing
from data_preprocessing import load_data  # Adjusted import based on sys.path modification

class TestLoadData(unittest.TestCase):
    
    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_load_data_existing_file(self, mock_read_csv, mock_exists):
        # Simulate the case when the file exists
        mock_exists.return_value = True
        
        # Creating a mock DataFrame to return
        mock_dataframe = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_csv.return_value = mock_dataframe
        
        filepath = "../data/raw/fraud_data.csv"
        result = load_data(filepath)
        
        mock_read_csv.assert_called_once_with(filepath)  # Check if pd.read_csv was called with the right file
        self.assertIsInstance(result, pd.DataFrame)  # Check that the result is a pandas DataFrame
        self.assertEqual(result.shape, (2, 2))  # Check the shape (rows, columns)
    
    @patch("os.path.exists")
    def test_load_data_file_not_found(self, mock_exists):
        # Simulating the case when the file doesn't exist
        mock_exists.return_value = False
        
        filepath = "data/raw/non_existent_file.csv"
        
        # Checking that the appropriate exception is raised
        with self.assertRaises(FileNotFoundError):
            load_data(filepath)
    
    @patch("os.path.exists")
    def test_load_data_empty_filepath(self, mock_exists):
        # Simulating an empty filepath
        mock_exists.return_value = False
        filepath = ""
        
        # Checking for FileNotFoundError when file path is empty
        with self.assertRaises(FileNotFoundError):
            load_data(filepath)

if __name__ == '__main__':
    unittest.main()

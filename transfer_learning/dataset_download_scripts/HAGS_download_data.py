"""
HAGS Dataset Download Script

This script retrieves the data from the Texas Data Repository. It assumes you have
an account with an associated API_KEY. You can pass this key in like so:

$ TDR_API_KEY=<YOUR_KEY> python HAGS_download_data.py

You still need to adjust the download path to follow the README.md instructions.
"""


import os
from io import BytesIO
import pandas as pd
from pyDataverse.api import NativeApi, DataAccessApi
import pyDataverse.exceptions
import requests

# Base URL of the Dataverse repository
BASE_URL = 'https://dataverse.tdl.org/'

# API token for authentication
API_TOKEN = os.environ["TDR_API_KEY"]

# Dataset DOI
DOI = "doi:10.18738/T8/85R7KQ"

# Initialize the NativeApi for interacting with Dataverse metadata
api = NativeApi(BASE_URL, API_TOKEN)

# Initialize the DataAccessApi for accessing data files
data_api = DataAccessApi(BASE_URL, API_TOKEN)

def download_files_recursive(directory, files_list):
    """
    Recursively download files from the specified directory.

    Parameters:
        directory (str): The directory to download files from.
        files_list (list): List of files in the dataset.
    """
    for file in files_list:
        # print("new file:")
        # print(file)
        try:
            filename = file["directoryLabel"]
            if filename.startswith(directory):
                # Check if the file is in the specified directory
                filename = file["dataFile"]["filename"]
                absolute_path = os.path.join(file["directoryLabel"], filename)

                # Skip file if it already exists
                if os.path.exists(absolute_path):
                    print("File already exists, skipping:", absolute_path)
                    continue

                print("Downloading:", filename)
                file_id = file["dataFile"]["id"]
                try:
                    response = data_api.get_datafile(file_id, is_pid=False)
                except requests.exceptions.RequestException as error:
                    print("An error occurred while fetching data: Network error -", str(error))
                    # Handle Network error or take appropriate action
                    continue
                except pyDataverse.exceptions.ApiResponseError as error:
                    print("An error occurred: ApiResponseError -", str(error))
                    # Handle ApiResponseError or take appropriate action
                    continue
                except pyDataverse.exceptions.DataverseApiError as error:
                    print("An error occurred: DataverseApiError -", str(error))
                    # Handle DataverseApiError or take appropriate action
                    continue
                except pyDataverse.exceptions.DataverseError as error:
                    print("An error occurred: DataverseError -", str(error))
                    # Handle DataverseError or take appropriate action
                    continue
                except Exception as error:
                    print("An unexpected error occurred:", str(error))
                    # Handle any other unexpected errors
                    continue

                # Determine file extension
                file_extension = filename.split('.')[-1]

                # Construct the absolute path to save the file
                parent_dir = os.path.dirname(absolute_path)
                if not os.path.exists(parent_dir):
                    print("Creating parent directory...")
                    os.makedirs(parent_dir, exist_ok=True)

                if file_extension == 'tab':
                    # If file is a .tab file, convert it to CSV format
                    content = BytesIO(response.content)
                    df = pd.read_csv(content, sep='\t')
                    csv_filename = os.path.splitext(absolute_path)[0] + '.csv'
                    df.to_csv(csv_filename, index=False)
                    print("File converted to CSV format and saved successfully:", csv_filename)
                else:
                    # Save other file formats directly
                    with open(absolute_path, "wb") as f:
                        f.write(response.content)
                    print("File saved successfully:", absolute_path)
        except:
            print(f"File not compatible!")

# Define the directories to start downloading from
DIRECTORIES = ["Dataset/Sampled_Frames", "Dataset/Annotations", "Dataset/Cropped_Videos"]

# Iterate over the directories
for directory in DIRECTORIES:
    # Get list of files in the dataset
    dataset = api.get_dataset(DOI)
    files_list = dataset.json()['data']['latestVersion']['files']
    # Download files recursively from the starting directory
    download_files_recursive(directory, files_list)

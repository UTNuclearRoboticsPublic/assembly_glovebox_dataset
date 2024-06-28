"""
WorkingHands Dataset Download Script

You still need to adjust the download path to follow the README.md instructions.
"""
import requests, zipfile, io, os

URL = "http://vision.cs.stonybrook.edu/~supreeth/Working_Hands/WorkingHands.zip"
r = requests.get(URL)
z = zipfile.ZipFile(io.BytesIO(r.content))

HOME_DIR = os.environ["HOME"]

PATH = f"{HOME_DIR}/assembly_glovebox_dataset"

z.extractall(PATH)


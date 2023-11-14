# read in tmp_dataset.json
# each key is a image label, and the corresponding value is a list of image urls
#  download the images and save them to the tmp_dataset folder, and a subfolder for each label
#  also name the file with the label id as a prefix.

import json
import os
import requests
import shutil
from tqdm import tqdm

# read in the dataset
with open("tmp_dataset.json", "r") as f:
    dataset = json.load(f)

base_dir = "tmp_dataset"
# create a folder for each label
for label in dataset:
    pth = os.path.join(base_dir, label)
    os.makedirs(pth, exist_ok=True)

# download the images and save them to the correct folder
for label in tqdm(dataset):
    for url in dataset[label]:
        # get the image
        response = requests.get(url, stream=True)
        # get the file name
        file_name = url.split("/")[-1]
        # save the image
        with open(os.path.join(base_dir, f"{label}", f"{label}_{file_name}"), "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        # # wait a bit
        # time.sleep(1)

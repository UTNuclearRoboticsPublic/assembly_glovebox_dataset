import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import json
import fnmatch
import shutil

def parse_json(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    # declare tuple with image name and id
    matches = []
    for obj in data:
        image_name = obj["image"].split("-")[-1]
        image_id = int(obj['id'])
        matches.append((image_name, image_id))
    return matches

def convert(path_to_images, path_to_labels, path_to_json):
    labels = os.listdir(path_to_labels)
    images = os.listdir(path_to_images)
    image = Image.open(f"{path_to_images}/{images[0]}")
    height = np.asarray(image).shape[0]
    width = np.asarray(image).shape[1]


    last_task = labels[-1].split("task-")[1].split("-")[0]


    color_map = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255)
    }

    matches = parse_json(path_to_json)

    for match in matches:
        image_name = match[0]
        image_id = match[1]
        
        matching_masks = []
        for label in labels:
            label_id = int(label.split("-")[1])
            if label_id == image_id:
                matching_masks.append(label)
        
        final_arr = np.empty([height, width])
        prev_indexes = np.empty([height, width])
        prev_val = 0
        for index, label in enumerate(matching_masks):
            mask = Image.open(f"{path_to_labels}/{label}")
            mask_arr = np.array(np.asarray(mask))
            if "Left Hand" in label:
                mask_arr[mask_arr>0] = 1
                indexes_mask = np.where(mask_arr == 1)
                current_val = 1
            if "Right Hand" in label:
                mask_arr[mask_arr>0] = 2
                indexes_mask = np.where(mask_arr == 2)
                current_val = 2

            final_arr+=mask_arr

            # check if there was overlap in the labels
            # if so, find which class was overlapped with and fix it
            final_indexes = np.where(final_arr == 3)
            if np.array_equal(np.unique(final_indexes), np.unique(indexes_mask)):
                print("here was a match")
                final_arr[final_arr==3] = current_val
            elif np.array_equal(np.unique(final_indexes), np.unique(prev_indexes)):
                print("here2 was a match")
                final_arr[final_arr==3] = prev_val
            # else set it to the background class so it will be apparent when checking for errors
            else:
                print("Error! Correct class for overlapping label not found.")
                final_arr[final_arr==3] = 0
            
            prev_indexes = indexes_mask
            prev_val = current_val



        image = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, color in color_map.items():
            image[final_arr == idx] = color
        image = Image.fromarray(image)

        if np.any(final_arr > 0):
            os.makedirs(f"{path_to_images}/Labels", exist_ok=True)
            image.save(f"{path_to_images}/Labels/{image_name}")
            # image.save(f"./Labels/{image_name}.png")
        
    plt.imshow(final_arr)
    plt.show()

def get_json_directory_paths(project_number):
    pattern = f'project-{project_number}*.json'
    matching_files = []
    for file in os.listdir('./raw'):
        if fnmatch.fnmatch(file, pattern):
            matching_files.append(file)
    file = matching_files[0]
    name = file.split('.')[0]

    for root, dirs, files in os.walk('./raw'):
        for directory in dirs:
            ## check if the string directory matches the name

            if directory.split('-')[-1] == name.split('-')[-1]:
                matching_folder = f"./raw/{directory}"
    
    return f'./raw/{file}', matching_folder

def filter_labels_by_experiment(experiment_name, matching_folder):
    new_dir = f'{matching_folder}/{experiment_name}'
    os.makedirs(new_dir, exist_ok=True)
    for file in os.listdir(matching_folder):
        if file.startswith("J"):
            shutil.move(f'{matching_folder}/{file}', new_dir)
    return new_dir


if __name__ == "__main__":

    'Edit the following to the project numbers that match before converting--'
    proj_num_to_type = {
        ("ood", "Top_View"): 10,
        ("ood", "Bottom_View"): 11,
        ("id", "Top_View") : 12,
        ("id", "Bottom_View") : 13
    }
    
    # edit the images path
    # do all the conversions at once
    for dis in ["ood", "id"]:
        for task in ["J", "TB"]:
            for view in ["Side_View", "Top_View"]:
                key = (dis, view)
                proj_num = int(proj_num_to_type[key])
                json_file_path, matching_folder = get_json_directory_paths(project_number = proj_num)
                matching_labels = filter_labels_by_experiment(experiment_name = task, matching_folder = matching_folder)
                convert(f'./images/Test_Subject_1/{dis}/{task}/{view}', matching_folder, json_file_path)
    
    # convert('./images/Test_Subject_1/ood/J/Side_View', matching_folder, json_file_path)
    
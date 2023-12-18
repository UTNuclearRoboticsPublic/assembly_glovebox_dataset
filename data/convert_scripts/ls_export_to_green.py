import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import fnmatch

def parse_json(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    # declare tuple with image name and id
    matches = []
    all_annotators = []
    prev_width = None
    prev_height = None
    for obj in data:
        # save the annotator ID and what their unique numbers are
        image_name = obj["image"].split("-")[-1]
        image_id = int(obj['id'])
        annotator = int(obj['annotator'])
        try:
            orig_width = int(obj['tag'][0]['original_width'])
            orig_height = int(obj['tag'][0]['original_height'])

            prev_width, prev_height = orig_width, orig_height
        except:
            orig_width = prev_width
            orig_height = prev_height
            print(f"exception for {image_name} with {image_id} for annotator {annotator} for {path_to_json}")

        all_annotators.append(annotator)
        matches.append((image_name, image_id, annotator, orig_width, orig_height))
    unique_annotators = list(set(all_annotators)) # this is to only find unique annotators
    return matches, unique_annotators

def convert(path_to_labels, path_to_json, participant_number):
    labels = os.listdir(path_to_labels) # binary ground truth
    # images = os.listdir(path_to_images)  # raw images (not the ground truth)
    # image = Image.open(f"{path_to_labels}/{labels[1]}") # why the second index? probs cause used for height and width
    # height = np.asarray(image).shape[0]
    # width = np.asarray(image).shape[1]


    # last_task = labels[-1].split("task-")[1].split("-")[0]


    # color_map = {
    #     0: (0, 0, 0)
    # }

    matches, unique_annotators = parse_json(path_to_json) # json object return where each object has (image name, id)
    
    """"
    to add annotators:
    - in parse json, get the numbers of the annotators
    - add an external for loop for each annotator
    - when saving the multiclass label add the number of the annotator at the end (use either 1 or 2)
    - 
    """

    for match in matches:
        image_name = match[0]
        image_id = match[1]
        annotator = match[2]
        orig_width = match[3]
        orig_height = match[4]
        # add annotator number here

        task = image_name.split("_")[0] 
        
        matching_masks = []
        for label in labels:
            label_id = int(label.split("-")[1]) # this is the id number on the ground truth
            # also extract the annotator number ("by-x")
            label_annot_id = int(label.split("-")[-4])
            if label_id == image_id: # if the gt id matches the json id, add if it matches the annotator
                if label_annot_id == annotator:
                    matching_masks.append(label)
        

        # final_arr = np.empty([orig_height, orig_width])
        # prev_indexes = np.empty([orig_height, orig_width])
        # prev_val = 0
        for label in matching_masks:
            image = Image.open(f"{path_to_labels}/{label}")
        #     mask_arr = np.array(np.asarray(mask))
        #     if "screen" in label:
        #         mask_arr[mask_arr>0] = 0
        #         indexes_mask = np.where(mask_arr == 0)
        #         current_val = 0
        #     # if "Right Hand" in label:
        #     #     mask_arr[mask_arr>0] = 2
        #     #     indexes_mask = np.where(mask_arr == 2)
        #     #     current_val = 2

        #     final_arr+=mask_arr

        #     # check if there was overlap in the labels
        #     # if so, find which class was overlapped with and fix it
        #     # final_indexes = np.where(final_arr == 3)
        #     # if np.array_equal(np.unique(final_indexes), np.unique(indexes_mask)):
        #     #     print(f"here was a match {image_name}, {image_id}, {annotator}, {task}")
        #     #     final_arr[final_arr==3] = current_val
        #     # elif np.array_equal(np.unique(final_indexes), np.unique(prev_indexes)):
        #     #     print(f"here2 was a match {image_name}, {image_id}, {annotator}, {task}")
        #     #     final_arr[final_arr==3] = prev_val
        #     # # else set it to the background class so it will be apparent when checking for errors
        #     # else:
        #     #     # print("Correct class for overlapping label not found.")
        #     #     final_arr[final_arr==3] = 0
            
        #     prev_indexes = indexes_mask
        #     prev_val = current_val



        # image = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
        # for idx, color in color_map.items():
        #     image[final_arr == idx] = color
        # image = Image.fromarray(image)

        # if np.any(final_arr > 0):
            # os.makedirs(f"{path_to_images}/Labels", exist_ok=True)

            # save annotators as either 1 or 2
            # if annotator < max(unique_annotators):
            #     num_save = 1
            # else:
            #     num_save = 2

            # added annotator to top of save directory for each participant
        os.makedirs(f"./temp/green_screen/Test_Subject_{participant_number}/{task}/{view}", exist_ok=True)
        image.save(f"./temp/green_screen/Test_Subject_{participant_number}/{task}/{view}/{image_name}")
            # image.save(f"./Labels/{image_name}.png")
        
    # plt.imshow(final_arr)
    # plt.show()

def get_json_directory_paths(project_number):
    pattern = f'project-{project_number}*.json'
    matching_files = []
    for file in os.listdir('./raw'):
        if fnmatch.fnmatch(file, pattern):
            matching_files.append(file)
    file = matching_files[0]
    name = file.split('.')[0]

    matching_folder = ""

    for root, dirs, files in os.walk('./raw'):
        for directory in dirs:
            ## check if the string directory matches the name

            if directory.split('-')[-1] == name.split('-')[-1]:
                matching_folder = f"./raw/{directory}"
    
    return f'./raw/{file}', matching_folder

# def filter_labels_by_experiment(experiment_name, matching_folder):
#     new_dir = f'{matching_folder}/{experiment_name}'
#     os.makedirs(new_dir, exist_ok=True)
#     for file in os.listdir(matching_folder):
#         if file.startswith("J"):
#             shutil.move(f'{matching_folder}/{file}', new_dir)
#     return new_dir


if __name__ == "__main__":

    os.chdir('./data')

    'Edit the following to the project numbers that match before converting--'
    proj_num_to_type = {
        # ("ood", "Top_View"): 51,
        # ("ood", "Side_View"): 52,
        # ("id", "Top_View") : 49,
        # ("id", "Side_View") : 50
        "Top_View": 4,
        "Side_View": 3
    }
    
    participant_number = 6

    # edit the images path
    # do all the conversions at once
    for view in ["Side_View", "Top_View"]:
        key = view
        proj_num = int(proj_num_to_type[key])
        json_file_path, matching_folder = get_json_directory_paths(project_number = proj_num)
        # matching_labels = filter_labels_by_experiment(experiment_name = task, matching_folder = matching_folder)
        convert(matching_folder, json_file_path, participant_number)

    # convert('./images/Test_Subject_1/ood/J/Side_View', matching_folder, json_file_path)
    
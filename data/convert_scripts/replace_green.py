"""
This script contains two classes: GreenRemover and OrganizeScreens.

GreenRemover:
- This class is responsible for removing the green screen from images and replacing it with random images from specified classes.
- It takes the participant number and total number of images as input during initialization.
- The replace() method replaces the green screen in the images for the given participant number.
- The get_new_img() method processes the green screen image and replaces it with a random image.
- The get_padding() method calculates the padding required for the replacement image.
- The get_replacement_urls() method retrieves URLs of random images from specified classes.

OrganizeScreens:
- This class is responsible for organizing the screens by parsing JSON files and converting labels.
- It takes the participant number as input during initialization.
- The parse_json() method parses the JSON file and returns a list of matches and unique annotators.
- The convert() method converts the labels by matching them with the JSON data and saving the converted images.
- The get_json_directory_paths() method retrieves the paths of the JSON file and matching folder.

Note: The script also includes the main execution code that initializes the OrganizeScreens class and performs conversions, and then initializes the GreenRemover class to replace the green screen in the images.
"""
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fastbook import *
import math
import requests
import random
from io import BytesIO
import json
import fnmatch
import os
import shutil

class GreenRemover:
    def __init__(self, participant_number, total_images, classes):
        self.participant_number = participant_number
        self.total_images = total_images
        self.urls = self.get_replacement_urls(classes = classes, total_images=total_images)

    def replace(self, participant_number):
        """
        This function replaces the green screen in the images for the given participant number.
        """
        label_dir = f'./temp/green_screen/Test_Subject_{participant_number}'
        image_dir = f'./images/Test_Subject_{participant_number}/ood'
        
        tasks = ["Jenga_task", "Toolbox_task"]
        views = ["Side_View", "Top_View"]

        for task in tasks:
            for view in views: 
                labels = os.listdir(f'{label_dir}/{task}/{view}')
                images = os.listdir(f'{image_dir}/{task}/{view}')
                images = [image for image in images if not image.endswith('.csv') and '_G_' in image]


                for raw_label, raw_image in zip(labels, images):
                    

                    final_img = self.get_new_img(f'{label_dir}/{task}/{view}/{raw_label}', f'{image_dir}/{task}/{view}/{raw_image}')

                    ## TODO: save this image to the right directory and remove plt.imshow
                    # plt.imshow(final_img)
                    # plt.show()

                    os.makedirs(f"./images/Test_Subject_{participant_number}/replaced_green_screen/{task}/{view}", exist_ok=True)
                    save_img = Image.fromarray(final_img)
                    save_img.save(f"./images/Test_Subject_{participant_number}/replaced_green_screen/{task}/{view}/{raw_image}")
            


    def get_new_img(self, raw_label, raw_image):
        """
        This function processes the green screen image and replaces it with a random image.
        """
        img_array = np.array(Image.open(raw_image))
        label_array = np.asarray(cv2.imread(raw_label))

        indices = np.where(np.all(label_array!=0, axis = -1))
        coords = zip(indices[0], indices[1])



        left_pad, right_pad, upper_pad, lower_pad, new_w, new_h = self.get_padding(indices, label_array)

        left_pad = math.ceil(left_pad)
        right_pad = math.ceil(right_pad)
        lower_pad = math.ceil(lower_pad)
        upper_pad = math.ceil(upper_pad)

        random_url = random.choice(self.urls)
        # try:
        #     response = requests.get(random_url)
        #     random_img = Image.open(BytesIO(response.content))
            
        #     self.urls.remove(random_url)

        # except:
        #     self.urls.remove(random_url)

        #     random_url = random.choice(self.urls)
        #     response = requests.get(random_url)
        #     random_img = Image.open(BytesIO(response.content))
        #     self.urls.remove(random_url)

        random_img = None
        while random_img is None:
            random_url = random.choice(self.urls)
            self.urls.remove(random_url)

            try:
                response = requests.get(random_url)
                random_img = Image.open(BytesIO(response.content))
                break
            except:
                random_img = None

        if random_img is None:
            print("No valid image found.")



        # random class image 
        # first, get the random image
        random_img = random_img.resize((int(new_w), int(new_h)))
        rand_img_arr = np.array(random_img)

        new_img = random_img

        width, height = new_img.size

        new_width = width + right_pad + left_pad
        new_height = height + upper_pad + lower_pad

        result = Image.new(new_img.mode, (int(new_width), int(new_height)), (0, 0, 0))
        result = result.convert('RGB')

        result.paste(new_img, (int(left_pad), int(upper_pad)))

        print(f"the image is {raw_image} and the label is {raw_label}")

        return np.where((label_array != 0), np.array(result), img_array)


    def get_padding(self, indices, label_array):
        try:
            upper_quartile_y = np.percentile(indices[0], 98) # not quartile, percent
            lower_quartile_y = np.percentile(indices[0], 2)

            upper_quartile_x = np.percentile(indices[1], 98)
            lower_quartile_x = np.percentile(indices[1], 2)
        except:
            print("ERROR WITH THE QUARTILESSSSSS")
            upper_quartile_y = 55
            lower_quartile_y = 25

            upper_quartile_x = 55
            lower_quartile_x = 25


        width = np.shape(label_array)[1]
        height = np.shape(label_array)[0]

        new_h = upper_quartile_y - lower_quartile_y
        new_w = upper_quartile_x - lower_quartile_x

        left_pad = lower_quartile_x
        right_pad = width - upper_quartile_x

        # these are flipped because of how indexing works
        upper_pad = lower_quartile_y
        lower_pad = height - upper_quartile_y

        return left_pad, right_pad, upper_pad, lower_pad, new_w, new_h


    def get_replacement_urls(self, classes, total_images):

        urls = []
        for object in classes:
            urls += search_images_ddg(object, max_images= math.ceil(total_images / len(classes)))

        return urls

class OrganizeScreens:
    def __init__(self, participant_number):
        self.participant_number = participant_number

    def parse_json(self, path_to_json):
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

    def convert(self, path_to_labels, path_to_json, participant_number):
        labels = os.listdir(path_to_labels) # binary ground truth

        matches, unique_annotators = self.parse_json(path_to_json) # json object return where each object has (image name, id)
        
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
            
            # TODO: this just saves the last one, so make it add all the similar ones together and save that
            matching_labels = np.zeros((orig_height, orig_width))
            for label in matching_masks:
                image = Image.open(f"{path_to_labels}/{label}")
                matching_labels += np.asarray(image)

            matching_labels[matching_labels!=0] = 255

            image = Image.fromarray(matching_labels)
            image = image.convert('L')

            os.makedirs(f"./temp/green_screen/Test_Subject_{participant_number}/{task}/{view}", exist_ok=True)
            image.save(f"./temp/green_screen/Test_Subject_{participant_number}/{task}/{view}/{image_name}")

    def get_json_directory_paths(self, project_number):
        """
        This function returns the path to the json file and the matching folder
        """
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
                

if __name__ == '__main__':
    os.chdir('./data')

    'Edit the following to the project numbers that match before converting--'
    participant_number = 1
    proj_num_to_type = {
            "Top_View": 61,
            "Side_View": 60
        }
    # the organizer class is responsible for organizing the screens by parsing JSON files and converting labels
    organizer = OrganizeScreens(participant_number=participant_number)

    # edit the images path
    # do all the conversions at once
    for view in ["Side_View", "Top_View"]:
        key = view
        proj_num = int(proj_num_to_type[key])
        json_file_path, matching_folder = organizer.get_json_directory_paths(project_number = proj_num)
        # matching_labels = filter_labels_by_experiment(experiment_name = task, matching_folder = matching_folder)
        organizer.convert(matching_folder, json_file_path, participant_number)


    ## now overlaying images onto the green screen
    # the green remover class is responsible for removing the green screen from images and replacing it with random images from specified classes
    total_images = 20

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "horse", "ship", "carrots", "pencils"]
    green_remover = GreenRemover(participant_number=participant_number, total_images=total_images, classes = classes)
    
    green_remover.replace(participant_number=participant_number)

    dir_path = './temp'

    if os.path.exists(dir_path):
        # remove all contents of directory
        shutil.rmtree(dir_path)

"""
participant_number = 7
proj_num_to_type = {
        "Top_View": 6,
        "Side_View": 5
    }

    
participant_number = 4
proj_num_to_type = {
        "Top_View": 2,
        "Side_View": 1
    }


participant_number = 6
proj_num_to_type = {
        "Top_View": 4,
        "Side_View": 3
    }

participant_number = 9
proj_num_to_type = {
        "Top_View": 36,
        "Side_View": 35
    }


participant_number = 10
proj_num_to_type = {
        "Top_View": 38,
        "Side_View": 37
    }


participant_number = 12
proj_num_to_type = {
        "Top_View": 6,
        "Side_View": 5
    }

participant_number = 11
proj_num_to_type = {
        "Top_View": 4,
        "Side_View": 1
    }

    
participant_number = 1
proj_num_to_type = {
        "Top_View": 61,
        "Side_View": 60
    }

participant_number = 2
proj_num_to_type = {
        "Top_View": 59,
        "Side_View": 58
    }

participant_number = 3
proj_num_to_type = {
        "Top_View": 57,
        "Side_View": 56
    }

"""
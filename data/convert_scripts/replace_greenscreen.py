from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fastbook import *
import math
import requests
import random

def replace(participant_number, urls):
    label_dir = f'./temp/green_screen/{participant_number}'
    image_dir = f'./images/{participant_number}/ood'
    
    tasks = ["J", "TB"]
    views = ["Side_View", "Top_View"]

    for task in tasks:
        for view in views: 
            labels = os.listdir(f'{label_dir}/{task}/{view}')
            images = os.listdir(f'{image_dir}/{task}/{view}')

            for raw_label, raw_image in zip(labels, images):
                
                random_url = random.choice(urls)
                urls.remove(random_url)

                final_img = get_new_img(raw_label, raw_image, random_url)

                ## TODO: save this image to the right directory and remove plt.imshow
                plt.imshow(final_img)

                os.makedirs(f"./Labels/Test_Subject_{participant_number}/replaced_green_screen/{task}/{view}", exist_ok=True)
                final_img.save(f"./Labels/Test_Subject_{participant_number}/replaced_green_screen/{task}/{view}/{raw_image}")
        


def get_new_img(raw_label, raw_image, url):
    img_array = np.array(Image.open(raw_image))
    label_array = np.asarray(Image.open(raw_label))

    indices = np.where(np.all(label_array!=0, axis = -1))
    coords = zip(indices[0], indices[1])

    left_pad, right_pad, upper_pad, lower_pad, new_w, new_h = get_padding(indices, label_array)

    random_img = Image.open(requests.get(url, stream=True).raw)

    # random class image 
    # first, get the random image
    random_img = random_img.resize((int(new_w), int(new_h)))
    rand_img_arr = np.array(random_img)

    new_img = random_img

    width, height = new_img.size

    new_width = width + right_pad + left_pad
    new_height = height + upper_pad + lower_pad

    result = Image.new(new_img.mode, (int(new_width), int(new_height)), (0, 0, 0))

    result.paste(new_img, (int(left_pad), int(upper_pad)))

    return np.where((label_array != 0), np.array(result), img_array)


def get_padding(indices, label_array):
    upper_quartile_y = np.percentile(indices[0], 98) # not quartile, percent
    lower_quartile_y = np.percentile(indices[0], 2)

    upper_quartile_x = np.percentile(indices[1], 98)
    lower_quartile_x = np.percentile(indices[1], 2)

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


def get_replacement_urls(classes, total_images):

    urls = []
    for object in classes:
        urls += search_images_ddg(object, max_images= math.ceil(total_images / len(classes)))

    return urls
            

if __name__ == '__main__':
    total_images = 16 ## TODO: find how to calculate this number
    urls = get_replacement_urls(classes = ["animals", "sports balls", "boxes", "books", "pencils"])
    os.chdir('./data')
    replace(participant_number=6, urls = urls)

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

class GreenRemover:
    def __init__(self, participant_number, total_images):
        self.participant_number = participant_number
        self.total_images = total_images
        self.urls = self.get_replacement_urls(classes = ["animals", "sports balls", "boxes", "books", "pencils"], total_images=total_images)
        os.chdir('./data')

    def replace(self, participant_number):
        label_dir = f'./temp/green_screen/Test_Subject_{participant_number}'
        image_dir = f'./images/Test_Subject_{participant_number}/ood'
        
        tasks = ["J", "TB"]
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
                    save_img.save(f"./images/Test_Subject_{participant_number}/replaced_green_screen/{task}/{view}/{raw_image}.png")
            


    def get_new_img(self, raw_label, raw_image):
        img_array = np.array(Image.open(raw_image))
        label_array = np.asarray(cv2.imread(raw_label))

        indices = np.where(np.all(label_array!=0, axis = -1))
        coords = zip(indices[0], indices[1])

        left_pad, right_pad, upper_pad, lower_pad, new_w, new_h = self.get_padding(indices, label_array)

        random_url = random.choice(self.urls)
        try:
            response = requests.get(random_url)
            random_img = Image.open(BytesIO(response.content))
            
            self.urls.remove(random_url)

        except:
            self.urls.remove(random_url)

            random_url = random.choice(self.urls)
            response = requests.get(random_url)
            random_img = Image.open(BytesIO(response.content))
            self.urls.remove(random_url)



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


        return np.where((label_array != 0), np.array(result), img_array)


    def get_padding(self, indices, label_array):
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


    def get_replacement_urls(self, classes, total_images):

        urls = []
        for object in classes:
            urls += search_images_ddg(object, max_images= math.ceil(total_images / len(classes)))

        return urls
            

if __name__ == '__main__':
    total_images = 20 ## TODO: find how to calculate this number -> it's automatically calculated
    green_remover = GreenRemover(participant_number=6, total_images=total_images)
    
    green_remover.replace(participant_number=6)

import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

def convert(path_to_images, path_to_labels):
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

    for i in range(int(last_task)+1):
        # print(labels)
        filtered_labels = [f for f in labels if f.startswith(f"task-{i}")]
        final_arr = np.empty([height, width])
        for index, label in enumerate(filtered_labels):
            print(f"in the for loop")
            mask = Image.open(f"{path_to_labels}/{label}")
            mask_arr = np.array(np.asarray(mask))
            if "Banana" in label:
                mask_arr[mask_arr>0] = 1
            if "Orange" in label:
                mask_arr[mask_arr>0] = 2
            final_arr+=mask_arr
        # final_label = Image.fromarray(mask_arr)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, color in color_map.items():
            image[final_arr == idx] = color
        image = Image.fromarray(image)
        # image = image.convert('L')
        
        if np.any(final_arr > 0):
            image.save(f"masks/task-{i}.png")
    plt.imshow(final_arr)
    plt.show()

if __name__ == "__main__":
    convert('./images', './raw')
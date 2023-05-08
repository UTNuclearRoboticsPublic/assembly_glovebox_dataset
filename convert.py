import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def convert(path_to_images, path_to_labels):
    labels = os.listdir(path_to_labels)
    print(f"the labels are {labels}")
    for index, label in enumerate(labels):

        task_number = label.split("task-")[1].split("-")[0]
        mask = Image.open(f"{path_to_labels}/{label}")
        mask = mask.convert("RGB")

        mask_arr = np.array(mask)

        if "Banana" in label:
            mask_arr[:, :, 0] = 0
            mask_arr[:, :, 2] = 0

        if "Orange" in label:
            mask_arr[:, :, 0] = 0
            mask_arr[:, :, 1] = 0


        mask_arr[mask_arr!=255] = 0

        print(f"the index is {index} and {label[index+1]}")
        try:
            next_task = labels[index+1].split("task-")[1].split("-")[0]
        except:
            print("reached end")
            next_task = -1

        if task_number == next_task:
            new_label = labels[index+1]
            new_mask = Image.open(f"{path_to_labels}/{new_label}")
            new_mask = new_mask.convert("RGB")
            new_mask_arr = np.array(np.asarray(new_mask))

            print(f"the shape of new is {new_mask_arr.shape}")

            if "Banana" in new_label:
                new_mask_arr[:, :, 0] = 0
                new_mask_arr[:, :, 2] = 0
            elif "Orange" in new_label:
                new_mask_arr[:, :, 1] = 0
                new_mask_arr[:, :, 2] = 0
            
            mask_arr = new_mask_arr + mask_arr

            mask_arr[(mask_arr!=255)] = 0

            print(np.unique(mask_arr))
            new_label = Image.fromarray(mask_arr)

            plt.imshow(new_label)
            new_label.save(f"./masks/task-{task_number}.png")
            plt.show()

if __name__ == "__main__":
    convert('./images', './raw')
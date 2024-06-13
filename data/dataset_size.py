"""
    This script reports the number of instances in the HAGS dataset.

    Maintainer: slwanna@utexas.edu
"""
import os


if __name__ == "__main__":
    num_id_frames = 0
    num_ood_frames = 0

    ORIGINAL_FRAMES_PATH = "/home/slwanna/assembly_glovebox_dataset/Dataset/Sampled_Frames"
    original_frames_filenames_list = []
    for dirname, subdirs, files in os.walk(ORIGINAL_FRAMES_PATH):
        for fname in files:
            og_path = os.path.join(dirname, fname)
            if "/ood/" in og_path:
                num_ood_frames += 1
            elif "/id/" in og_path:
                num_id_frames +=1

    print(f"Total ID frames: {num_id_frames}")
    print(f"Total OOD frames - without augmentations: {num_ood_frames}")


    num_id_frames = 0
    num_ood_frames = 0
    ORIGINAL_LABELS_PATH = "/home/slwanna/assembly_glovebox_dataset/Dataset/Annotations"
    original_labels_filenames_list = []
    for dirname, subdirs, files in os.walk(ORIGINAL_LABELS_PATH):
        for fname in files:
            og_path = os.path.join(dirname, fname)
            if "/ood/" in og_path:
                num_ood_frames += 1
            elif "/id/" in og_path:
                num_id_frames +=1

    print(f"Total ID labels: {int(num_id_frames/2)}")
    print(f"Total OOD labels - without augmentations: {int(num_ood_frames/2)}")
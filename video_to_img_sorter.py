import os
import random
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from math import floor

# Option 1:
# randomly samples a set of 110 frames from a participant's videos
# TO DO: randomly take 10 to be used for the test set

# randomly samples 10 frames to use as out of distribution (green screen and participant without gloves)

def sorter(files):
    id = []
    ood = []
    for file in files:
        # print(f"{file}")
        parse = file.split("_")[1]
        # if the file has gloves
        if parse.startswith("GL"):
            # if it is not a green screen video
            x = file.split("_")[-1]
            y = x.startswith("GL")
            print(f"file is {file} and {x} is {y}")
            if (file.split("_")[-1]).startswith("GL"):
                id.append(file)
            else:
                ood.append(file)
                x = file.split("_")[-1]
        elif parse.startswith("NG"):
            ood.append(file)
        else:
            print(f"lonely {file}")
    return id, ood

def choose_frames(files, view, dist_type, subject_number, frames_to_sample):
    head_dir = f"/Test_Subject_{subject_number}/{view}"

    for file in files:
        name = file

        file = VideoFileClip(f"./Test_Subject_{subject_number}/{view}" + f"/{file}")

        starting_frame = 0

        num_frames = int(file.fps * file.duration)


        frame_numbers = np.linspace(0, num_frames, frames_to_sample).tolist()
        frame_numbers = [int(floor(frame_num)) for frame_num in frame_numbers]
        print(f"the frame numbers are {frame_numbers}")

        frame_numbers = np.array(frame_numbers, dtype=float)
        print(f"the frame numbers are {frame_numbers}")
        frame_times = frame_numbers * (1/file.fps)
        frame_times = np.floor(frame_times)

        frames = []
        for time in frame_times:
            print(f"the time is {time}")
            frames.append(file.get_frame((time)))

        for frame, frame_num in zip(frames, frame_numbers):
            name = name.split(".")[0]
            experiment_type = name.split("_")[0]

            save_path = f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/{name}_{frame_num}.png"
            
            if os.path.isfile(save_path):
                print("Same frame already exists!")         
                while True:
                    frame_num+=1
                    if os.path.isfile(save_path):
                        continue
                    else:
                        frame_time = frame_num * (1/file.fps) * (1/60)
                        frame = file.get_frame(float(frame_time))
                        break
            print(f"the frame is {frame_num}")
            image = Image.fromarray(frame)
            # change directories to include experiment
            # save the starting frame when changing later
            # adjust the number of frames you get per video by calculating how many videos you have and how 
            # many frames you need


            os.makedirs(f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/", exist_ok=True)
            image.save(f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/{name}_{frame_num}.png")
                
def sampler(subject_number):
    top_files = os.listdir(f"./Test_Subject_{subject_number}/Top_View")
    side_files = os.listdir(f"./Test_Subject_{subject_number}/Side_View")

    print(f"the top files are {top_files}")
    print(f"the side files are {side_files}")

    id_top_files, ood_top_files = sorter(top_files)

    id_side_files, ood_side_files = sorter(side_files)

    choose_frames(id_top_files, "Top_View", "id", subject_number, 55)
    choose_frames(id_side_files, "Side_View", "id", subject_number, 55)

    choose_frames(ood_top_files, "Top_View", "ood", subject_number, 5)
    choose_frames(ood_side_files, "Side_View", "ood", subject_number, 5)


if __name__=="__main__":
    # enter participant number to generate image directory with all values
    sampler(subject_number = 1) 
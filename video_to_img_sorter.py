import os
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from math import floor
from datetime import datetime

def sorter(files):
    """
        Sorts the files into either in-distribution or out-of distribution list
    """

    id = []
    ood = []
    for file in files:
        parse = file.split("_")[1]
        # if the file has gloves
        if parse.startswith("GL"):
            # if it is not a green screen video, it is ID (because green screen is mentioned in the last part of the file naming convention)
            if (file.split("_")[-1]).startswith("GL"):
                id.append(file)
            # if it is a greed screen video, it is OOD
            else:
                ood.append(file)
        # if there are no gloves in the video, it is OOD
        elif parse.startswith("NG"):
            ood.append(file)
        else:
            print(f"lonely {file}")
    return id, ood

def choose_frames(files, view, dist_type, subject_number, frames_to_sample, initial_frame):
    """
        Collects an equally distributed sample of frames from the videos sent in 
        and saves the unlabelled frames to its respective folder location.
    """

    head_dir = f"/Test_Subject_{subject_number}/{view}"

    def get_save_path(subject_number, dist_type, experiment_type, view, name, frame_num):
        "Returns the path where the label will be saved"
        save_path = f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/{name}_{frame_num}.png"
        return save_path

    for file in files:
        name = file

        file = VideoFileClip(f"./Test_Subject_{subject_number}/{view}" + f"/{file}")

        num_frames = int(file.fps * file.duration)


        # here we get an equal distribution of frames from the initial frame defined and the last frame of the video
        frame_numbers = np.linspace(initial_frame, num_frames, frames_to_sample).tolist()
        frame_numbers = [int(floor(frame_num)) for frame_num in frame_numbers]

        # moviepy only takes time in seconds to return a frame, so we calculate the time(s) of the video 
        # using the frame we want and the FPS of the video
        frame_numbers = np.array(frame_numbers, dtype=float)
        frame_times = frame_numbers * (1/file.fps)
        # we have to floor each time otherwise the last frame runs us into problems
        frame_times = np.floor(frame_times)

        frames = []
        for time in frame_times:
            frames.append(file.get_frame((time)))

        # going through each frame and saving it as an image
        final_frame_nums = []
        for frame, frame_num in zip(frames, frame_numbers):
            name = name.split(".")[0]
            experiment_type = name.split("_")[0]

            save_path = get_save_path(subject_number,dist_type, experiment_type, view, name, frame_num)
            
            # checking if a file with the same frame number already exists
            # if it does, check if the next frame in the video also already exists
            if os.path.isfile(save_path):
                while True:
                    frame_num+=1
                    save_path = get_save_path(subject_number,dist_type, experiment_type, view, name, frame_num)

                    if os.path.isfile(save_path):
                        continue
                    else:
                        frame_time = floor(frame_num * (1/file.fps))
                        frame = file.get_frame(float(frame_time))
                        break
            final_frame_nums.append(frame_num)
            image = Image.fromarray(frame)

            os.makedirs(f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/", exist_ok=True)
            image.save(get_save_path(subject_number,dist_type, experiment_type, view, name, frame_num))
        
        text_file_path = f"./images/Test_Subject_{subject_number}/{dist_type}/{experiment_type}/{view}/labelhistory.txt"

        # write to a text file about the images obtained from the videos in this code run if we want more for further use
        with open(text_file_path, "w") as file:
            file.write('\n')
            file.write(f"Time of file writes: {datetime.now()}")
            file.write('\n')
            file.write(f"Initial frame: {initial_frame}")
            file.write('\n')
            file.write(f"Frame nums saved: {final_frame_nums}")


        
                
def sampler(subject_number, initial_frame):
    """
        Takes files from a given test subject number and gathers a list
        of equally distributed frames to save for future annotations. 

        Initial frame is given to offset when gathering new sets of images to be used for annotations.
    """

    top_files = os.listdir(f"./Test_Subject_{subject_number}/Top_View")
    side_files = os.listdir(f"./Test_Subject_{subject_number}/Side_View")

    id_top_files, ood_top_files = sorter(top_files)

    id_side_files, ood_side_files = sorter(side_files)

    # Save the frames into the respective folders
    choose_frames(id_top_files, view="Top_View", dist_type="id", subject_number=subject_number, frames_to_sample=55, initial_frame=initial_frame)
    choose_frames(id_side_files, "Side_View", "id", subject_number, 55, initial_frame)

    choose_frames(ood_top_files, "Top_View", "ood", subject_number, 5, initial_frame)
    choose_frames(ood_side_files, "Side_View", "ood", subject_number, 5, initial_frame)


if __name__=="__main__":
    # enter participant number to generate image directory with all values
    sampler(subject_number = 1, initial_frame = 0) 
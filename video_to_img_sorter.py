import os
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from math import floor, ceil
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

    for i, file in enumerate(files):

        name = file

        file = VideoFileClip(f"./Test_Subject_{subject_number}/{view}" + f"/{file}")

        num_frames = int(file.fps * file.duration)

        num_samples = (frames_to_sample/len(files))

        # if the num_samples calculated is not an integer, then
        # we have to split it as evenly as we can. This means alternating 
        # within 1 of the float value given by num_samples for each video in
        # this series of files to ensure as little bias as possible
        if not num_samples == int(num_samples):
            notInt = True

        # define how many images to sample in this video
        if notInt:
            if i%2==0:
                num_samples = floor(frames_to_sample/len(files))
            else:
                num_samples = ceil(frames_to_sample/len(files))
        else:
            num_samples = (frames_to_sample/len(files))
        print(f"the sampling interval is {num_samples}")

        # here we get an equal distribution of frames from the initial frame defined and the last frame of the video
        # using num_samples as a bound to ensure that the sampled
        # frames come from the meat of the video, not on the beginning and last frames
        edge_cutter = (1/(num_samples*2)) * num_frames
        frame_numbers = np.linspace(initial_frame+edge_cutter, num_frames-edge_cutter, num_samples).tolist()
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

        # write to a text file about the images obtained from the videos in this code run if we want more images for future use
        with open(text_file_path, "a") as file:
            file.write('\n')
            file.write(f"file name: {name}")
            file.write('\n')
            file.write(f"Time of file writes: {datetime.now()}")
            file.write('\n')
            file.write(f"Initial frame: {initial_frame}")
            file.write('\n')
            file.write(f"Frame nums saved: {final_frame_nums}")


        
                
def sampler(participants, initial_frame):
    """
        Takes files from a given test subject number and gathers a list
        of equally distributed frames to save for future annotations. 

        Initial frame is given to offset when gathering new sets of images to be used for annotations.
    """
    
    # Getting 1100 total in-distrubution frames and 100 total out of distribution frames
    num_participant = 10
    id_per_part = 1100/num_participant
    ood_per_part = 100/num_participant

    for i, participant in enumerate(participants):

        top_files = os.listdir(f"./Test_Subject_{participant}/Top_View")
        side_files = os.listdir(f"./Test_Subject_{participant}/Side_View")

        id_top_files, ood_top_files = sorter(top_files)

        id_side_files, ood_side_files = sorter(side_files)
   
        top_ood = (ood_per_part/2)
        side_ood = (ood_per_part/2)

        top_id = (id_per_part/2)
        side_id = (id_per_part/2)
        
        print(f"{top_id}, {side_id}, {top_ood}, {side_ood}")    
        # Save the frames into the respective folders
        choose_frames(id_top_files, view="Top_View", dist_type="id", subject_number=participant, frames_to_sample=top_id, initial_frame=initial_frame)
        choose_frames(id_side_files, "Side_View", "id", participant, side_id, initial_frame)

        choose_frames(ood_top_files, "Top_View", "ood", participant, top_ood, initial_frame)
        choose_frames(ood_side_files, "Side_View", "ood", participant, side_ood, initial_frame)


if __name__=="__main__":
    # enter participant number to generate image directory with all values
    sampler(participants = [1], initial_frame = 0) 
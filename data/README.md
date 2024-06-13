Each subject's respective images, labels, and videos can be placed here to follow the directory structure needed when using code in the folder. 

From the folder of all the data, place the images and Labels folder into the directory. 

When converting videos to frames, place them into this folder as well into the 'videos' folder. Then, run `video_to_img_sorter.py`. (in the convert_scripts directory)

For converting labels from Label Studio to multiclass labels, place the respective PNG exports and JSON file into the 'raw' folder Then, run `ls_export_to_multiclass.py` (in the convert_scripts directory)

These are the following directories that should be in this folder after dropping them in.
- videos
- images
- Labels
- raw
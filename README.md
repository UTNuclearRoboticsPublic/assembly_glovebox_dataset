# assembly_glovebox_dataset

`pip install -e .` To get modules setup

FOR MOBILE SAM-

Clone into root directory
Then, copy over MedSAM's Mask Decoder from their site into mask_decoder.py
Then, change into the MobileSAM directory and do `pip install -e .`

sudo kill -9 (PID HERE AFTER LOOKING AT nvidia-smi)

Viewing logs-
tensorboard --logdir lightning_logs




## Environment

Setup your environment using the files under `/config`

## Testing

For all tests, relevant versions must be stored under `/lightning_logs`. 

`tester.sh` runs tests on the model described in the checkpoint path. Uncomment only of the models and the 4 associated YAMLs at a time to run experiments for all dataset configurations. For custom dataset configurations, comment out relevant datasets under `DATASET CONFIGURATIONS`.


To run tests on model ensembles, run `python -m training.lightning_trainers.ensemble_model` after editing the model type [here](https://github.com/UTNuclearRobotics/assembly_glovebox_dataset/blob/45c52dcb4d2b49c24846c390b5d12e09007390e3/training/lightning_trainers/ensemble_model.py#L100). Checkpoints used are described in ensemble models' main method in `/training/lightning_trainers/ensemble_model.py`.

Test time for inference using `/training/time_testing.ipynb`

## Training

Training is based off of model configs. All model configs for each model can be used for training. There are separate model configs for each model only for the respective test sets, but training should be the same. 

To train, run one of the commands in `command.sh`. To run even after exiting terminal, run the bash file as `nohup trainer.sh &` instead, or run each individual command in the terminal using `nohup` in a similar manner.

These commands can also be customized, for example, `python -m training.cli fit --config ./training/config.yml --data.batch_size=5 --trainer.fast_dev_run=True --ckpt_path=./lightning_logs/version_8/checkpoints/epoch=49-step=350.ckpt`

This command uses cli.py, the relevant config files under `/training/training_configs` , and the models under `/training/lighting_trainers` and `/training/models`


## Data setup

## Data processing/analysis

To convert exported Label Studio projects and files to PNGs, run `ls_export_to_multiclass.py.` If so, the associated exported PNGs and JSON must be stored in the `/data/raw` folder, and must be matched to the participant and type of project by editing [these lines](https://github.com/UTNuclearRobotics/assembly_glovebox_dataset/blob/45c52dcb4d2b49c24846c390b5d12e09007390e3/data/convert_scripts/ls_export_to_multiclass.py#L176-L183).

To obtain extracted frames from raw videos, run `video_to_img_sorter.py`. The videos should be stored in `/data/videos` in the  `/videos/Test_Subject_{subjectnumber}/{View}` format. For example, videos stored under `/videos/Test_Subject_1/Top_View` and `/videos/Test_Subject_1/Top_View` are accepted. Video names should following the `{activity type}_{gloves or no gloves}_{green screen if relevant}.mp4`. For example, `J_GL_G.mp4` and `J_NG.mp4` are acceptable. 

To produce the dataset of replacing the green screen with images using the Label Studio raw project files, run `replace_green.py`. Note: you must edit [these lines](https://github.com/UTNuclearRobotics/assembly_glovebox_dataset/blob/45c52dcb4d2b49c24846c390b5d12e09007390e3/data/convert_scripts/replace_green.py#L293-L297) to match the Label Studio files (JSON and PNG export) in `/data/raw` to the participants and view type.

Return Cohen's Kappa on the dataset by running `/data/covert_scripts/cohens_kappa.ipynb`.

Obtain tensorboard data in charts and tables using `tb_to_csv.ipynb`. Make sure the paths to the versions you want to use under `/lightning_logs` are updated correctly. The ones used for test results by the authors are updated. 

After running the above, all images should be saved in `/data/images` and all labels in `/data/Labels`. These can also be done by just copying in the files from the uploaded dataset, skipping the video and Label Studio raw steps.
- The images directory should have an id, ood, and replaced_green_screen folder for each participant
- each participant in the Labels folder should have a By_1 and By_2 folder that each have a id and ood folder, respectively

## Other

Dataloaders and lightning module is in `/training/dataloaders/`. Random transformations are in `dataloader.py`.

Lightning trainers run are in `/training/lightning_trainers`. These can all be run through the cli as described above with `tester.sh`, but these files are here for adjusting metrics. Many of the methods for the models are inhereted from lightning_model.py. The UNet model is trained directly using `lightning_model.py`, and the other models such as MobileSAM and BiSeNetv2 inherit from this, and also change some of these methods in their respective child classes. 

`/training/metrics.py` includes the predictive entropy metric and draft of adaptive_calibration_error. 

`training/ignore_this_folder` is used for brainstorming and quick analysis. Nothing useful for final results here.







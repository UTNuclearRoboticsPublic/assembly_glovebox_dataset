#!/bin/bash

# train a UNET model
python -m training.cli fit --config ./training/training_configs/unet_id.yml


# train a BiSeNetV2 model
python -m training.cli fit --config ./training/training_configs/bisenet_v2_id.yml


# train a MobileSAM model
python -m training.cli fit --config ./training/training_configs/mobilesam_id.yml
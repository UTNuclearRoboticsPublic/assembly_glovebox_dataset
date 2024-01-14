# uncomment the code under one of the models at a time to test all dataset configurations


# bisenet
# CKPT_PATH="./lightning_logs/version_79/checkpoints/epoch=3373-step=26992.ckpt"
# ID_YML="./training/training_configs/bisenet_v2_id.yml"
# OOD_YML="./training/training_configs/bisenet_v2_ood.yml"
# GS_YML="./training/training_configs/bisenet_v2_gs.yml"
# GS_OOD_YML="./training/training_configs/bisenet_v2_gs+ood.yml"

# unet
# CKPT_PATH="./lightning_logs/version_76/checkpoints/epoch=1856-step=29712.ckpt"
# ID_YML="./training/training_configs/unet_id.yml"
# OOD_YML="./training/training_configs/unet_ood.yml"
# GS_YML="./training/training_configs/unet_gs.yml"
# GS_OOD_YML="./training/training_configs/unet_gs+ood.yml"

# mobileSAM
CKPT_PATH="./lightning_logs/version_75/checkpoints/epoch=1849-step=29600.ckpt"
ID_YML="./training/training_configs/mobilesam_id.yml"
OOD_YML="./training/training_configs/mobilesam_ood.yml"
GS_YML="./training/training_configs/mobilesam_gs.yml"
GS_OOD_YML="./training/training_configs/mobilesam_gs+ood.yml"



# DATASET CONFIGURATIONS. 
## COMMENT/UNCOMMENT RELEVANT ONES

# # # ID, without dropout
python -m training.cli test \
--config $ID_YML \
--ckpt_path=$CKPT_PATH 

# # # # ID, with dropout
python -m training.cli test --config $ID_YML \
--ckpt_path=$CKPT_PATH \
--model.init_args.test_dropout=True

# # OOD, without dropout
python -m training.cli test --config $OOD_YML \
--ckpt_path=$CKPT_PATH

# # # # OOD, with dropout
python -m training.cli test --config $OOD_YML \
--ckpt_path=$CKPT_PATH  \
--model.init_args.test_dropout=True

# # green screen, without dropout
python -m training.cli test --config $GS_YML \
--ckpt_path=$CKPT_PATH

# # green screen, with dropout
python -m training.cli test --config $GS_YML \
--ckpt_path=$CKPT_PATH  \
--model.init_args.test_dropout=True

# # OOD and Green Screen, without dropout
python -m training.cli test --config $GS_OOD_YML \
--ckpt_path=$CKPT_PATH 

# # # OOD and Green Screen, with dropout
python -m training.cli test --config $GS_OOD_YML \
--ckpt_path=$CKPT_PATH  \
--model.init_args.test_dropout=True

# ensemble training -> send in the models from above 
# python -m training.lightning_trainers.ensemble_model
## testing commands
# python -m training.cli test --config ./training/training_configs/unet.yml --trainer.fast_dev_run=True


# ckpt path to use
# assumes all are set to ood

# using bisenet
# CKPT_PATH_1="./lightning_logs/version_79/checkpoints/epoch=3373-step=26992.ckpt"
# CKPT_PATH_2="./lightning_logs/version_77/checkpoints/epoch=1083-step=8672.ckpt"
# CKPT_PATH_3="./lightning_logs/version_72/checkpoints/epoch=1440-step=11528.ckpt"

# using unet
# CKPT_PATH_1="./lightning_logs/version_78/checkpoints/epoch=3223-step=51584.ckpt"
# CKPT_PATH_2="./lightning_logs/version_76/checkpoints/epoch=1856-step=29712.ckpt"
# CKPT_PATH_3="./lightning_logs/version_73/checkpoints/epoch=1011-step=8096.ckpt"

# using mobilesam
# CKPT_PATH_1="./lightning_logs/version_80/checkpoints/epoch=3027-step=24224.ckpt"
# CKPT_PATH_2="./lightning_logs/version_75/checkpoints/epoch=1849-step=29600.ckpt"
# CKPT_PATH_3="./lightning_logs/version_69/checkpoints/epoch=1312-step=10704.ckpt"

# 

# ID_YML="./training/training_configs/unet_id.yml"
# ID_YML="./training/training_configs/mobilesam_id.yml"

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
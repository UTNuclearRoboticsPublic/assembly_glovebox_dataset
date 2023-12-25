## testing commands
# python -m training.cli test --config ./training/training_configs/unet.yml --trainer.fast_dev_run=True


# ckpt path to use
# assumes all are set to ood

# using bisenet
CKPT_PATH="./lightning_logs/version_79/checkpoints/epoch=3373-step=26992.ckpt"

ID_YML="./training/training_configs/bisenet_v2_id.yml"
OOD_YML="./training/training_configs/bisenet_v2_ood.yml"
GS_YML="./training/training_configs/bisenet_v2_gs.yml"


# # OOD, without dropout
# python -m training.cli test --config $OOD_YML \
# --ckpt_path=$CKPT_PATH

# OOD, with dropout
python -m training.cli test --config $OOD_YML \
--ckpt_path=$CKPT_PATH  \
--model.init_args.test_dropout=True \
--trainer.fast_dev_run=True

# # ID, without dropout
# python -m training.cli test \
# --config $ID_YML \
# --ckpt_path=$CKPT_PATH


# # ID, with dropout
# python -m training.cli test --config $ID_YML \
# --ckpt_path=$CKPT_PATH \
# --model.init_args.test_dropout=True \

# # OOD and Green Screen, without dropout
# python -m training.cli test --config $YML \
# --ckpt_path=CKPT_PATH 

# # OOD and Green Screen, with dropout
# python -m training.cli test --config $YML \
# --ckpt_path= CKPT_PATH  \
# --model.init_args.test_dropout=True 

# ensemble training -> send in the models from above 
# python -m training.lightning_trainers.ensemble_model
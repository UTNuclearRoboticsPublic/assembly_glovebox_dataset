- edit model params from config.yml in training folder

EXAMPLE -> RUN THIS FROM assembly_glovebox_dataset directory.-
`python -m training.cli fit --config ./training/config.yml --data.batch_size=5 --trainer.fast_dev_run=True --ckpt_path=./lightning_logs/version_8/checkpoints/epoch=49-step=350.ckpt`

If you want to change the model you are using, do it directly from the cli.py
Edit model params from configuration files in `training_configs`

EXAMPLE -> RUN THIS FROM assembly_glovebox_dataset directory.-
`python -m training.cli fit --config ./training/config.yml --data.batch_size=5 --trainer.fast_dev_run=True --ckpt_path=./lightning_logs/version_8/checkpoints/epoch=49-step=350.ckpt`

If you want to change to a new model architecture not used in the paper, make sure to import it into the `cli.py` and then use one of the other config files as a template. 
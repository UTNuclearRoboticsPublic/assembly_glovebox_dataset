from setuptools import setup, find_packages

setup(
    name="assembly_glovebox_dataset_loca",
    license="BSD 3.0",
    version = "0.0.2",
    # packages = ["dataloaders", "models", "lightning_trainers", "tl_dataloaders", "tl_models", "tl_lightning_trainers"],
    packages = find_packages("assembly_glovebox_dataset", exclude=["config", "Dataset", "Other_Datasets",]),

    package_dir = {
        "dataloaders": "./training/dataloaders",
        "models": "./training/models",
        "lightning_trainers": "./training/lightning_trainers",
        "tl_dataloaders": "./transfer_learning/dataloader",
        "tl_models": "./transfer_learning/models",
        "tl_lightning_trainers": "./transfer_learning/lightning_trainers",
    },
    install_requires=[
        "albumentations",
        "glob2",
        "jsonargparse[signatures]>=4.27.7",
        "lightning",
        "matplotlib",
        "mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git",
        "numpy",
        "netcal",
        "opencv-python",
        "pandas",
        "pillow",
        "timm",
        "torch",
        "torchmetrics",
        "torchaudio",
        "torchvision",
        "transformers",
    ],
)
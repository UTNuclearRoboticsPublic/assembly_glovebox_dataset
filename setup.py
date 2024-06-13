from setuptools import setup, find_packages

setup(
    name="assembly_glovebox_dataset_loca",
    license="BSD 3.0",
    version = "0.0.2",
    packages = ["dataloaders", "models", "lightning_trainers"],
    package_dir = {
        "dataloaders": "./training/dataloaders",
        "models": "./training/models",
        "lightning_trainers": "./training/lightning_trainers"
    },
    install_requires=[
        "albumentations",
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
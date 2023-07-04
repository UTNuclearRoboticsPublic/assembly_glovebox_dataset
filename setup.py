from setuptools import setup, find_packages

setup(
    name="assembly_glovebox_dataset_loca",
    version = "0.0.2",
    packages = ["dataloaders", "models", "lightning_trainers"],
    package_dir = {
        "dataloaders": "./training/dataloaders",
        "models": "./training/models",
        "lightning_trainers": "./training/lightning_trainers"
    }
)
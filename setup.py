from setuptools import setup, find_packages

setup(
    name="assembly_glovebox_dataset_loca",
    version = "0.0.1",
    packages = ["lcl_loaders", "lcl_models", "lcl_trainers"],
    package_dir = {
        "lcl_loaders": "./training/dataloaders",
        "lcl_models": "./training/models",
        "lcl_trainers": "./training/training_modules"
    }
)
from setuptools import setup, find_packages

setup(
    name="chestxray",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "torch>=2.0.0",
        "torchvision",
        "pytorch-lightning",
        "tensorboard",
        # "tensorflow-gpu>=2.10.0"  # Added GPU version
    ]
)
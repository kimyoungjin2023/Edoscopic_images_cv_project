from setuptools import setup, find_packages

setup(
    name="multitask-vision",
    version="0.1.0",
    description="Object Detection and Segmentation Multi-task Learning",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourname/multitask-vision",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.8",

    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.21.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

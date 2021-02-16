"""
This module sets up the package and is run everytime the package is installed.
"""
import setuptools

install_requires = [
    'ctgan',
    'torch<2,>=1.0',
    'torchvision<1,>=0.4.2',
    'scikit-learn<0.24,>=0.21',
    'numpy<2,>=1.17.4',
    'pandas<1.1.5,>=0.24',
    'rdt>=0.2.7,<0.4',
    'packaging',
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "SuperconGAN", # Replace with your own username
    version = "0.0.9",
    author = "Rajeev Atla",
    author_email = "rajeev@rajeevatla.com",
    description = "GAN trained on superconductivity data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/RajeevAtla/SuperconGAN",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = '>=3.6',
    install_requires = install_requires
)

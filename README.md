# SuperconGAN

[![Downloads](https://static.pepy.tech/personalized-badge/supercongan?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/supercongan)

A program to train a GAN using superconductivity data.
It was inspired by and is based off of the CTGAN library for generating GANs for tabular datasets.

## Installation
To install the latest version, please use the following command in a terminal window:

```PowerShell
python3 -m pip install SuperconGAN --upgrade
```

## Starter Example

To get a feel for the package, try running the following code, after installing the package (above):

```python
import SuperconGAN

model = SuperconGAN.Synthesizer()
model.fit(epochs = 5)
model.sample(n = 10)
```

## Documentation

Will be added shortly.

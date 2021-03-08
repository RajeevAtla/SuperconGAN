# SuperconGAN

[![Build Status](https://travis-ci.com/RajeevAtla/SuperconGAN.svg?branch=master)](https://travis-ci.com/RajeevAtla/SuperconGAN)

A program to train a GAN using superconductivity data.
It was inspired by and is based off of the CTGAN library for generating GANs for tabular datasets.

To install, please use the following command in a terminal window:

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

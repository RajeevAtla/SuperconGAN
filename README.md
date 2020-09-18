# SuperconGAN

A program to create a GAN using superconductivity data.
It was inspired by and is based off of the CTGAN library for generated GANs for tabular datasets.

## Individual Parts

### read_data.py
This function reads data.
The input is:
- the filename of the csv file name you wish to read (csv_filename)
- the filename of the metadata for the csv file (meta_filename)
- whether or not you want the program to infer a header in csv file (header)
- which of the columns are discrete variables (discrete)

The output is:
- The data read by the function (data)
- Which columns are discrete (discrete_columns)

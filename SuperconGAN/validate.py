"""
Module to predict superconductivity in a material
"""

import sdmetrics
from sdmetrics.single_table import MLEfficacy
import pandas as pd
import numpy as np


class Validator():
    """
    Docstring
    """
    def __init__(self, predicted_data, verbose, aggregate = True):
        """
        Docstring
        """
        if verbose:
            print("Started importing data!")

        url_1 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/train.csv"
        url_2 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/unique_m.csv"

        material_data = pd.read_csv(url_1, header = 0, sep = ",")
        material_data.pop("critical_temp")

        element_data = pd.read_csv(url_2, header = 0, sep = ",")
        element_data.pop("material")
        element_data.pop("critical_temp")

        data = material_data.join(element_data)
        discrete_columns = ['number_of_elements']

        if verbose:
            print("Finished importing data!")

        self.real_data = data
        self.predicted_data = predicted_data
        self.verbose = verbose
        self.aggregate = aggregate

    def validate(self):
        """
        Docstring
        """
        if self.verbose:
            print("Started validating data!")
            print("Started defining metrics!")

        val_scores = []

        for target in self.real_data:
            score = MLEfficacy.compute(real_data = self.real_data,
                synthetic_data = self.predicted_data,
                target = target)

            val_scores.append(score)

        for score in val_scores:
            score = np.tanh(score)

        return np.nanmean(val_scores)

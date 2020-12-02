"""
This files serves only to get data from a UCI repository.
"""
import pandas as pd

def get_data():
    """
    Function used to obtain data from url
    """
    print("Started importing data!")

    URL1 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/train.csv"
    URL2 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/unique_m.csv"

    material_data = pd.read_csv(URL1, header = 0, sep = ",")
    material_data.pop("critical_temp")

    element_data = pd.read_csv(URL2, header = 0, sep = ",")
    element_data.pop("material")
    element_data.pop("critical_temp")

    combined_data = [material_data, element_data]
    data = pd.concat(combined_data)
    print("Finished importing data!")
    return data

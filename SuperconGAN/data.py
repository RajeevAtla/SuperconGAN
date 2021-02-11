import pandas as pd

def get_data():
    """
    Function used to obtain data from a URL going to a UCI repository
    """
    print("Started importing data!")

    URL1 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/train.csv"
    URL2 = "https://raw.githubusercontent.com/RajeevAtla/Superconductivity-Dataset/master/unique_m.csv"

    material_data = pd.read_csv(URL1, header = 0, sep = ",")
    material_data.pop("critical_temp")

    element_data = pd.read_csv(URL2, header = 0, sep = ",")
    element_data.pop("material")
    element_data.pop("critical_temp")

    data = material_data.join(element_data)


    discrete_columns = ['number_of_elements']

    print("Finished importing data!")
    return data, discrete_columns

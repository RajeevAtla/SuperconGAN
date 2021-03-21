"""
This module serves to create the model, which will be evaluated in another module.
"""

from ctgan import CTGANSynthesizer
import torch
import pandas as pd
class Synthesizer():

    """
    The class to make the model.

    The following is from CTGAN's documentation.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step. Has to be divisible by 10.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the [WGAN paper](https://arxiv.org/abs/1701.07875). WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
    """

    def __init__(
        self,
        embedding_dim = 128,
        generator_dim = (256, 256),
        discriminator_dim = (256, 256),
        generator_lr = 2e-4,
        generator_decay = 1e-6,
        discriminator_lr = 2e-4,
        discriminator_decay = 1e-6,
        batch_size = 500,
        discriminator_steps = 1,
        log_frequency = True,
        verbose = True,
        epochs = 300):

        if verbose:
            print("Started initializing model!")


        if batch_size % 10 != 0:
            raise ValueError('Batch size must be divisible by 10.')

        model = CTGANSynthesizer(
            embedding_dim = embedding_dim,
            generator_dim = generator_dim,
            discriminator_dim = discriminator_dim,
            generator_lr = generator_lr,
            generator_decay = generator_decay,
            discriminator_lr = discriminator_lr,
            discriminator_decay = discriminator_decay,
            batch_size = batch_size,
            discriminator_steps = discriminator_steps,
            log_frequency = log_frequency,
            verbose = verbose,
            epochs = epochs)

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

        self.verbose = verbose
        self.model = model
        self.data, self.discrete_columns = data, discrete_columns

        if verbose:
            print("Finished initializing model!")

    def fit(self):

        """
        Function to fit to superconductivity data.
        """
        if self.verbose:
            print("Started fitting data!")

        model = self.model
        data = self.data
        discrete_columns = self.discrete_columns

        model.fit(data, discrete_columns)

        if self.verbose:
            print("Finished fitting data!")

    def sample(
        self,
        n,
        condition_column = None,
        condition_value = None):

        """
        Taken from CTGAN.
        Args:
            n (int):
                The number of rows of data to generate.
            condition_column (string):
                Name of a discrete column
            condition_value:
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        """
        if self.verbose:
            print("Started sampling data!")

        model = self.model

        samples = model.sample(
            n = n,
            condition_column = condition_column,
            condition_value = condition_value)

        if self.verbose:
            print("Finished sampling data!")

        return samples

    def save(self, path = 'my_model.pkl'):
        """
        Args:
            path (string):
                File path to where the model should be saved and what the .pkl file's name should
                be.
        """
        if self.verbose:
            print("Started saving model!")

        model = self.model

        model.save(path)

        if self.verbose:
            print("Finished saving model!")

    @staticmethod
    def load(self, path):
        """
        Args:
            path (string):
                File path to where the model should be saved and what the .pkl file's name should
                be.
        """

        if self.verbose:
            print("Started loading model!")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torch.load(path)
        model.set_device(device)

        if self.verbose:
            print("Finished loading model!")
        
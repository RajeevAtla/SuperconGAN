from ctgan import CTGANSynthesizer
import torch
import pandas as pd
from data import get_data

class Synthesizer():

    def __init__(
        self,
        *args,
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

        print("Started initializing model!")
        assert batch_size % 10 == 0

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
        
        self.model = model
        self.data, self.discrete_columns = get_data()

        print("Finished initializing model!")
    
    def fit(self):
        
        print("Started fitting data!")

        model = self.model
        data = self.data
        discrete_columns = self.discrete_columns

        model.fit(data, discrete_columns)
        print("Finished fitting data!")
    
    def sample(self, n, condition_column = None, condition_value = None):
        print("Started sampling data!")

        model = self.model

        samples = model.sample(
            n = n,
            condition_column = condition_column,
            condition_value = condition_value)

        print("Finished sampling data!")

        return samples

    def save(self, path = 'my_model.pkl'):
        print("Started saving model!")
 
        model = self.model

        model.save(path)

        print("Finished saving model!")

    @staticmethod
    def load(cls, path):
        print("Started loading model!")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torch.load(path)
        model.set_device(device)

        print("Finished loading model!")
        
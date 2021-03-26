import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#get data from .txt files
df1 = pd.read_csv('file1.txt', delimiter = ",", header = None, names = ['Epoch', 'Loss G', 'Loss D'])
df2 = pd.read_csv('file2.txt', delimiter = ",", header = None, names = ['Epoch', 'Loss G', 'Loss D'])
df3 = pd.read_csv('file3.txt', delimiter = ",", header = None, names = ['Epoch', 'Loss G', 'Loss D'])

#combine the dataframes into one and drop epochs column since its just the index
df = pd.concat([df1, df2, df3], axis = 1)
df = df.drop(columns = 'Epoch', axis = 1)

#split data into discriminator and generator loss
dfG = df.drop(columns = 'Loss D')
dfD = df.drop(columns = 'Loss G')

#find mean
dfMeanG = dfG.mean(axis = 1)
dfMeanD = dfD.mean(axis = 1)

#find standard deviation
dfstdG = dfG.std(axis = 1)
dfstdD= dfD.std(axis = 1)

#find epochs
epochs = np.linspace(1, 250, num = 250)

plt.plot(epochs, dfMeanG, color='red', label = 'Generator Loss')
plt.plot(epochs, dfMeanD, color = 'green', label = 'Discriminator Loss')
plt.title('Loss vs. Epochs Trained')
plt.xlabel('Epochs Trained')
plt.ylabel('Loss')
plt.legend(loc="lower right")
plt.savefig('graph1.jpeg')
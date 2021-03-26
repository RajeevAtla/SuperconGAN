import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.special import erfinv

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

z_crit = erfinv(0.95)

fig, ax = plt.subplots(2, 1)

ax[0].plot(epochs, dfMeanG)
ax[0].fill_between(epochs, dfMeanG - z_crit * (dfstdG/math.sqrt(3)), dfMeanG + z_crit * (dfstdG/math.sqrt(3)), alpha=0.25)
ax[0].set_xlabel('Epochs Trained')
ax[0].set_ylabel('Generator Loss')


ax[1].plot(epochs, dfMeanD, color = 'red')
ax[1].fill_between(epochs, dfMeanD - z_crit * (dfstdD/math.sqrt(3)), dfMeanD + z_crit * (dfstdD/math.sqrt(3)), alpha=0.25, color = 'red')
ax[1].set_xlabel('Epochs Trained')
ax[1].set_ylabel('Discriminator Loss')

fig.tight_layout()
fig.suptitle("Loss vs. Epochs Trained", y = 1)
plt.savefig('graph2.jpeg')
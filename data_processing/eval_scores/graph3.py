import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.txt', delimiter = ',')

epochs = df['Epochs']
eval_scores = df.iloc[:,1]

fig, ax = plt.subplots()
ax.plot(epochs, eval_scores)
ax.set_xlabel('Epochs')
ax.set_ylabel('Evaluation Score')
ax.set_title('Evaluation Score vs Epochs')
plt.savefig('graph3.jpeg')

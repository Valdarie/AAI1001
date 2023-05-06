import os
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

folder = r'C:\Users\Xuanting\Desktop\data'  # Remember to change to your file paths for the "data" folder (paths cannot have spacings)
num_exp = 2
num_lr = 5
num_reps = 10

# Initialize an empty list to store the data from all files
data_list = []

# Iterate over all experiment, learning rate, and repetition indices
for exp in range(num_exp):
    for lr in range(num_lr):
        for rep in range(num_reps):
            # Construct the filename and file path for the current file
            filename = 'exp{:d}_lrind{:d}_run{:d}_acc.pt'.format(exp+1, lr, rep)
            filepath = os.path.join(folder, filename)

            # Load the data from the file using PyTorch's torch.load() function
            with open(filepath, 'rb') as f:
                data = torch.load(f)

            # Append the data to the list of data
            data_list.append({'experiment': exp+1, 'learning_rate': lr, 'repetition': rep, 'accuracy': data.item()})

# Convert the list of data to a pandas dataframe
df = pd.DataFrame(data_list)

# Set the seaborn style
sns.set_style("whitegrid")

# Plot the accuracy values for experiment 1
sns.lineplot(data=df[df['experiment'] == 1], x='learning_rate', y='accuracy',
             errorbar='sd', label='Experiment 1')

# Plot the accuracy values for experiment 2
sns.lineplot(data=df[df['experiment'] == 2], x='learning_rate', y='accuracy',
             errorbar='sd', label='Experiment 2')

plt.show()
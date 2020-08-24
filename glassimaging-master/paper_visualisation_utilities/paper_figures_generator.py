import numpy as np
import matplotlib.pyplot as plt

# This short piece of code generates a bunch of figures to demonstrate how the eventual figures for the paper would look like

full_data = []
for i in range(8):
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    full_data.append(data)

fig, axs = plt.subplots(1, 1)

# basic plot
axs.boxplot(full_data)
axs.set_title('basic plot')

fig.show()

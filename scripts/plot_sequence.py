"""
You perform a sequence simulation in IDEA (Siemens), make a sequence plot,
and save the save as .txt.

Then you can use this script to plot a nice looking sequence diagram
for presentations or papers.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('>> file directory: ', DIR)

# %%
# specify the .txt file here:
with open(DIR + '/SimulationProtocol_SMS_Segment.txt', 'r') as file:
    lines = file.readlines()

line_arrays = []
for line in lines[10:]:
    str = line.split()
    arr = [float(s) for s in str]
    line_arrays.append(arr)

line_arrays = np.array(line_arrays)
line_arrays = line_arrays[23900:136800, :]  # specify the time interval to be plotted

print(line_arrays.shape)

array_ADC = line_arrays[:, 0]  # specify the ADC column in the line_arrays
array_GX = line_arrays[:,  5]  # specify the GX  column in the line_arrays
array_GY = line_arrays[:,  6]  # specify the GY  column in the line_arrays
array_GZ = line_arrays[:,  1]  # specify the GZ  column in the line_arrays
array_RF = line_arrays[:,  4]  # specify the RF  column in the line_arrays

N_samples = line_arrays.shape[0]

sample_time = np.arange(1, N_samples+1, 1) * 0.01  # ms

# %%
fig, ax = plt.subplots(4, 1, figsize=(12, 4))

axes_str = ['RF', '$G_z$', '$G_y$', '$G_x$']

ax[0].plot(sample_time, array_RF, '-b')
ax[1].plot(sample_time, array_GZ, '-b')
ax[2].plot(sample_time, array_GY, '-b')
ax[3].plot(sample_time, array_GX, '-b')

for n in range(4):
    ax[n].annotate("", xy=(sample_time[-1], 0),
                   xytext=(sample_time[ 0], 0),
                   arrowprops=dict(arrowstyle="->"))
    ax[n].text(-40, 0, axes_str[n], fontsize=12)

for n in range(4):
    ax[n].set_xticks([sample_time[0], sample_time[-1]])
    ax[n].set_axis_off()

plt.savefig(DIR + '/seq.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

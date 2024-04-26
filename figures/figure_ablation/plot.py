import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> current directory: ', DIR)

# %% ablation 1: regularization weight
print('> ablation 1')
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

f = h5py.File(DIR + '/ablation_regu.h5', 'r')

regu_list = [0, 0.01, 0.02]

for r in range(len(regu_list)):

    regu = regu_list[r]

    data_str = 'lamda_%4.2f'%regu

    dwi_comb = f[data_str][:]

    dwi_comb = np.flip(dwi_comb, axis=[-2, -1])

    N_y, N_x = dwi_comb.shape[-2:]

    img = abs(dwi_comb[1, 0, 0, 1, :, :])

    print(dwi_comb.shape)

    ax[r].imshow(img, cmap='gray',
                 vmin=0, vmax=img.max() * 0.6,
                 interpolation='None')


    if r == 0:

        rect_x0 = int(N_x*0.48)
        rect_y0 = int(N_y*0.50)

        Rect = Rectangle((rect_x0, rect_y0), 100, 100, edgecolor='y', facecolor='none')
        ax[r].add_patch(Rect)


    if regu == 0.01:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    else:
        props=None


    ax[r].text(0.76 * N_x, 0.96 * N_y, '%.2f'%regu,
               bbox=props,
               color='w', fontsize=20)
    ax[r].set_axis_off()

plt.suptitle('(A) varying $\lambda$, keeping block as 6 and stride as 1',
             horizontalalignment='left',
             x=fig.subplotpars.left,
             fontsize=21, fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/ablation_regu.png',
            bbox_inches='tight', pad_inches=0, dpi=300)


# %% ablation 2: LLR block size
print('> ablation 2')
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

f = h5py.File(DIR + '/ablation_block.h5', 'r')

block_list = [3, 6, 9]

for r in range(len(block_list)):

    block = block_list[r]

    data_str = 'block_%2d'%block

    dwi_comb = f[data_str][:]

    dwi_comb = np.flip(dwi_comb, axis=[-2, -1])

    N_y, N_x = dwi_comb.shape[-2:]

    img = abs(dwi_comb[1, 0, 0, 1, :, :])

    print(dwi_comb.shape)

    ax[r].imshow(img, cmap='gray',
                 vmin=0, vmax=img.max() * 0.6,
                 interpolation='None')

    if block == 6:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    else:
        props=None

    ax[r].text(0.76 * N_x, 0.96 * N_y, '%2d'%block,
               bbox=props,
               color='w', fontsize=20)
    ax[r].set_axis_off()

plt.suptitle('(B) varying block width, keeping $\lambda$ 0.01 and stride as 1',
             horizontalalignment='left',
             x=fig.subplotpars.left,
             fontsize=21, fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/ablation_block.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

# %% ablation 3: LLR overlap factor
print('> ablation 3')
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

f = h5py.File(DIR + '/ablation_stride.h5', 'r')

stride_list = [1, 3, 6]

for r in range(len(stride_list)):

    stride = stride_list[r]

    data_str = 'stride_%2d'%stride

    dwi_comb = f[data_str][:]

    dwi_comb = np.flip(dwi_comb, axis=[-2, -1])

    N_y, N_x = dwi_comb.shape[-2:]

    img = abs(dwi_comb[1, 0, 0, 1, :, :])

    print(dwi_comb.shape)

    img_res = img[rect_y0 : rect_y0 + 100, rect_x0 : rect_x0 + 100]

    ax[r].imshow(img_res, cmap='gray',
                 vmin=0, vmax=img.max() * 0.6,
                 interpolation='None')

    if stride == 1:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    else:
        props=None

    ax[r].text(0.76 * 100, 0.96 * 100, '%2d'%stride,
               bbox=props,
               color='w', fontsize=20)
    ax[r].set_axis_off()

plt.suptitle('(C) varying stride, keeping $\lambda$ as 0.01 and block as 6',
             horizontalalignment='left',
             x=fig.subplotpars.left,
             fontsize=21, fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/ablation_stride.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

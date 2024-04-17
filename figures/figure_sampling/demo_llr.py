"""
This script demonstrates
 * the rationale of locally low rank regularization for DWIs.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import sigpy as sp
from sigpy import linop, util

import h5py
import os

DIR = os.path.dirname(os.path.realpath(__file__))

# %%
SRC_FILE = DIR + '/JETS_slice_059.h5'

f = h5py.File(SRC_FILE, 'r')
DWI = f['DWI'][:]
f.close()

N_diff, N_y, N_x = DWI.shape

N_cube = int(N_x * 0.75)

DWI_crop = sp.resize(DWI, [N_diff] + [N_cube] * 2)
DWI_crop = np.flip(DWI_crop, axis=1)

N_diff, N_y, N_x = DWI_crop.shape
print('DWI shape - ')
print('  # of DWIs: ', str(N_diff))
print('  # of phase encodings: ', str(N_y))
print('  # of readouts: ', str(N_x))

# %%
blk_shape = [9, 9]
blk_strides = [9, 9]


ATB = sp.linop.ArrayToBlocks([N_diff] + blk_shape,
                             blk_shape, blk_strides)


def _linop_reshape(ATB):

    D = len(ATB.blk_shape)

    oshape = [util.prod(ATB.ishape[:-D]),
              util.prod(ATB.num_blks),
              util.prod(ATB.blk_shape)]

    R1 = linop.Reshape(oshape, ATB.oshape)
    R2 = linop.Transpose(R1.oshape, axes=(1, 0, 2))
    return R2 * R1


Reshape = _linop_reshape(ATB)


Fwd = Reshape * ATB  # Compose forward linear operator

# %%
dwi_idx = [19, 43, 119]
dwi_colors = ['g', 'b', 'r']

blk_colors = ['y']
blk_x0 = [68]
blk_y0 = [96]
blk_idx = []
for b in range(len(blk_colors)):
    slc_d = slice(0, N_diff)
    slc_y = slice(blk_y0[b], blk_y0[b] + blk_shape[-2])
    slc_x = slice(blk_x0[b], blk_x0[b] + blk_shape[-1])

    blk_idx.append((slc_d, slc_y, slc_x))


f, ax = plt.subplots(1, 3, figsize=(8.2, 3))

for d in range(3):
    img = abs(DWI_crop[dwi_idx[d], ...])
    ax[d].imshow(img, vmin=0, vmax=0.00012, cmap='gray')
    ax[d].set_axis_off()
    ax[d].set_xticklabels([])
    ax[d].set_yticklabels([])

    # label shells
    ax[d].scatter(N_x*0.05, N_y*0.05, color=dwi_colors[d], s=160, marker='*')

    # label blocks
    for b in range(len(blk_colors)):
        Rect = Rectangle((blk_idx[b][-1].start, blk_idx[b][-2].start),
                         blk_shape[-1], blk_shape[-2],
                         edgecolor=blk_colors[b], facecolor='none')
        ax[d].add_patch(Rect)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/DWI.png', bbox_inches='tight',
            pad_inches=0, dpi=300)

# %%

fontsize = 16

f, ax = plt.subplots(1, 2, figsize=(8.2, 3))

for b in range(len(blk_colors)):
    rect = DWI_crop[blk_idx[b]]
    dest = Fwd(rect)

    ax[0].imshow(abs(np.squeeze(dest)), cmap='gray')
    ax[0].set_axis_off()

    u, s, vh = np.linalg.svd(dest, full_matrices=False)

    s *= 1000

    ax[1].plot(s[0, ...], color=blk_colors[b], linewidth=3)
    ax[1].grid(color='gray', linestyle='--')
    ax[1].set_ylim(0, int(np.amax(s)))
    ax[1].set_xlim(0, s.shape[-1])

    ax[1].set_yticks([0, int(np.amax(s))], fontsize=fontsize)
    ax[1].set_xticks([0, s.shape[-1]], fontsize=fontsize)

plt.savefig(DIR + '/DWI_LLR_Property.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

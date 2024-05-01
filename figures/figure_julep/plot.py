import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sigpy as sp

from sigpy.mri import sms

import torchvision.transforms as T

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> curr directory: ', DIR)

HOME_DIR = DIR.rsplit('/', 1)[0].rsplit('/', 1)[0]
print('> home directory: ', HOME_DIR)

JULEP_DATA_DIR = HOME_DIR + '/JULEP/data/'

# %%
print('> read in MUSE and JULEP recon result from Dr. Erpeng Dai.')

f = h5py.File(JULEP_DATA_DIR + '/img_MB3_julep_7t.mat', 'r')
R_julep = f['b1imfinal2'][:]  # float
f.close()


f = h5py.File(JULEP_DATA_DIR + '/img_MB3_muse_7t.mat', 'r')
R_muse = f['b1imfinal2'][:]  # float
f.close()

N_band, N_y, N_x = R_muse.shape
N_accel_PE = 1


# SMS phase shift
yshift = []
for b in range(N_band):
    yshift.append(b / N_accel_PE)

sms_phase = sms.get_sms_phase_shift([N_band, N_y, N_x],
                                    MB=N_band,
                                    yshift=yshift)

R_muse = sp.ifft(sms_phase * sp.fft(R_muse, axes=[-2, -1]), axes=[-2, -1])
R_julep = sp.ifft(sms_phase * sp.fft(R_julep, axes=[-2, -1]), axes=[-2, -1])

R_muse = np.flip(R_muse, axis=[-2, -1])
R_julep = np.flip(R_julep, axis=[-2, -1])


# %% read in denoised MUSE recon
# f = h5py.File(DIR + '/MUSE_cplx_denoise.h5', 'r')
# R_muse_denoise = np.transpose(f['DWI'][:])
# f.close()

# R_muse_denoise_63 = abs(R_muse_denoise[10, 63, ...])
# R_muse_denoise_63 = np.flip(R_muse_denoise_63, axis=[-2, -1])

# print('> R_muse_denoise shape: ', R_muse_denoise.shape)

# %%
print('> read in JETS recon.')

f = h5py.File(DIR + '/jets.h5', 'r')
dwi_fully_jets = np.flip(np.squeeze(f['DWI'][:]), axis=[-2, -1])
f.close()

dwi_fully_jets_7 = dwi_fully_jets[10, ...]

# %%
print('> plotting')
N_row, N_col = 2, 3

f, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4+1))

jets_img = abs(dwi_fully_jets_7[1, ...])
jets_max = np.amax(jets_img) * 0.5
ax[0][0].imshow(jets_img, vmin=0, vmax=jets_max,
             cmap='gray', interpolation='none')
ax[0][0].text(0.02*N_x, 0.08*N_y, 'JETS',
           color='w', fontsize=14, weight='bold')

rect_x0 = int(N_x/2.4)
rect_y0 = N_y//3

Rect = Rectangle((rect_x0, rect_y0), 50, 50, edgecolor='y', facecolor='none')
ax[0][0].add_patch(Rect)


# TR = T.Resize([N_y, N_x])
# jets_tensor = sp.to_pytorch(jets_img)
# print(jets_tensor.shape)
# jets_res = TR(jets_tensor).cpu().detach().numpy()

jets_res = jets_img[rect_y0 : rect_y0 + 50, rect_x0 : rect_x0 + 50]

ax[1][0].imshow(jets_res, vmin=0, vmax=jets_max,
             cmap='gray', interpolation='none')


muse_img = abs(R_muse[1, ...])
muse_max = np.amax(muse_img) * 0.5
ax[0][1].imshow(muse_img, vmin=0, vmax=muse_max,
             cmap='gray', interpolation='none')
ax[0][1].text(0.02*N_x, 0.08*N_y, 'MUSE',
           color='w', fontsize=14, weight='bold')

muse_res = muse_img[rect_y0 : rect_y0 + 50, rect_x0 : rect_x0 + 50]

ax[1][1].imshow(muse_res, vmin=0, vmax=muse_max,
             cmap='gray', interpolation='none')


# muse_denoise_max = np.amax(R_muse_denoise_63) * 0.5
# ax[0][2].imshow(R_muse_denoise_63, vmin=0, vmax=muse_denoise_max,
#              cmap='gray', interpolation='none')
# ax[0][2].text(0.02*N_x, 0.08*N_y, 'MUSE + Denoiser',
#            color='w', fontsize=14, weight='bold')

# mues_denoise_res = R_muse_denoise_63[rect_y0 : rect_y0 + 50, rect_x0 : rect_x0 + 50]

# ax[1][2].imshow(mues_denoise_res, vmin=0, vmax=muse_denoise_max,
#              cmap='gray', interpolation='none')


julep_img = abs(R_julep[1, ...])
julep_max = np.amax(julep_img) * 0.5
ax[0][2].imshow(julep_img, vmin=0, vmax=julep_max,
             cmap='gray', interpolation='none')
ax[0][2].text(0.02*N_x, 0.08*N_y, 'JULEP',
           color='w', fontsize=14, weight='bold')

julep_res = julep_img[rect_y0 : rect_y0 + 50, rect_x0 : rect_x0 + 50]

ax[1][2].imshow(julep_res, vmin=0, vmax=julep_max,
             cmap='gray', interpolation='none')


for m in range(N_row):
    for n in range(N_col):
        ax[m][n].set_axis_off()


f.tight_layout()

plt.suptitle('8th DW image from 4-shot iEPI @ 1 mm ISO',
             fontsize=24, fontweight='bold')
plt.subplots_adjust(top=0.9)  # because of tight_layout()

# plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/julep.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

import h5py
import numpy as np

import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> current directory: ', DIR)

# %% read in files
# fully sampled
f = h5py.File('recon_fully.h5', 'r')
# dwi_fully_muse = np.flip(np.squeeze(f['muse'][:]), axis=-2)
dwi_fully_jets = np.flip(np.squeeze(f['jets'][:]), axis=[-2, -1])
f.close()

# retro with shift 0
f = h5py.File('recon_retr0.h5', 'r')
# dwi_retr0_muse = np.flip(np.squeeze(f['muse'][:]), axis=-2)
dwi_retr0_jets = np.flip(np.squeeze(f['jets'][:]), axis=[-2, -1])
f.close()

# retro with shift 1
f = h5py.File('recon_retr1.h5', 'r')
# dwi_retr1_muse = np.flip(np.squeeze(f['muse'][:]), axis=-2)
dwi_retr1_jets = np.flip(np.squeeze(f['jets'][:]), axis=[-2, -1])
f.close()

N_y, N_x = dwi_fully_jets.shape[-2:]
# %% plot one-dir DWI - JETS
f, ax = plt.subplots(2, 3, figsize=(12.3, 8))

diff_idx = 10

# refer_muse = abs(np.squeeze(dwi_fully_muse[diff_idx, 1, ...]))
refer_jets = abs(np.squeeze(dwi_fully_jets[diff_idx, 1, ...]))
retr0_jets = abs(np.squeeze(dwi_retr0_jets[diff_idx, 1, ...]))
retr1_jets = abs(np.squeeze(dwi_retr1_jets[diff_idx, 1, ...]))


NRMSE_retr0_jets = mean_squared_error(retr0_jets, refer_jets)
NRMSE_retr1_jets = mean_squared_error(retr1_jets, refer_jets)

ssim_retr0_jets = ssim(refer_jets, retr0_jets, data_range=retr0_jets.max() - retr0_jets.min())
ssim_retr1_jets = ssim(refer_jets, retr1_jets, data_range=retr1_jets.max() - retr1_jets.min())

print('> 7th: fully_muse vs retr0_jets nrmse - ' + str(NRMSE_retr0_jets))
print('> 7th: fully_muse vs retr1_jets nrmse - ' + str(NRMSE_retr1_jets))

print('> 7th: fully_muse vs retr0_jets ssim - ' + str(ssim_retr0_jets))
print('> 7th: fully_muse vs retr1_jets ssim - ' + str(ssim_retr1_jets))


vmax = np.amax(refer_jets) * 0.5
print(vmax)

ax[0, 0].imshow(refer_jets,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')
ax[0, 0].set_title('4-shot iEPI', fontsize=18)

ax[0, 1].imshow(retr0_jets,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')
ax[0, 1].set_title('retro. 1-shot w/o shift', fontsize=18)
ax[0, 1].text(0.75*N_x, 0.96*N_y, '%5.3f'%ssim_retr0_jets,
              color='w', fontsize=16, weight='bold')

ax[0, 2].imshow(retr1_jets,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')
ax[0, 2].set_title('retro. 1-shot w/ shift', fontsize=18)
ax[0, 2].text(0.75*N_x, 0.96*N_y, '%5.3f'%ssim_retr1_jets,
              color='w', fontsize=16, weight='bold')

ax[0, 0].set_ylabel('8th DWI', fontsize=18)


# %% plot averaged DWI
# dwi_fully_muse_ave = np.average(abs(dwi_fully_muse), axis=0)
dwi_fully_jets_ave = np.average(abs(dwi_fully_jets), axis=0)
dwi_retr0_jets_ave = np.average(abs(dwi_retr0_jets), axis=0)
dwi_retr1_jets_ave = np.average(abs(dwi_retr1_jets), axis=0)


# refer_muse_ave = dwi_fully_muse_ave[1, ...]
refer_jets_ave = dwi_fully_jets_ave[1, ...]
retr0_jets_ave = dwi_retr0_jets_ave[1, ...]
retr1_jets_ave = dwi_retr1_jets_ave[1, ...]

vmax = np.amax(refer_jets_ave) * 0.5
print(vmax)


diff_retr0_jets_ave = refer_jets_ave - retr0_jets_ave
diff_retr1_jets_ave = refer_jets_ave - retr1_jets_ave

NRMSE_retr0_jets_ave = mean_squared_error(retr0_jets_ave, refer_jets_ave)
NRMSE_retr1_jets_ave = mean_squared_error(retr1_jets_ave, refer_jets_ave)

ssim_retr0_jets_ave = ssim(refer_jets_ave, retr0_jets_ave, data_range=retr0_jets_ave.max() - retr0_jets_ave.min())
ssim_retr1_jets_ave = ssim(refer_jets_ave, retr1_jets_ave, data_range=retr1_jets_ave.max() - retr1_jets_ave.min())

# print('> fully_muse vs fully_jets nrmse - ' + str(NRMSE_retr0_jets_ave))
print('> ave: fully_muse vs retr0_jets nrmse - ' + str(NRMSE_retr0_jets_ave))
print('> ave: fully_muse vs retr1_jets nrmse - ' + str(NRMSE_retr1_jets_ave))

print('> ave: fully_muse vs retr0_jets ssim - ' + str(ssim_retr0_jets_ave))
print('> ave: fully_muse vs retr1_jets ssim - ' + str(ssim_retr1_jets_ave))


ax[1, 0].imshow(refer_jets_ave,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')

ax[1, 1].imshow(retr0_jets_ave,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')
ax[1, 1].text(0.75*N_x, 0.96*N_y, '%5.3f'%ssim_retr0_jets_ave,
              color='w', fontsize=16, weight='bold')

ax[1, 2].imshow(retr1_jets_ave,
                cmap='gray', vmin=0, vmax=vmax,
                interpolation='none')
ax[1, 2].text(0.75*N_x, 0.96*N_y, '%5.3f'%ssim_retr1_jets_ave,
              color='w', fontsize=16, weight='bold')

ax[1, 0].set_ylabel('mean DWI', fontsize=18)



for m in range(2):
    for n in range(3):
        ax[m][n].set_xticks([])
        ax[m][n].set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/gt.png',
            bbox_inches='tight', pad_inches=0, dpi=300)

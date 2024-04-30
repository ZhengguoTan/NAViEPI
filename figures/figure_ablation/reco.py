import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torchvision.transforms as T

from sigpy.mri import app, muse, retro, sms

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> current directory: ', DIR)

HOME_DIR = DIR.rsplit('/', 1)[0].rsplit('/', 1)[0]

DAT_DIR = HOME_DIR + '/data'
print('> data directory: ', DAT_DIR)

# %%

# navi
f = h5py.File(DAT_DIR + '/0.5x0.5x2.0mm_R3x2_navi_slice_000.h5', 'r')
navi = f['navi'][:]
MB = f['MB'][()]
f.close()

navi = np.squeeze(navi)
navi = np.swapaxes(navi, -2, -3)
navi = np.swapaxes(navi, 0, 1)
navi = navi[..., None, :, :]

N_diff, N_shot, N_coil, N_z, N_y_navi, N_x_navi = navi.shape

print('> navi shape: ', navi.shape)


# kdat
f = h5py.File(DAT_DIR + '/0.5x0.5x2.0mm_R3x2_kdat_slice_000.h5', 'r')
kdat = f['kdat'][:]
N_segments = f['Segments'][()]
N_slices = f['Slices'][()]
N_Accel_PE = f['Accel_PE'][()]
f.close()

kdat = np.squeeze(kdat)
kdat = np.swapaxes(kdat, -2, -3)

N_diff = kdat.shape[-4]
kdat_prep = []
for d in range(N_diff):
    k = retro.split_shots(kdat[d, ...], shots=N_segments)
    kdat_prep.append(k)

kdat_prep = np.array(kdat_prep)
kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

print('> kdat_prep shape: ', kdat_prep.shape)


# coil
f = h5py.File(DAT_DIR + '/0.5x0.5x2.0mm_R3x2_coil.h5', 'r')
coil = f['coil'][:]
f.close()

coil1 = coil[:, [1, 31], :, :]
print('> coil1 shape: ', coil1.shape)

N_y, N_x = coil1.shape[-2:]

mask = sp.rss(coil1[:, 1, ...])

# %% SMS phase shift
N_Accel_PE = 3

yshift = []
for b in range(MB):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x],
                                    MB=MB,
                                    yshift=yshift)


navi_small_fov = [N_y_navi, N_x_navi * 2]

navi_resi = sp.resize(navi,
                      oshape=list(navi.shape[:-2]) +\
                                  navi_small_fov)

coil_tensor = sp.to_pytorch(coil1)
TR = T.Resize(navi_small_fov)
coil_resi_r = TR(coil_tensor[..., 0]).cpu().detach().numpy()
coil_resi_i = TR(coil_tensor[..., 1]).cpu().detach().numpy()
coil_resi = coil_resi_r + 1j * coil_resi_i

sms_phase_resi = sms.get_sms_phase_shift([MB] + navi_small_fov,
                                            MB=MB, yshift=yshift)

# %% shot phase
_, navi_shot_muse = muse.MuseRecon(
                                navi_resi, coil_resi, MB=MB,
                                acs_shape=navi_small_fov,
                                lamda=0.001, max_iter=60,
                                yshift=yshift,
                                device=sp.Device(-1))

_, navi_shot_muse_phase = muse._denoising(navi_shot_muse,
                                          full_img_shape=[N_y, N_x])


# %% ablation 1: regularizatin weight

regu_list = [0, 0.01, 0.02]

f = h5py.File(DIR + '/ablation_regu.h5', 'w')

for r in range(len(regu_list)):

    regu = regu_list[r]

    dwi_comb = app.HighDimensionalRecon(
                            kdat_prep, coil1,
                            phase_sms=sms_phase,
                            combine_echo=True,
                            phase_echo=np.conj(navi_shot_muse_phase),
                            regu='LLR',
                            blk_shape=(1, 6, 6),
                            blk_strides=(1, 1, 1),
                            normalization=True,
                            solver='ADMM',
                            lamda=regu,
                            rho=0.05,
                            max_iter=15,
                            show_pbar=False, verbose=True,
                            device=sp.Device(-1)).run()

    data_str = 'lamda_%4.2f'%regu
    f.create_dataset(data_str, data=sp.to_device(dwi_comb))

f.close()

# %% ablation 2: LLR block size

block_list = [3, 6, 9]

f = h5py.File(DIR + '/ablation_block.h5', 'w')

for r in range(len(block_list)):

    block = block_list[r]

    dwi_comb = app.HighDimensionalRecon(
                            kdat_prep, coil1,
                            phase_sms=sms_phase,
                            combine_echo=True,
                            phase_echo=np.conj(navi_shot_muse_phase),
                            regu='LLR',
                            blk_shape=(1, block, block),
                            blk_strides=(1, 1, 1),
                            normalization=True,
                            solver='ADMM',
                            lamda=0.01,
                            rho=0.05,
                            max_iter=15,
                            show_pbar=False, verbose=True,
                            device=sp.Device(-1)).run()

    data_str = 'block_%2d'%block
    f.create_dataset(data_str, data=sp.to_device(dwi_comb))

f.close()


# %% ablation 3: LLR overlap factor

stride_list = [1, 3, 6]

f = h5py.File(DIR + '/ablation_stride.h5', 'w')

for r in range(len(stride_list)):

    stride = stride_list[r]

    dwi_comb = app.HighDimensionalRecon(
                            kdat_prep, coil1,
                            phase_sms=sms_phase,
                            combine_echo=True,
                            phase_echo=np.conj(navi_shot_muse_phase),
                            regu='LLR',
                            blk_shape=(1, 6, 6),
                            blk_strides=(1, stride, stride),
                            normalization=True,
                            solver='ADMM',
                            lamda=0.01,
                            rho=0.05,
                            max_iter=15,
                            show_pbar=False, verbose=True,
                            device=sp.Device(-1)).run()

    data_str = 'stride_%2d'%stride
    f.create_dataset(data_str, data=sp.to_device(dwi_comb))

f.close()

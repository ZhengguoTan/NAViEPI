import h5py
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from sigpy.mri import sms, retro

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> current directory: ', DIR)

HOME_DIR = DIR.rsplit('/', 1)[0].rsplit('/', 1)[0]

DAT_DIR = HOME_DIR + '/data'
print('> data directory: ', DAT_DIR)

DAT_JULEP_DIR = HOME_DIR + '/JULEP/data'
print('> data JULEP directory: ', DAT_JULEP_DIR)

# %%
slice_idx = 31

f = h5py.File(DAT_DIR + '/1.0mm_20-dir_R1x3_kdat_slices.h5', 'r')
kdat = f['kdat'][:]
kdat = kdat[slice_idx]
N_band = f['MB'][()]
N_segments = f['Segments'][()]
N_slices = f['Slices'][()]
N_Accel_PE = f['Accel_PE'][()]
f.close()

print('> kdat shape: ', kdat.shape)
print('> N_band ' + str(N_band) + ' N_segments ' + str(N_segments) + ' N_slices ' + str(N_slices) + ' Accel_PE ' + str(N_Accel_PE))

f = h5py.File(DAT_DIR + '/1.0mm_20-dir_R1x3_coil.h5', 'r')
coil = f['coil'][:]

slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(slice_idx, N_slices, N_band)

print('> slice_mb_idx: ', slice_mb_idx)

coil1 = coil[:, slice_mb_idx, :, :]
print('> coil1 shape: ', coil1.shape)

N_y, N_x = coil1.shape[-2:]

# %%
# SMS phase shift
yshift = []
for b in range(N_band):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([N_band, N_y, N_x],
                                    MB=N_band,
                                    yshift=yshift)

# %%
kdat_prep = np.squeeze(kdat)
kdat_prep = np.swapaxes(kdat_prep, -2, -3)

kdat_prep_1 = np.squeeze(kdat_prep[10, ...])  # 10th diffusion
kdat_prep_1a = kdat_prep_1[:, 50:, :]

print(kdat_prep_1a.shape)

f = h5py.File(DAT_JULEP_DIR + '/julep_k.h5', 'w')
f.create_dataset('k', data=kdat_prep_1a)
f.close()

# %%
N_coils, N_band, N_y, N_x = coil1.shape

coil1_shift = sp.ifft(sms_phase * sp.fft(coil1, axes=[-2, -1]), axes=[-2, -1])

coil1_shift_ext = np.swapaxes(coil1_shift, -2, -3)
coil1_shift_ext = np.reshape(coil1_shift_ext, (N_coils, N_y, N_band * N_x))

print(coil1_shift_ext.shape)

f = h5py.File(DAT_JULEP_DIR + '/julep_c.h5', 'w')
f.create_dataset('c', data=coil1_shift_ext)
f.close()
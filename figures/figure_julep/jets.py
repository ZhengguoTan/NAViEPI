import h5py
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

# %% read in k-space data and coil
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
print('> N_band ' + str(N_band) +\
      ' N_segments ' + str(N_segments) +\
      ' N_slices ' + str(N_slices) +\
      ' Accel_PE ' + str(N_Accel_PE))

f = h5py.File(DAT_DIR + '/1.0mm_20-dir_R1x3_coil.h5', 'r')
coil = f['coil'][:]

slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(slice_idx, N_slices, N_band)

coil1 = coil[:, slice_mb_idx, :, :]
print('> coil1 shape: ', coil1.shape)

N_y, N_x = coil1.shape[-2:]

# %% SMS phase shift
yshift = []
for b in range(N_band):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([N_band, N_y, N_x],
                                    MB=N_band,
                                    yshift=yshift)

# %% split k-space data into shots
kdat = np.squeeze(kdat)
kdat = np.swapaxes(kdat, -2, -3)

N_diff = kdat.shape[-4]
kdat_prep = []
for d in range(N_diff):
    k = retro.split_shots(kdat[d, ...], shots=N_segments)
    kdat_prep.append(k)

kdat_prep = np.array(kdat_prep)
kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

N_diff = kdat_prep.shape[0]

print('> kdat_prep shape: ', kdat_prep.shape)

# %% MUSE
def _muse_recon(kdat, coil, yshift=None, MB=3, device=sp.Device(0)):

    N_y, N_x = coil.shape[-2:]

    acs_shape = list(np.array([N_y, N_x]) // 4)

    dwi_muse, dwi_shot = muse.MuseRecon(kdat, coil, MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.01, max_iter=30,
                                yshift=yshift,
                                device=device)

    dwi_muse = sp.to_device(dwi_muse)

    return dwi_muse

# %% JETS
def _jets_recon(kdat, coil, yshift=None, MB=3,
                sms_phase=None,
                pi_shot=True,
                device=sp.Device(0)):

    if kdat.shape[-5] > 1:

        N_y, N_x = coil.shape[-2:]

        shot_fov = list(np.array([N_y, N_x]) // 4)
        kdat_resi = sp.resize(kdat,
                            oshape=list(kdat.shape[:-2]) + shot_fov)

        coil_tensor = sp.to_pytorch(coil)
        TR = T.Resize(shot_fov)
        coil_resi_r = TR(coil_tensor[..., 0]).cpu().detach().numpy()
        coil_resi_i = TR(coil_tensor[..., 1]).cpu().detach().numpy()
        coil_resi = coil_resi_r + 1j * coil_resi_i

        sms_phase_resi = sms.get_sms_phase_shift([MB] + shot_fov,
                                                MB=MB, yshift=yshift)

        acs_shape = shot_fov

        if pi_shot is False:

            dwi_shot = app.HighDimensionalRecon(kdat_resi, coil_resi,
                                        phase_sms=sms_phase_resi,
                                        combine_echo=False,
                                        regu='LLR',
                                        blk_shape=(1, 6, 6),
                                        blk_strides=(1, 1, 1),
                                        normalization=True,
                                        solver='ADMM', lamda=0.01, rho=0.05,
                                        max_iter=15,
                                        show_pbar=False, verbose=True,
                                        device=device).run()

        else:
            _, dwi_shot = muse.MuseRecon(kdat, coil,
                                        MB=MB,
                                        acs_shape=acs_shape,
                                        lamda=0.01, max_iter=30,
                                        yshift=yshift,
                                        device=device)

        dwi_shot = sp.to_device(dwi_shot)

        _, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x], max_iter=5)

        # # shot-combined recon

        dwi_comb_llr_imag = app.HighDimensionalRecon(kdat, coil,
                                    phase_sms=sms_phase,
                                    combine_echo=True,
                                    phase_echo=dwi_shot_phase,
                                    regu='LLR',
                                    blk_shape=(1, 6, 6),
                                    blk_strides=(1, 1, 1),
                                    normalization=True,
                                    solver='ADMM', lamda=0.01, rho=0.05,
                                    max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

        dwi_comb_llr_imag = sp.to_device(dwi_comb_llr_imag)

        return dwi_comb_llr_imag, dwi_shot_phase

    else:

        dwi_shot_llr = app.HighDimensionalRecon(kdat, coil,
                                    phase_sms=sms_phase,
                                    combine_echo=False,
                                    regu='LLR',
                                    blk_shape=(1, 6, 6),
                                    blk_strides=(1, 1, 1),
                                    normalization=True,
                                    solver='ADMM', lamda=0.01, rho=0.05,
                                    max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

        dwi_shot_llr = sp.to_device(dwi_shot_llr)

        return dwi_shot_llr, None

# %% recon

# JETS
dwi_jets_fully, phs_shot_fully = _jets_recon(kdat_prep , coil1, yshift=yshift, MB=N_band,
                             sms_phase=sms_phase)

# %% save recon outputs
f = h5py.File(DIR + '/jets.h5', 'w')
f.create_dataset('DWI', data=dwi_jets_fully)
f.close()

"""
This script performs DWI reconstruction

Stepwise,

    1. shot phase estimation
    2. MUSE or JETS reconstruction

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import argparse
import h5py
import os
import torch

import numpy as np
import sigpy as sp
import torchvision.transforms as T

from sigpy.mri import retro, app, sms, muse
from os.path import exists

DIR = os.path.dirname(os.path.realpath(__file__))

# %% options
parser = argparse.ArgumentParser(description='run reconstruction.')

parser.add_argument('--data',
                    default='meas_MID00183_FID00166_Seg3Nav_0p7In_Small.dat',
                    help='raw dat file')

parser.add_argument('--slice_idx', type=int, default=-1,
                    help='recon only one slice given its index [default: -1].')

parser.add_argument('--slice_inc', type=int, default=1,
                    help='slice increments [default: 1].')

parser.add_argument('--navi', action='store_true',
                    help='use navigator to estimate phase.')

parser.add_argument('--full_fov', action='store_true',
                    help='estimate shot phase from full FOV k-space.')

parser.add_argument('--muse', action='store_true',
                    help='run MUSE recon.')

parser.add_argument('--jets', action='store_true',
                    help='run JETS recon.')

parser.add_argument('--admm_lamda', type=float, default=0.04,
                    help='lamda in ADMM')

parser.add_argument('--admm_rho', type=float, default=0.05,
                    help='rho in ADMM')

parser.add_argument('--admm_iter', type=int, default=15,
                    help='number of iterations in ADMM')

args = parser.parse_args()

print('> data: ', args.data)

device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
xp = device.xp

# %% read in raw data
instr = DIR + '/' + args.data
outprefstr = instr.split('.dat')[0]

# read in coils
coil_file = outprefstr + '/coils_reord'
print('> coil: ' + coil_file)
f = h5py.File(coil_file + '.h5', 'r')
coil = f['coil'][:]
f.close()

print('> coil shape: ', coil.shape)

N_y, N_x = coil.shape[-2:]

# read in other parameters
f = h5py.File(outprefstr + '/kdat_slice_000.h5', 'r')
MB = f['MB'][()]
N_slices = f['Slices'][()]
N_segments = f['Segments'][()]
N_Accel_PE = f['Accel_PE'][()]
f.close()

# number of collapsed slices
N_slices_collap = N_slices // MB

# SMS phase shift
yshift = []
for b in range(MB):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

# %% run reconstruction
if args.slice_idx >= 0:
    slice_loop = range(args.slice_idx, args.slice_idx + args.slice_inc, 1)
else:
    slice_loop = range(N_slices_collap)

for s in slice_loop:

    slice_str = str(s).zfill(3)
    print('> collapsed slice idx: ', slice_str)

    # read in k-space data
    f = h5py.File(outprefstr + '/kdat_slice_' + slice_str + '.h5', 'r')
    kdat = f['kdat'][:]
    f.close()

    kdat = np.squeeze(kdat)  # 4 dim
    kdat = np.swapaxes(kdat, -2, -3)

    # read in navi data
    f = h5py.File(outprefstr + '/navi_slice_' + slice_str + '.h5', 'r')
    navi = f['navi'][:]
    f.close()

    navi = np.squeeze(navi)
    navi = np.swapaxes(navi, -2, -3)
    navi = np.swapaxes(navi, 0, 1)
    navi_prep = navi[..., None, :, :]  # 6 dim

    # split kdat into shots
    N_diff = kdat.shape[-4]
    kdat_prep = []
    for d in range(N_diff):
        k = retro.split_shots(kdat[d, ...], shots=N_segments)
        kdat_prep.append(k)

    kdat_prep = np.array(kdat_prep)
    kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

    print('>> kdat_prep shape: ', kdat_prep.shape)
    N_diff, N_shot, N_coil, N_z, N_y, N_x = kdat_prep.shape

    print('>> navi_prep shape: ', navi_prep.shape)
    N_navi_y, N_navi_x = navi_prep.shape[-2:]


    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

    print('>> slice_mb_idx: ', slice_mb_idx)
    coil2 = coil[:, slice_mb_idx, :, :]

    # %% shot phase estimation based on full-fov k-space

    if args.full_fov:
        kdat_resi = kdat_prep
        navi_resi = sp.resize(navi_prep, oshape=kdat_prep.shape)
        coil_resi = coil2
        sms_phase_resi = sms_phase

        acs_shape = list(np.array([N_y, N_x]) // 4)

        FOV_STR = '-FULL'

    else:
        navi_small_fov = [N_navi_y, N_navi_x * 2]
        navi_resi = sp.resize(navi_prep,
                              oshape=list(navi_prep.shape[:-2]) +\
                                  navi_small_fov)
        kdat_resi = sp.resize(kdat_prep,
                              oshape=list(kdat_prep.shape[:-2]) +\
                                  navi_small_fov)

        coil_tensor = sp.to_pytorch(coil2)
        TR = T.Resize(navi_small_fov)
        coil_resi_r = TR(coil_tensor[..., 0]).cpu().detach().numpy()
        coil_resi_i = TR(coil_tensor[..., 1]).cpu().detach().numpy()
        coil_resi = coil_resi_r + 1j * coil_resi_i

        sms_phase_resi = sms.get_sms_phase_shift([MB] + navi_small_fov,
                                                 MB=MB, yshift=yshift)

        acs_shape = navi_small_fov

        FOV_STR = '-REDU'


    if args.navi:
        # # muse
        navi_comb_muse, navi_shot_muse = muse.MuseRecon(
                                navi_resi, coil_resi, MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.001, max_iter=60,
                                yshift=yshift,
                                device=device)

        navi_shot_muse_full, navi_shot_muse_phase = muse._denoising(navi_shot_muse, full_img_shape=[N_y, N_x])

        NAVI_STR = '_PHASE-NAVI' + FOV_STR

    else:
        # # muse
        navi_comb_muse, navi_shot_muse = muse.MuseRecon(
                                kdat_resi, coil_resi, MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.001, max_iter=60,
                                yshift=yshift,
                                device=device)

        navi_shot_muse_full, navi_shot_muse_phase = muse._denoising(navi_shot_muse, full_img_shape=[N_y, N_x])

        NAVI_STR = '_PHASE-IMAG' + FOV_STR

    # %% shot-combined recon

    OUT_STR = NAVI_STR + '_slice_' + slice_str

    # # MUSE
    if args.muse is True:

        kdat_prep_dev = sp.to_device(kdat_prep, device=device)
        coil2_dev = sp.to_device(coil2, device=device)
        navi_phase_dev = sp.to_device(np.conj(navi_shot_muse_phase), device=device)

        dwi_comb_muse = []

        for d in range(N_diff):

            k1 = kdat_prep_dev[d, ...]
            p1 = navi_phase_dev[d, ...]

            A = muse.sms_sense_linop(k1, coil2_dev, yshift, phase_echo=p1)

            R = muse.sms_sense_solve(A, k1, lamda=0.01, max_iter=30)

            dwi_comb_muse.append(sp.to_device(R))

        dwi_comb_muse = np.array(dwi_comb_muse)

        print('>>> dwi_comb_muse shape: ', dwi_comb_muse.shape)

        # store output
        f = h5py.File(outprefstr + '/MUSE' + OUT_STR + '.h5', 'w')
        f.create_dataset('nav_shot_muse', data=navi_shot_muse_full)
        f.create_dataset('nav_shot_muse_phase', data=navi_shot_muse_phase)
        f.create_dataset('dwi_comb_muse', data=dwi_comb_muse)
        f.close()

    # # JETS
    if args.jets is True:

        dwi_comb_jets = app.HighDimensionalRecon(
                                kdat_prep, coil2,
                                phase_sms=sms_phase,
                                combine_echo=True,
                                phase_echo=np.conj(navi_shot_muse_phase),
                                regu='LLR',
                                blk_shape=(1, 6, 6),
                                blk_strides=(1, 1, 1),
                                solver='ADMM',
                                lamda=args.admm_lamda,
                                rho=args.admm_rho,
                                max_iter=15,
                                show_pbar=False, verbose=True,
                                device=device).run()

        dwi_comb_jets = sp.to_device(dwi_comb_jets)

        # store output
        f = h5py.File(outprefstr + '/JETS' + OUT_STR + '.h5', 'w')
        f.create_dataset('nav_shot_muse', data=navi_shot_muse_full)
        f.create_dataset('dwi_comb_jets', data=dwi_comb_jets)
        f.close()

import argparse
import h5py
import os

import numpy as np
import sigpy as sp

from sigpy.mri import retro, app, sms, muse, mussels
from os.path import exists

DIR = os.path.dirname(os.path.realpath(__file__))

# %% options
parser = argparse.ArgumentParser(description='run reconstruction.')

parser.add_argument('--data',
                    default='meas_MID00103_FID00035_3scantrace_Seg3_0p7inplane_2Slice_Pat3.dat',
                    help='raw dat file')

parser.add_argument('--slice_idx', type=int, default=-1,
                    help='recon only one slice given its index [default: -1].')

parser.add_argument('--slice_inc', type=int, default=1,
                    help='slice increments [default: 1].')

parser.add_argument('--pat', type=int, default=3,
                    help='in-plane acceleration factor [default: 3].')

parser.add_argument('--pi', action='store_true',
                    help='run paralel imaging recon.')

parser.add_argument('--muse', action='store_true',
                    help='run MUSE recon.')

parser.add_argument('--mussels', action='store_true',
                    help='run MUSSELS recon.')

parser.add_argument('--jets', action='store_true',
                    help='run JETS recon.')

parser.add_argument('--device', type=int, default=0,
                    help='which device to run recon [default: 0]')

parser.add_argument('--split', type=int, default=1,
                    help='split diffusion encodings in recon [default: 1]')

parser.add_argument('--reduce', type=float, default=1,
                    help='reduce recon k-space size [default: 1]')

args = parser.parse_args()

print('> data: ', args.data)

device = sp.Device(args.device)
xp = device.xp

# %% read in raw data
instr = DIR + '/' + args.data
outprefstr = instr.split('.dat')[0]

# slice order
# slice_order = np.load(outprefstr + '/' + args.slice_order)
# print('> slice order: ', slice_order)

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
f.close()

# number of collapsed slices
N_slices_collap = N_slices // MB

# SMS phase shift
yshift = []
for b in range(MB):
    yshift.append(b / args.pat)

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

    # map from collapsed slice index to interleaved uncollapsed slice list
    # slice_mb_idx = sms.get_uncollap_slice_idx(N_slices, MB, s)
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)
    # slice_mb_idx = [46, 93]

    print('>> slice_mb_idx: ', slice_mb_idx)

    coil2 = coil[:, slice_mb_idx, :, :]


    # %% ###### comb-shot parallel imaging
    if args.pi is True:

        kdat_prep = sp.to_device(kdat_prep, device=device)
        coil2 = sp.to_device(coil2, device=device)
        sms_phase = sp.to_device(sms_phase, device=device)

        with device:
            kdat_comb = xp.sum(kdat_prep, axis=-5, keepdims=True)
            img_shape = coil2.shape[1:]

            S = sp.linop.Multiply(img_shape, coil2)
            F = sp.linop.FFT(S.oshape, axes=[-2, -1])
            P = sp.linop.Multiply(F.oshape, sms_phase)
            M = sp.linop.Sum(P.oshape, axes=(-3, ))

            dwi_pi_comb = []
            for d in range(N_diff):

                k = xp.squeeze(kdat_comb[d, ...])

                weights = (sp.rss(k, axes=(0, ), keepdims=True) > 0).astype(k.dtype)
                W = sp.linop.Multiply(M.oshape, weights)

                A = W * M * P * F * S

                AHA = lambda x: A.N(x) + 0.01 * x
                AHy = A.H(k)

                img = xp.zeros(img_shape, dtype=k.dtype)
                alg_method = sp.alg.ConjugateGradient(AHA, AHy, img, max_iter=30, verbose=True)

                while (not alg_method.done()):
                    alg_method.update()

                dwi_pi_comb.append(sp.to_device(img))

        dwi_pi_comb = np.array(dwi_pi_comb)

        # store output
        recon_file = '/PI_CombShots_slice_' + slice_str + '.h5'
        f = h5py.File(outprefstr + recon_file, 'w')
        f.create_dataset('DWI', data=dwi_pi_comb)
        f.close()

    # %% ###### Muse Recon
    if args.muse is True:

        acs_shape = list(np.array([N_y, N_x]) // 4)

        dwi_muse, dwi_shot = muse.MuseRecon(kdat_prep, coil2, MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.01, max_iter=30,
                                yshift=yshift,
                                device=device)

        dwi_muse = sp.to_device(dwi_muse)

        # store output
        recon_file = '/MUSE_slice_' + slice_str + '.h5'
        f = h5py.File(outprefstr + recon_file, 'w')
        f.create_dataset('DWI', data=dwi_muse)
        f.close()

    # %% ###### Mussels Recon
    if args.mussels is True:

        dwi_mussels = mussels.MusselsRecon(kdat_prep, coil2, MB=MB,
                                lamda=0.02, rho=0.05, max_iter=50,
                                yshift=yshift,
                                device=device)

        dwi_mussels = sp.to_device(dwi_mussels)

        # store output
        recon_file = '/MUSSELS_slice_' + slice_str + '.h5'
        f = h5py.File(outprefstr + recon_file, 'w')
        f.create_dataset('DWI', data=dwi_mussels)
        f.close()

    # %% ###### Shot LLR Recon
    if (args.jets is True) and (N_segments == 1):

        N_diff_split = N_diff // args.split

        for s in range(args.split):

            if args.split == 1:
                split_str = ""
            else:
                split_str = "_part_" + str(s).zfill(1)

            diff_idx = range(s * N_diff_split, (s+1) * N_diff_split if s < args.split else N_diff)

            kdat_prep_split = kdat_prep[diff_idx, ...]

            shot_llr = app.HighDimensionalRecon(kdat_prep_split, coil2,
                        combine_echo=False, phase_sms=sms_phase,
                        regu='LLR', blk_shape=(1, 6, 6), blk_strides=(1, 1, 1),
                        solver='ADMM', lamda=0.04, rho=0.05, max_iter=15,
                        show_pbar=False, verbose=True,
                        device=device).run()

            shot_llr = sp.to_device(shot_llr)

            # store output
            recon_file = '/JETS1_slice_' + slice_str + split_str + '.h5'
            f = h5py.File(outprefstr + recon_file, 'w')
            f.create_dataset('DWI', data=shot_llr)
            f.close()

    # %% ###### Shot-combined LLR Recon
    if (args.jets is True) and (N_segments > 1):

        lamda = 0.02
        rho = 0.05

        N_diff_split = N_diff // args.split

        for s in range(args.split):

            if args.split == 1:
                split_str = ""
            else:
                split_str = "_part_" + str(s).zfill(1)

            diff_idx = range(s * N_diff_split, (s+1) * N_diff_split if s < args.split else N_diff)

            kdat_prep_split = kdat_prep[diff_idx, ...]


            acs_shape = list(np.array([N_y, N_x]) // 4)
            ksp_acs = sp.resize(kdat_prep_split, oshape=list(kdat_prep_split.shape[:-2]) + acs_shape)

            import torchvision.transforms as T

            coils_tensor = sp.to_pytorch(coil2)
            TR = T.Resize(acs_shape)
            mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
            mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
            mps_acs = mps_acs_r + 1j * mps_acs_i

            sms_phase_acs = sms.get_sms_phase_shift([MB] + acs_shape, MB=MB, yshift=yshift)

            print('>> step 1. joint recon of all shots')
            USE_SHOT_LLR = False
            if USE_SHOT_LLR is True:

                dwi_shot = app.HighDimensionalRecon(ksp_acs, mps_acs,
                                    phase_sms=sms_phase_acs,
                                    combine_echo=False,
                                    regu='LLR', blk_shape=(1, 6, 6), blk_strides=(1, 1, 1),
                                    solver='ADMM', lamda=lamda, rho=rho, max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

                dwi_shot = sp.to_device(dwi_shot)

            else:

                _, dwi_shot = muse.MuseRecon(ksp_acs, mps_acs,
                                    MB=MB,
                                    acs_shape=acs_shape,
                                    lamda=0.01, max_iter=30,
                                    yshift=yshift,
                                    device=device)

            # reduce k-space size

            reduce_str = ''
            if args.reduce > 1:
                reduce_str = '_reduce_' + str(args.reduce)

            red_shape = list((np.array([N_y, N_x]) // args.reduce).astype(int))
            ksp_red = sp.resize(kdat_prep_split, oshape=list(kdat_prep_split.shape[:-2]) + red_shape)

            coils_tensor = sp.to_pytorch(coil2)
            TR = T.Resize(red_shape)
            mps_red_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
            mps_red_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
            mps_red = mps_red_r + 1j * mps_red_i

            sms_phase_red = sms.get_sms_phase_shift([MB] + red_shape, MB=MB, yshift=yshift)

            print('>> step 2. phase denoising')
            _, dwi_show_phase = muse._denoising(dwi_shot, full_img_shape=red_shape)

            print('>> step 3. shot-combined joint recon')
            dwi_llr_comb = app.HighDimensionalRecon(ksp_red, mps_red,
                                phase_sms=sms_phase_red,
                                combine_echo=True, phase_echo=dwi_show_phase,
                                regu='LLR', blk_shape=(1, 6, 6), blk_strides=(1, 1, 1),
                                solver='ADMM', lamda=lamda, rho=rho, max_iter=15,
                                show_pbar=False, verbose=True,
                                device=device).run()

            dwi_llr_comb = sp.to_device(dwi_llr_comb)

            # store output
            recon_file = '/JETS2_slice_' + slice_str + split_str + reduce_str + '.h5'
            f = h5py.File(outprefstr + recon_file, 'w')
            f.create_dataset('DWI', data=dwi_llr_comb)
            f.create_dataset('DWI_Shots', data=dwi_shot)
            f.create_dataset('PHS_Shots', data=dwi_show_phase)
            f.close()

"""
This script converts .dat to .h5 files

The data contains no navigator

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import argparse
import h5py
import os
import pathlib
import torch
import twixtools

import numpy as np
import sigpy as sp

from sigpy.mri import app, epi, sms, cc

DIR = os.path.dirname(os.path.realpath(__file__))
print(DIR)

# %% argument parser
parser = argparse.ArgumentParser(description='prepare data and store output k-space data separately.')

parser.add_argument('--data',
                    default='meas_MID00399_FID02548_EPISSh_67_1p0iso_kyShift.dat',
                    help='raw dat file.')

args = parser.parse_args()


if torch.cuda.is_available():
    device = sp.Device(0)
else:
    device = sp.cpu_device

print('> device: ', device)

# %% prepare output string
print('>>> dat: ', args.data)

instr = DIR + '/' + args.data
outprefstr = instr.split('.dat')[0]

# make a new directory if not exist
pathlib.Path(outprefstr).mkdir(parents=True, exist_ok=True)

# %% read in twix data
twixobj = twixtools.read_twix(instr)

twix = twixobj[-1]

# %% use hdr
hdr_twix = twixobj[-1]['hdr']

MB = hdr_twix['Phoenix']['sSliceAcceleration']['lMultiBandFactor']
N_slices = hdr_twix['Phoenix']['sSliceArray']['lSize']
N_segments = hdr_twix['Phoenix']['sFastImaging']['lSegments']
N_EchoTrainLength = int(hdr_twix['Meas']['EchoTrainLength'])
N_Accel_PE = hdr_twix['Phoenix']['sPat']['lAccelFactPE']

print('> multi-band ' + str(MB) +\
      ', slices ' + str(N_slices) +\
      ', segments ' + str(N_segments) +\
      ', echo train length ' + str(N_EchoTrainLength) +\
      ', iPat ' + str(N_Accel_PE))

REMOVE_OS = True
if REMOVE_OS is True:
    os = 2
else:
    os = 1


mapped = twixtools.map_twix(instr)


# kdat twix
kdat_twix = mapped[-1]['image']

kdat_twix.flags['regrid'] = True
kdat_twix.flags['remove_os'] = REMOVE_OS
kdat_twix.flags['zf_missing_lines'] = True
kdat_twix.flags['average']['Seg'] = False

# %% phase-correction and reference scan data

# phase-correction
pcor_twix = mapped[-1]['phasecorr']
pcor_twix.flags['regrid'] = True
pcor_twix.flags['remove_os'] = REMOVE_OS
pcor_twix.flags['skip_empty_lead'] = True
pcor_twix.flags['average']['Seg'] = False

# refscan
if N_Accel_PE > 1:
    refs_twix = mapped[-1]['refscan']
    refs_twix.flags['regrid'] = True
    refs_twix.flags['remove_os'] = REMOVE_OS
    refs_twix.flags['skip_empty_lead'] = True
    refs_twix.flags['average']['Seg'] = False

# %% data shape
N_diff = kdat_twix.shape[-9]  # Rep
N_y = kdat_twix.shape[-3]     # Lin
N_x = kdat_twix.shape[-1]     # Col
N_coil = kdat_twix.shape[-2]  # Cha

# %% read out data
# kdat = kdat_twix[:]
print('> kdat shape: ', kdat_twix.shape)

pcor = pcor_twix[:]
print('> pcor shape: ', pcor.shape)

if N_Accel_PE > 1:
    refs = refs_twix[:]
    print('> refs shape: ', refs.shape)


# %% coil sensitivity maps
if N_Accel_PE > 1:

    if False:
        N_virt = 12
        refs_cc, S = cc.cc_huang(refs, P=N_virt, coil_dim=-2, device=device)
    else:
        N_virt = N_coil
        refs_cc = refs.copy()

    # zero-fill refs
    refs_cc_shape = list(refs_cc.shape)
    refs_zf = sp.resize(refs_cc, refs_cc_shape[:-3] + [N_y, N_virt, N_x])

    # reshape
    refs_prep = np.squeeze(refs_zf)
    refs_prep = np.swapaxes(refs_prep, -2, -3)
    refs_prep = np.swapaxes(refs_prep, -3, -4)

    f = h5py.File(outprefstr + '/refs.h5', 'w')
    f.create_dataset('refs', data=refs_prep)
    f.close()


    print('> estimate coil sensitivity maps: ')

    mps = []
    for s in range(N_slices):
        print('  ' + str(s).zfill(3))

        c = app.EspiritCalib(refs_prep[:, s, :, :],
                             crop=0.,
                             device=device, show_pbar=False).run()
        mps.append(sp.to_device(c))

    mps = np.array(mps)
    mps = np.swapaxes(mps, 0, 1)

    # coil sensitivity maps are acquired in the single-slice mode
    mps_reord = sms.reorder_slices_mb1(mps, N_slices)

    f = h5py.File(outprefstr + '/coils_reord.h5', 'w')
    f.create_dataset('coil', data=mps_reord)
    f.close()


# %% extract useful kdat
N_slices_collap = N_slices // MB

if (N_slices_collap % 2) and (MB % 2) and (N_slices % 2):
    N_offset_2 = 1
    N_offset_1 = 0
else:
    N_offset_2 = 0
    N_offset_1 = 1

if (N_slices_collap % 2 == 0) and (MB % 2 == 1) and (N_slices % 2 == 0):
    N_offset_2 = 0
    N_offset_1 = 0

if (MB % 2 == 0) and (N_slices % 2 == 0) and (N_slices_collap % 2 == 0): # even
    N_offset_2 = 0
    N_offset_1 = 0

N_slices_half = N_slices_collap // 2 + N_offset_2  # interleaved slice mode

N_slices_dat = kdat_twix.shape[-5]
print('> N_slices_dat ', N_slices_dat)

sms_idx = list(range(0, N_slices_half, 1)) \
        + list(range(N_slices_dat - N_slices_half - N_offset_1 + N_offset_2, N_slices_dat, 1))

print('> extracted multi-band slices indices:')

for id in range(len(sms_idx)):

    sid = sms_idx[id]

    print('  ' + str(id).zfill(3) + ' ' + str(sid).zfill(3))

    # correcting kdat
    k = kdat_twix[..., [sid], :, :, :, :]
    p = pcor[..., [sid], :, :, :, :]

    kcor = epi.phase_corr(k, p)

    f = h5py.File(outprefstr + '/kdat_slice_' + str(id).zfill(3) + '.h5', 'w')
    f.create_dataset('kdat', data=kcor)
    f.create_dataset('MB', data=MB)
    f.create_dataset('Slices', data=N_slices)
    f.create_dataset('Segments', data=N_segments)
    f.create_dataset('Accel_PE', data=N_Accel_PE)
    f.create_dataset('slice_idx', data=id)
    f.close()

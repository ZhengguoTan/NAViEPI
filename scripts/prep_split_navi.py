"""
This script converts .dat (Siemens twix format) to .h5 files.

The data involves navigator acquisition.

Stepwise,

    1. use 'twixtools' to read in .dat file;
    2. split Imaging echo and Navigator echo;
    3. store data slice by slice;
    4. estimate coil sensitivity maps using Espirit in SigPy

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

from sigpy.mri import app, epi, sms

DIR = os.path.dirname(os.path.realpath(__file__))
print(DIR)

# %% argument parser
parser = argparse.ArgumentParser(description='prepare data and store output k-space data separately.')

parser.add_argument('--data',
                    default='meas_MID00062_FID00601_Seg5Nav_0p5in_plane_3Scan.dat',
                    help='raw dat file.')

parser.add_argument('--keep_os', action='store_true',
                    help='keep readout oversampling in the raw data.')

args = parser.parse_args()


if torch.cuda.is_available():
    device = sp.Device(0)
else:
    device = sp.cpu_device

print('> device: ', device)

# %% special function:
#    retrospectively split Imaging echo and Navigator echo
def set_navi_shot_ind(twixobj, verbose=False):

    hdr_twix = twixobj[-1]['hdr']

    MB = hdr_twix['Phoenix']['sSliceAcceleration']['lMultiBandFactor']
    N_slices = hdr_twix['Phoenix']['sSliceArray']['lSize']
    N_segments = hdr_twix['Phoenix']['sFastImaging']['lSegments']

    if verbose is True:
        print('> multi-band ' + str(MB) + ', slices ' + str(N_slices) + ', segments ' + str(N_segments))

    cnt_lines = 0
    cnt_slices = 1
    cnt_diff = 1
    cnt_shots = 1

    total_lines = 0

    for mdb in twixobj[-1]['mdb']:

        if mdb.is_image_scan() and mdb.cSet == 1:

            # change counter to distinguish mdb
            mdb.mdh.Counter.Ida = cnt_shots - 1

            cnt_lines = cnt_lines + 1

            total_lines = total_lines + 1

            if verbose is True:
                print('>>> line ' + str(cnt_lines).zfill(2) +\
                         ' slice ' + str(cnt_slices).zfill(3) +\
                         ' shot ' + str(cnt_shots) +\
                         ' diff ' + str(cnt_diff))

            if mdb.is_flag_set('LASTSCANINSLICE'):  # 1st loop: lines
                cnt_lines = 0
                cnt_slices = cnt_slices + 1

        if cnt_slices > N_slices // MB:    # 2nd loop: slices
            cnt_slices = 1
            cnt_shots = cnt_shots + 1

        if cnt_shots > N_segments:        # 3rd loop: diff
            cnt_shots = 1
            cnt_diff = cnt_diff + 1

    if verbose is True:
        print('> total lines: ', total_lines)

    return twixobj

# %% prepare output string
print('>>> dat: ', args.data)

instr = DIR + '/' + args.data
outprefstr = instr.split('.dat')[0]

# make a new directory if not exist
pathlib.Path(outprefstr).mkdir(parents=True, exist_ok=True)

# %% read in twix data
twixobj = twixtools.read_twix(instr)
twixobj = set_navi_shot_ind(twixobj)

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

REMOVE_OS = not (args.keep_os)
if REMOVE_OS is True:
    os = 2
else:
    os = 1

# %% split sets
twixobj_kdat = {'hdr': twix['hdr'], 'hdr_str': twix['hdr_str'], 'mdb': []}
twixobj_navi = {'hdr': twix['hdr'], 'hdr_str': twix['hdr_str'], 'mdb': []}

for mdb in twix['mdb']:
    if mdb.is_image_scan():
        if mdb.cSet == 0:
            twixobj_kdat['mdb'].append(mdb)
        elif mdb.cSet == 1:
            twixobj_navi['mdb'].append(mdb)

map_kdat = twixtools.map_twix(twixobj_kdat)
map_navi = twixtools.map_twix(twixobj_navi)


# kdat twix
kdat_twix = map_kdat['image']

kdat_twix.flags['regrid'] = True
kdat_twix.flags['remove_os'] = REMOVE_OS
kdat_twix.flags['zf_missing_lines'] = True
kdat_twix.flags['average']['Seg'] = False


# navi twix
navi_twix = map_navi['image']

navi_twix.flags['regrid'] = True
navi_twix.flags['remove_os'] = REMOVE_OS
navi_twix.flags['average']['Ida'] = False
navi_twix.flags['average']['Seg'] = False
navi_twix.flags['average']['Set'] = True

# %% phase-correction and reference scan data
mapped = twixtools.map_twix(instr)

# phase-correction
pcor_twix = mapped[-1]['phasecorr']
pcor_twix.flags['regrid'] = True
pcor_twix.flags['remove_os'] = REMOVE_OS
pcor_twix.flags['skip_empty_lead'] = True
pcor_twix.flags['average']['Seg'] = False

# refscan
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
# print('> kdat shape: ', kdat.shape)

navi = navi_twix[:]
print('> navi shape: ', navi.shape)

pcor = pcor_twix[:]
print('> pcor shape: ', pcor.shape)

refs = refs_twix[:]
print('> refs shape: ', refs.shape)


# %% zero-fill navi
navi_samp_shape = navi.shape

navi_samp_y = navi_samp_shape[-3]
navi_samp_x = navi_samp_shape[-1]

navi_diff_y = navi_samp_x * os - navi_samp_y

navi_zero_shape = list(navi_samp_shape[:-3]) +\
                    [navi_diff_y] +\
                    list(navi_samp_shape[-2:])

navi_zero = np.zeros_like(navi, shape=navi_zero_shape)

navi_fill = np.concatenate((navi_zero, navi), axis=-3)
print('> navi zero-fill shape: ', navi_fill.shape)

# %% extract useful (non-zero) kdat
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
    f.close()


    # correcting navi
    n = navi_fill[..., [sid], :, :, :, :]

    if False:
        ncor = np.sum(n, axis=-11, keepdims=True)
    else:
        p1 = sp.resize(p, oshape=list(p.shape[:-1]) + [n.shape[-1]])
        ncor = epi.phase_corr(n, p1)

    f = h5py.File(outprefstr + '/navi_slice_' + str(id).zfill(3) + '.h5', 'w')
    f.create_dataset('navi', data=ncor)
    f.create_dataset('MB', data=MB)
    f.create_dataset('Slices', data=N_slices)
    f.create_dataset('Segments', data=N_segments)
    f.close()


# %% coil sensitivity maps
# zero-fill refs
refs_shape = list(refs.shape)
refs_zf = sp.resize(refs, refs_shape[:-3] + [N_y, N_coil, N_x])

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
                         crop=0.98,
                         device=device, show_pbar=False).run()
    mps.append(sp.to_device(c))

mps = np.array(mps)
mps = np.swapaxes(mps, 0, 1)

# coil sensitivity maps are acquired in the single-band mode
mps_reord = sms.reorder_slices_mb1(mps, N_slices)

f = h5py.File(outprefstr + '/coils_reord.h5', 'w')
f.create_dataset('coil', data=mps_reord)
f.close()

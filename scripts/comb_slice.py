import argparse
import h5py
import os

import numpy as np
import sigpy as sp

from sigpy.mri import sms

DIR = os.path.dirname(os.path.realpath(__file__))

# %% options
parser = argparse.ArgumentParser(description='Read in slice files, append them, and save in correct order.')

parser.add_argument('--dir', default='meas_MID00058_FID01423_SSH_1p0Iso',
                    help='directory in which the data are read.')

parser.add_argument('--method', default='MUSE',
                    help='recon method.')

parser.add_argument('--key', default='dwi_shot_jets',
                    help='prefix of the file name to look for.')

parser.add_argument('--MB', type=int, default=3,
                    help='multi-band factor.')

parser.add_argument('--slices', type=int, default=38,
                    help='total number of slices.')

parser.add_argument('--parts', type=int, default=0,
                    help='total number of parts per slice.')

parser.add_argument('--shot_rss', action='store_true',
                    help='perform rss shot combination.')

parser.add_argument('--shot_axis', type=int, default=-5,
                    help='total number of parts per slice.')

args = parser.parse_args()

# %%
acq_slice = []

for s in range(args.slices):

    print('> slice ' + str(s).zfill(3))

    fstr = DIR + '/' + args.dir + '/' + args.method + '_slice_' + str(s).zfill(3)

    if args.parts == 0:

        f = h5py.File(fstr + '.h5', 'r')
        acq_slice.append(f[args.key][:])
        f.close()

    else:

        acq_part = []
        for p in range(args.parts):

            fpstr = fstr + '_part_' + str(p)
            f = h5py.File(fpstr + '.h5', 'r')
            acq_part.append(f[args.key][:])
            f.close()

        acq_part = np.array(acq_part)
        acq_part = np.reshape(acq_part, [-1] + list(acq_part.shape[2:]))

        print('  shape: ', acq_part.shape)

        acq_slice.append(acq_part)


acq_slice = np.array(acq_slice)
print('> acq_slice shape: ', acq_slice.shape)

if args.shot_rss:
    acq_slice = sp.rss(acq_slice, axes=(-5,))

# total number of slices
N_slices = args.slices * args.MB

# old:
# reo_slice = sms.reorder_slices(acq_slice, args.MB, N_slices)
reo_slice = sms.reorder_slices_mbx(acq_slice, args.MB, N_slices)
reo_slice = np.squeeze(reo_slice)

f = h5py.File(DIR + '/' + args.dir + '/' + args.method + '.h5', 'w')
f.create_dataset('DWI', data=reo_slice)
f.close()

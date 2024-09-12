import argparse
import h5py
import nibabel as nib
import numpy as np
import os

DIR = os.path.dirname(os.path.realpath(__file__))
print(DIR)

# %%
parser = argparse.ArgumentParser(description='convert .h5 to .nii')

parser.add_argument('--input_data',
                    default='meas_MID00402_FID02551_EPISeg2_67_1p0iso_kyShift',
                    help='input data folder')

parser.add_argument('--input_file',
                    default='MUSE',
                    help='input .h5 file name')

parser.add_argument('--input_key',
                    default='DWI',
                    help='input .h5 file key')

parser.add_argument('--output_prefix',
                    default='MUSE',
                    help='output .nii file prefix')

args = parser.parse_args()

# %%
DAT_DIR = DIR + '/' + args.input_data

f = h5py.File(DAT_DIR + '/' + args.input_file + '.h5', 'r')
DWI = f[args.input_key][:]
f.close()

DWI = np.squeeze(DWI).T

print('DWI shape: ', DWI.shape)

DWI_abs = np.abs(DWI)
DWI_phs = np.angle(DWI)

img_abs = nib.Nifti1Image(DWI_abs, affine=np.eye(DWI.ndim))
nib.save(img_abs, DAT_DIR + '/' + args.output_prefix + '_abs.nii')

img_phs = nib.Nifti1Image(DWI_phs, affine=np.eye(DWI.ndim))
nib.save(img_phs, DAT_DIR + '/' + args.output_prefix + '_phs.nii')

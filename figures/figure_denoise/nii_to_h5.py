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
                    default='MUSE_cplx_denoise',
                    help='input .nii file name')

args = parser.parse_args()

# %%
DAT_DIR = DIR + '/' + args.input_data

img = nib.load(DAT_DIR + '/' + args.input_file + '.nii')
DWI = np.asanyarray(img.dataobj)
print('DWI shape: ', DWI.shape)

f = h5py.File(DAT_DIR + '/' + args.input_file + '.h5', 'w')
f.create_dataset('DWI', data=DWI)
f.close()

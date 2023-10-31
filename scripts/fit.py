import os
import h5py

import nibabel as nib
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

# %% b-values and vectors
f = h5py.File(DIR + '/1shell_30dir_diff_encoding.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]

# bvals = bvals.reshape(-1, 1)
# B = epi.get_B(bvals, bvecs)

print(bvals.shape)
print(bvecs.shape)

from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, atol=0.1)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
tenmodel = dti.TensorModel(gtab)


# %% 2. loop over all source files to fit MD, FA, and RGB
src_files = ['JETS']

for cnt in range(len(src_files)):

    src_file = DIR + '/' + src_files[cnt]
    print('> src: ' + src_file)

    f = h5py.File(src_file + '.h5', 'r')
    dwi = f['DWI'][:]
    f.close()

    dwi = abs(np.squeeze(dwi)).T * 1000
    print(dwi.shape)

    N_x, N_y, N_z, N_diff = dwi.shape


    b0 = np.mean(abs(dwi), axis=-1)
    id = b0 > np.amax(b0) * 0.01
    mask = np.zeros_like(b0)
    mask[id] = 1
    # b0_mask, mask = median_otsu(b0,
    #                             median_radius=4,
    #                             numpass=4)

    b1 = np.mean(abs(dwi[..., 1:]), axis=-1)


    tenfit = tenmodel.fit(dwi)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = np.squeeze(color_fa(FA, tenfit.evecs))
    MD = tenfit.md

    FA  = ((mask.T) * (FA.T)).T
    RGB = ((mask.T) * (RGB.T)).T
    MD  = ((mask.T) * (MD.T)).T

    f = h5py.File(src_file + '_dti_fit.h5', 'w')
    f.create_dataset('FA', data=FA)
    f.create_dataset('cFA', data=RGB)
    f.create_dataset('MD', data=MD)
    f.create_dataset('mean_dwi', data=b1)
    f.create_dataset('mask', data=mask)
    f.close()

"""
This script performs multi-band SENSE reconstruction
for single-shot EPI acquisition.

Authors:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import h5py
import os
import torch

import numpy as np
import sigpy as sp

from sigpy.mri import app, sms
from pathlib import Path

# %%
def MbSenseRecon(kdat, coil, sms_phase,
                 regu=0.001, max_iter=30,
                 device=sp.cpu_device):

    xp = device.xp

    output_devie = sp.get_device(kdat)

    kdat_device = sp.to_device(kdat, device=device)
    coil_device = sp.to_device(coil, device=device)
    sms_phase_device = sp.to_device(sms_phase, device=device)

    with device:

        N_coil, N_z, N_y, N_x = coil_device.shape

        img_shape = [1, N_z, N_y, N_x]

        # 1. coils
        S = sp.linop.Multiply(img_shape, coil_device)
        # 2. FFT
        F = sp.linop.FFT(S.oshape, axes=[-2, -1])
        # 4. SMS
        PHI = sp.linop.Multiply(F.oshape, sms_phase_device)
        SUM = sp.linop.Sum(PHI.oshape, axes=(-3,), keepdims=True)
        M = SUM * PHI
        # 5. sampling mask
        weights = app._estimate_weights(kdat_device, None, None, coil_dim=-4)
        W = sp.linop.Multiply(M.oshape, weights**0.5)

        # chain
        A = W * M * F * S

        print('  MR Forward Operator Input Shape: ', A.ishape)
        print('  MR Forward Operator Output Shape: ', A.oshape)

        # # Linear Least Square Solve
        AHA = lambda x: A.N(x) + regu * x
        AHy = A.H(kdat_device)

        img = xp.zeros(A.ishape, dtype=kdat_device.dtype)
        alg_method = sp.alg.ConjugateGradient(AHA, AHy, img,
                                max_iter=max_iter, verbose=True)

        while (not alg_method.done()):
            alg_method.update()

        return sp.to_device(img, device=output_devie)

# %%
def _assert_shape(shape1, shape2):

    assert len(shape1) == len(shape2)

    for i in range(len(shape1)):
        assert shape1[i] == shape2[i]

# %%
if __name__ == "__main__":
    print('> run the multi-band SENSE recon on a single-shot EPI data')

    # directory
    DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = Path(DIR).parents[0]
    DATA_DIR = BASE_DIR / 'data'
    print('> data directory: ', DATA_DIR)

    # device
    device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
    print('> run on device: ', device)

    # k-space imaging echo
    f = h5py.File(DATA_DIR / '1.7x1.7x4.0mm_kdat.h5', 'r')
    kdat = f['kdat'][:]
    MB = f['MB'][()]
    N_Accel_PE = f['Accel_PE'][()]
    f.close()

    kdat = np.squeeze(kdat)  # 4 dim
    kdat = np.swapaxes(kdat, -2, -3)

    N_diff, N_coil, N_y, N_x = kdat.shape
    kdat = kdat[:, None, :, None, :, :]  # 6dim

    print('> kdat shape: ', kdat.shape)

    # coil sensitivity maps
    f = h5py.File(DATA_DIR / '1.7x1.7x4.0mm_refs.h5', 'r')
    refs = f['refs'][:]
    f.close()

    N_slices = refs.shape[1]
    _assert_shape(kdat.shape[-2:], refs.shape[-2:])

    mps = []
    for s in range(N_slices):
        print('  ' + str(s).zfill(3))

        c = app.EspiritCalib(refs[:, s, :, :], thresh=0.03,
                            crop=0.95,
                            device=device, show_pbar=False).run()
        mps.append(sp.to_device(c))

    mps = np.array(mps)
    mps = mps[[0, 2, 1], ...]  # slice reordering
    mps = np.swapaxes(mps, 0, 1)

    f = h5py.File(DATA_DIR / '1.7x1.7x4.0mm_coil.h5', 'w')
    f.create_dataset('C', data=mps)
    f.close()

    print('> mps shape: ', mps.shape)

    # multi-band phase shift
    yshift = []
    for b in range(MB):
        yshift.append(b / N_Accel_PE)

    sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB,
                                        yshift=yshift)

    R = []
    for d in range(N_diff):
        img = MbSenseRecon(kdat[d], mps, sms_phase, device=device)
        R.append(img)

    f = h5py.File(Path(DIR) / '1.7x1.7x4.0mm_MbSense.h5', 'w')
    f.create_dataset('R', data=R)
    f.close()

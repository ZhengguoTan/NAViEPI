#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate mrtrix

if [ -z "$1" ]; then
    echo 'DATA is not specified.'
    DATA='meas_MID00402_FID02551_EPISeg2_67_1p0iso_kyShift'
else
    DATA=$1
fi

echo $DATA


if [ -z "$2" ]; then
    echo 'FILE is not specified.'
    FILE='MUSE'
else
    FILE=$2
fi

echo $FILE

python h5_to_nii.py --input_data ${DATA} --input_file ${FILE} --input_key DWI --output_prefix ${FILE}

cd $DATA
pwd

mrcalc ${FILE}_abs.nii ${FILE}_phs.nii -force -polar ${FILE}_cplx.nii
dwidenoise ${FILE}_cplx.nii ${FILE}_cplx_denoise.nii -force -noise noise.nii --estimator Exp1

cd ..
pwd

python nii_to_h5.py --input_data ${DATA} --input_file ${FILE}_cplx_denoise
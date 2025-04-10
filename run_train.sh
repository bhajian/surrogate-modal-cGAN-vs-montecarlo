#!/bin/bash
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2_p310
python train_cgan.py

# conda activate pytorch
nohup python train.py >tmp2 &
nohup python collect_data.py >tmp1 &

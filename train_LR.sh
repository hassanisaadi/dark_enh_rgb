#!/bin/bash

model_name="LR"

if [ $# -eq 0 ]
then
   log_dir="./logs/dual-"$model_name"-"`date "+%Y%m%d_%H%M%S/"`
   echo "Everything run from scratch."
   if [ ! -d "$log_dir" ]; then
      mkdir -p $log_dir
   else
      echo "log path exists. Please change it."
      exit 1
   fi
else
   log_dir=$1
   printf "continue running from %s\n" $1
   if [ ! -d "$1" ]; then
      echo "Log path does not exist."
      exit 1
   fi
fi

echo "Starting run at: `date`"

#tensorboard --logdir=$logdir"tb" --port=8008 &

./main.py\
  --epoch 10\
  --batch_size 64\
  --lr 0.001\
  --nPatchNum 64\
  --gpuid 0\
  --lmbd_lum 0.5\
  --lmbd_ycrcb 0.5\
  --num_layers 32\
  --feature_map 64\
  --eval_every_ep 1\
  --is_single 0\
  --gamma_en 1\
  --phase train\
  --log_dir $log_dir"tb"\
  --checkpoint_dir $log_dir"checkpoint"\
  --sample_dir $log_dir"eval_results"\
  --logfile_path $log_dir"log.txt"\
  --model_name $model_name\
  --hdf5_file ./data/data_lumchr_LR_da0_p33_s24_par64_shft1_tr517586.hdf5


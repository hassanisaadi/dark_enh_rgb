#!/bin/bash

model_name="single"

if [ $# -eq 0 ]
then
   log_dir="./logs/"$model_name"-"`date "+%Y%m%d_%H%M%S/"`
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
  --epoch 25\
  --batch_size 16\
  --lr 0.001\
  --use_gpu 1\
  --nPatchNum 32\
  --gpuid 0\
  --eval_every_ep 1\
  --phase train\
  --log_dir $log_dir"tb"\
  --checkpoint_dir $log_dir"checkpoint"\
  --sample_dir $log_dir"eval_results"\
  --eval_path ./data/eval\
  --logfile_path $log_dir"log.txt"\
  --model_name $model_name\
  --hdf5_file ./data/data_lumchr_da0_p64_s64_par32_tr72676.hdf5


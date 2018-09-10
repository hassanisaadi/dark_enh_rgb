#! /usr/bin/env python2

import argparse
from glob import glob

import tensorflow as tf

from model import imdualenh
from utils import *
import sys
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch',          dest='epoch',       type=int,   default=50,     help='# of epoch')
parser.add_argument('--batch_size',     dest='batch_size',  type=int,   default=64,    help='# images in batch')
parser.add_argument('--lr',             dest='lr',          type=float, default=0.001,  help='initial learning rate for adam')
parser.add_argument('--nPatchNum',      dest='PARALLAX',    type=int,   default=64 ,    help='# of neighbor patches in right image')
parser.add_argument('--gpuid',          dest='gpuid',       type=int,   default=1,      help='GPU id: 0/1')
parser.add_argument('--lmbd_lum',       dest='lmbd_lum',    type=float, default=0.5,   help='Lambda for luminance loss')
parser.add_argument('--lmbd_ycrcb',     dest='lmbd_ycrcb',  type=float, default=0.5,   help='Lambda for Y, Cr, Cb loss')
parser.add_argument('--num_layers',     dest='num_layers',  type=int,   default=32,     help='Number of layers in lum, chr networks')
parser.add_argument('--feature_map',    dest='feature_map', type=int,   default=64,     help='Feature map size in conv2d layers')
parser.add_argument('--eval_every_ep',  dest='eval_vryep',  type=int,                   help='evaluation every epoch.')
parser.add_argument('--is_single',      dest='is_single',   type=int,   default=0,      help='using one camera(1) or two(0)?')
parser.add_argument('--gamma_en',       dest='gamma_en',    type=int,   default=0,      help='gamma correction is enabled(1) or not(0)')
parser.add_argument('--phase',          dest='phase',       default='train',            help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir',    default='./checkpoint',     help='models are saved here')
parser.add_argument('--sample_dir',     dest='sample_dir',  default='./sample',         help='evaluation results are saved here')
parser.add_argument('--log_dir',        dest='log_dir',     default='./logs',           help='log dir for tensorboard')
parser.add_argument('--logfile_path',   dest='logfile_path',default='./log.txt',   help='log file path')
parser.add_argument('--model_name',	dest='model',	    default='dualenh',		help='model_name: dual | single')
parser.add_argument('--hdf5_file',      dest='hdf5_file',                               help='data for training')
parser.add_argument('--test_dir',       dest='test_dir',    default='./test',           help='test sample are saved here')
#parser.add_argument('--test_set',       dest='test_set',    default='BSD68',            help='dataset for testing')


args = parser.parse_args()
if args.phase == 'train':
    logfile = open(args.logfile_path, "w")
    vv = vars(args)
    for i in vv:
        print(i, vv[i])
        logfile.write(str(i) + "\t" + str(vv[i]) + "\n")
    logfile.flush()
    
    is_evalp = 'evalP' in args.hdf5_file
else:
    logfile = open("./data/test/log.txt","w")
    is_evalp = False

def model_train(model_name, lr):
    with load_data(filepath=args.hdf5_file) as data:
        model_name.train(data           = data,
                       batch_size       = args.batch_size,
                       ckpt_dir         = args.ckpt_dir, 
                       sample_dir       = args.sample_dir,
                       log_dir          = args.log_dir,
                       epoch            = args.epoch,
                       lr               = lr,
                       eval_every_epoch = args.eval_vryep)

def model_test(model_name):
    test_files_X_left  = sorted(glob(args.test_dir + '/X_left/*.png'))
    test_files_X_right = sorted(glob(args.test_dir + '/X_right/*.png'))
    test_files_Y_left  = sorted(glob(args.test_dir + '/Y_left_gt/*.png'))
    model_name.test(test_files_X_left, test_files_X_right, test_files_Y_left, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir+'/Y_left')


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpuid

    lr = args.lr * np.ones([args.epoch])
    lr[3:] = lr[0] / 10.0   ###!!!
    lr[6:] = lr[0] / 100.0
    # added to control the gpu memory
    print("GPU\n")
    sys.stdout.flush()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = imdualenh(sess, batch_size=args.batch_size, PARALLAX=args.PARALLAX, model=args.model, logfile=logfile,
                          lmbd1=args.lmbd_lum, lmbd2=args.lmbd_ycrcb,
                          L=args.num_layers, fm=args.feature_map, is_evalp=is_evalp, is_single=args.is_single,
                          gamma_en=args.gamma_en)
        if args.phase == 'train':
            model_train(model, lr=lr)
        elif args.phase == 'test':
            model_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
    logfile.close()
if __name__ == '__main__':
    tf.app.run()

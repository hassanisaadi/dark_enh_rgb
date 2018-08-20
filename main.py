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
parser.add_argument('--batch_size',     dest='batch_size',  type=int,   default=128,    help='# images in batch')
parser.add_argument('--lr',             dest='lr',          type=float, default=0.001,  help='initial learning rate for adam')
parser.add_argument('--nPatchNum',      dest='PARALLAX',    type=int,   default=65 ,    help='# of neighbor patches in right image')
parser.add_argument('--gpuid',          dest='gpuid',       type=int,   default=1,      help='GPU id: 0/1')
parser.add_argument('--lmbd_lum',       dest='lmbd_lum',    type=float, default=0.33,   help='Lambda for luminance loss')
parser.add_argument('--lmbd_ycrcb',     dest='lmbd_ycrcb',  type=float, default=0.33,   help='Lambda for Y, Cr, Cb loss')
parser.add_argument('--lmbd_vgg',       dest='lmbd_vgg',    type=float, default=0.33,   help='Lambda for VGG loss')
parser.add_argument('--num_layers',     dest='num_layers',  type=int,   default=32,     help='Number of layers in lum, chr networks')
parser.add_argument('--feature_map',    dest='feature_map', type=int,   default=64,     help='Feature map size in conv2d layers')
parser.add_argument('--eval_every_ep',  dest='eval_vryep',  type=int,                   help='evaluation every epoch.')
parser.add_argument('--phase',          dest='phase',       default='train',            help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir',    default='./checkpoint',     help='models are saved here')
parser.add_argument('--sample_dir',     dest='sample_dir',  default='./sample',         help='evaluation results are saved here')
parser.add_argument('--log_dir',        dest='log_dir',     default='./logs',           help='log dir for tensorboard')
parser.add_argument('--eval_path',      dest='eval_path',   default='./data/eval',      help='data for eval in training')
parser.add_argument('--logfile_path',   dest='logfile_path',default='./logs/log.txt',   help='log file path')
parser.add_argument('--model_name',	dest='model',	    default='dualenh',		help='model_name: dual | single')
parser.add_argument('--hdf5_file',      dest='hdf5_file',                               help='data for training')
#parser.add_argument('--test_dir',       dest='test_dir',    default='./test',           help='test sample are saved here')
#parser.add_argument('--test_set',       dest='test_set',    default='BSD68',            help='dataset for testing')
args = parser.parse_args()

def model_train(model_name, lr):
    with load_data(filepath=args.hdf5_file) as data:
        eval_files_YL = sorted(glob('{}/Y*.png'.format(args.eval_path)))
        eval_files_XL = sorted(glob('{}/X*_0.png'.format(args.eval_path)))
        eval_files_XR = sorted(glob('{}/X*_1.png'.format(args.eval_path)))
        eval_data_YL = load_images(eval_files_YL) # list of array of different size, 4-D, pixel value range is 0-255
        eval_data_XL = load_images(eval_files_XL)
        eval_data_XR = load_images(eval_files_XR) 
        model_name.train(data           = data,
                       eval_data_YL     = eval_data_YL,
                       eval_data_XL     = eval_data_XL,
                       eval_data_XR     = eval_data_XR,
                       batch_size       = args.batch_size,
                       ckpt_dir         = args.ckpt_dir, 
                       sample_dir       = args.sample_dir,
                       log_dir          = args.log_dir,
                       epoch            = args.epoch,
                       lr               = lr,
                       eval_every_epoch = args.eval_vryep)

#def model_test(model_name):
#    test_files = glob('./data/test/{}/*.png'.format(args.test_set))
#    model_name.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)


def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    #if not os.path.exists(args.test_dir):
    #    os.makedirs(args.test_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpuid

    lr = args.lr * np.ones([args.epoch])
    lr[5:] = lr[0] / 10.0   ###!!!
    lr[9:] = lr[0] / 100.0
    # added to control the gpu memory
    print("GPU\n")
    sys.stdout.flush()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = imdualenh(sess, batch_size=args.batch_size, PARALLAX=args.PARALLAX, model=args.model, logfile=args.logfile_path,
                          lmbd1=args.lmbd_lum, lmbd2=args.lmbd_ycrcb, lmbd3=args.lmbd_vgg,
                          L=args.num_layers, fm=args.feature_map)
        if args.phase == 'train':
            model_train(model, lr=lr)
        elif args.phase == 'test':
            model_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)
if __name__ == '__main__':
    tf.app.run()

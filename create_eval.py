#! /usr/bin/env python2

import glob
import numpy as np
import h5py
import sys
import subprocess
import random
import os
from utils import *
import cv2

def generate_hdf5():
    SRCDIR = '../Data/dark_enh_mb2014/eval/'
    DSTDIR = './data/eval/'
    
    fpdata_evalX  = sorted(glob.glob(SRCDIR + 'X_left/*.png'))
    fpdata_evalY  = sorted(glob.glob(SRCDIR + 'Y_left/*.png'))
    numPics_evalX = len(fpdata_evalX)
    numPics_evalY = len(fpdata_evalY)

    print('%d eval scenes' % numPics_evalY)
    print('%d eval images' % numPics_evalX)

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    
    c = 0
    for i in range(numPics_evalY):
        imgyl = cv2.imread(SRCDIR + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=cv2.INTER_CUBIC)

        fdataxl = sorted(glob.glob(SRCDIR + 'X_left/im%d_*.png' % i))
        
        for j in range(len(fdataxl)):
            imgxl = cv2.imread(SRCDIR + 'X_left/im%d_%d.png'  % (i,j), flags=cv2.IMREAD_COLOR)
            imgxr = cv2.imread(SRCDIR + 'X_right/im%d_%d.png' % (i,j), flags=cv2.IMREAD_COLOR)
            imgxl_rs = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
            imgxr_rs = cv2.resize(imgxr, (int(imgxr.shape[1]/2), int(imgxr.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(DSTDIR + 'X%d_0.png' % c, imgxl_rs)
            cv2.imwrite(DSTDIR + 'X%d_1.png' % c, imgxr_rs)
            cv2.imwrite(DSTDIR + 'Y%d.png'   % c, imgyl_rs)
            c += 1

if __name__ == '__main__':
    generate_hdf5()

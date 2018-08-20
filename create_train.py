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
    PARALLAX = 32
    PATCH_SIZE = 224
    STEPH = 0
    STEPW = PARALLAX
    STRIDE = 112
    SRCDIR = '../Data/dark_enh_mb2014/train/'
    INTERPOLATION = cv2.INTER_CUBIC
    DATAAUG = 0

    fdatax  = sorted(glob.glob(SRCDIR + 'X_left/*.png'))
    fdatay  = sorted(glob.glob(SRCDIR + 'Y_left/*.png'))
    numPicsX = len(fdatax)
    numPicsY = len(fdatay)

    DSTDIR = './data/'
    SAVEPROB = 1
    CHKDIR = './data/chk/'

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    if not os.path.exists(CHKDIR):
        os.makedirs(CHKDIR)
    subprocess.check_call('rm -f {}/*'.format(CHKDIR), shell=True)
    
    count = 0
    for i in range(numPicsY):
        fdatal = sorted(glob.glob(SRCDIR + 'X_left/im%d_*.png' % i))
        for j in range(len(fdatal)):
            imgxl = cv2.imread(fdatal[j], flags=cv2.IMREAD_COLOR)
            imgxl_rs = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=INTERPOLATION)
            im_h, im_w, _ = imgxl_rs.shape
            for x in range(0+STEPH, (im_h-PATCH_SIZE), STRIDE):
                for y in range(0+STEPW, (im_w-PATCH_SIZE), STRIDE):
                    count += 1
    if DATAAUG == 1:
        MODE = [0,1]
        numPatches = count * 2
    else:
        MODE = [0]
        numPatches = count

    FDATA = DSTDIR + ('data_lumchr_da%d_p%d_s%d_par%d_tr%d.hdf5' 
                    % (DATAAUG, PATCH_SIZE, STRIDE, PARALLAX, numPatches))

    print("[*] Info ..")
    print("\t Number of train images = %d" % numPicsX)
    print("\t Number of train scenes = %d" % numPicsY)
    print("\t Number of patches = %d" % numPatches)
    print("\t Patch size = %d" % PATCH_SIZE)
    print("\t Source dir = %s" % SRCDIR)
    print("\t Dest dir = %s" % DSTDIR)
    print("\t Dest file = %s" % FDATA)
    sys.stdout.flush()

    shape_X_lum = (numPatches, PATCH_SIZE, PATCH_SIZE, PARALLAX+1)
    shape_Y_lum = (numPatches, PATCH_SIZE, PATCH_SIZE, 1)
    shape_X_chr = (numPatches, PATCH_SIZE, PATCH_SIZE, 2)
    shape_Y_chr = (numPatches, PATCH_SIZE, PATCH_SIZE, 3)

    hdfile = h5py.File(FDATA, mode = 'w')
    hdfile.create_dataset("X_lum", shape_X_lum, np.uint8)
    hdfile.create_dataset("Y_lum", shape_Y_lum, np.uint8)
    hdfile.create_dataset("X_chr", shape_X_chr, np.uint8)
    hdfile.create_dataset("Y_chr", shape_Y_chr, np.uint8)

    print("[*] Processing Train Images")
    
    c = 0
    for i in range(numPicsY):
        print("\t Tr scene [%2d/%2d]" % (i+1, numPicsY))
        sys.stdout.flush()

        imgyl = cv2.imread(SRCDIR + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)  # BGR
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=INTERPOLATION)
        imgyl_rs_ycrcb = cv2.cvtColor(imgyl_rs, cv2.COLOR_BGR2YCR_CB) # Y Cr Cb [0,1,2]

        fdataxl = sorted(glob.glob(SRCDIR + 'X_left/im%d_*.png' % i))
        fdataxr = sorted(glob.glob(SRCDIR + 'X_right/im%d_*.png' % i))
        for j in range(len(fdataxl)):
            assert fdataxl[j][-5] == fdataxr[j][-5]
            imgxl = cv2.imread(fdataxl[j], flags=cv2.IMREAD_COLOR)
            imgxr = cv2.imread(fdataxr[j], flags=cv2.IMREAD_COLOR)

            imgxl_rs = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=INTERPOLATION)
            imgxr_rs = cv2.resize(imgxr, (int(imgxr.shape[1]/2), int(imgxr.shape[0]/2)), interpolation=INTERPOLATION)

            imgxl_rs_ycrcb = cv2.cvtColor(imgxl_rs, cv2.COLOR_BGR2YCR_CB)
            imgxr_rs_ycrcb = cv2.cvtColor(imgxr_rs, cv2.COLOR_BGR2YCR_CB)

            im_h, im_w, _ = imgxl_rs_ycrcb.shape
            for mode in MODE: # data augmentation: [0,1]
                for x in range(0+STEPH, im_h-PATCH_SIZE, STRIDE):
                    for y in range(0+STEPW, im_w-PATCH_SIZE, STRIDE):
                        xx_lum = np.zeros((1,PATCH_SIZE, PATCH_SIZE, PARALLAX+1))
                        xx_lum[0,:,:,0] = data_augmentation(imgxl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)
                        pp = 0
                        for p in range(1,PARALLAX+1, 1):
                            xx_lum[0,:,:,p] = data_augmentation(imgxr_rs_ycrcb[x:x+PATCH_SIZE,y-pp:y+PATCH_SIZE-pp,0], mode)
                            pp += 1

                        yy_lum = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                        yy_lum[0,:,:,0] = data_augmentation(imgyl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)

                        xx_chr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 2))
                        xx_chr[0,:,:,:] = data_augmentation(imgxl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,1:], mode)
                        yy_chr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
                        yy_chr[0,:,:,:] = data_augmentation(imgyl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:], mode)
                        
                        hdfile["X_lum"][c, ...] = xx_lum
                        hdfile["Y_lum"][c, ...] = yy_lum
                        hdfile["X_chr"][c, ...] = xx_chr
                        hdfile["Y_chr"][c, ...] = yy_chr
   
                        if random.random() > SAVEPROB:
                            for p in range(0,PARALLAX+1,1):
                                cv2.imwrite(CHKDIR + ('%d_lum_in_%d.png' % (c, p)),xx_lum[0,:,:,p])
                            cv2.imwrite(CHKDIR + ('%d_lum_out.png' % c), yy_lum[0,:,:,:])
                        c += 1
    print('%d patches saved.' % c)

if __name__ == '__main__':
    generate_hdf5()

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
    SHIFT_FACTOR = 4
    PATCH_SIZE = 64
    STEPH = 0
    STEPW = PARALLAX * SHIFT_FACTOR
    STRIDE = 64
    SRCDIR = '../Data/dark_enh_mb2014/train/'
    SRCDIR_EVAL = '../Data/dark_enh_mb2014/eval/'
    INTERPOLATION = cv2.INTER_CUBIC
    DATAAUG = 0
    LL_en = 0   # WARNING IMPORTANT!!!!!!

    fdatax  = sorted(glob.glob(SRCDIR + 'X_left/*.png'))
    fdatay  = sorted(glob.glob(SRCDIR + 'Y_left/*.png'))
    numPicsX = len(fdatax)
    numPicsY = len(fdatay)

    fdatax_eval = sorted(glob.glob(SRCDIR_EVAL + 'X_left/*.png'))
    fdatay_eval = sorted(glob.glob(SRCDIR_EVAL + 'Y_left/*.png'))
    numEvalX = len(fdatax_eval)
    numEvalY = len(fdatay_eval)

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

    if LL_en == 1:
        FDATA = DSTDIR + ('data_lumchr_LL_da%d_p%d_s%d_par%d_tr%d.hdf5' 
                       % (DATAAUG, PATCH_SIZE, STRIDE, PARALLAX, numPatches))
    else:
        FDATA = DSTDIR + ('data_lumchr_LR_da%d_p%d_s%d_par%d_tr%d.hdf5' 
                       % (DATAAUG, PATCH_SIZE, STRIDE, PARALLAX, numPatches))


    print("[*] Info ..")
    print("\t Number of train images = %d" % numPicsX)
    print("\t Number of train scenes = %d" % numPicsY)
    print("\t Number of eval images = %d" % numEvalX)
    print("\t Number of eval scenes = %d" % numEvalY)
    print("\t Number of patches = %d" % numPatches)
    print("\t Patch size = %d" % PATCH_SIZE)
    print("\t Source dir = %s" % SRCDIR)
    print("\t Dest dir = %s" % DSTDIR)
    print("\t Dest file = %s" % FDATA)
    sys.stdout.flush()

    shape_X_lum0_tr =  (numPatches, PATCH_SIZE, PATCH_SIZE, 1)
    shape_X_lum_tr =   (numPatches, PATCH_SIZE, PATCH_SIZE, PARALLAX)
    shape_X_chr_tr =   (numPatches, PATCH_SIZE, PATCH_SIZE, 2)
    
    shape_Y_lum_tr =   (numPatches, PATCH_SIZE, PATCH_SIZE, 1)
    shape_Y_ycrcb_tr = (numPatches, PATCH_SIZE, PATCH_SIZE, 3)

    imgxl_eval = cv2.imread(SRCDIR_EVAL + 'X_left/im0_0.png', flags=cv2.IMREAD_COLOR)
    imgxl_rs_eval = cv2.resize(imgxl_eval, (int(imgxl_eval.shape[1]/2), int(imgxl_eval.shape[0]/2)), interpolation=INTERPOLATION)
    height_eval, width_eval, _ = imgxl_rs_eval.shape

    shape_X_lum0_eval = (numEvalX, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 1)
    shape_X_lum_eval = (numEvalX, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, PARALLAX)
    shape_X_chr_eval = (numEvalX, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 2)
    
    shape_Y_lum_eval = (numEvalX, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 1)
    shape_Y_ycrcb_eval = (numEvalX, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 3)


    hdfile = h5py.File(FDATA, mode = 'w')
    hdfile.create_dataset("X_lum0_tr", shape_X_lum0_tr, np.uint8)
    hdfile.create_dataset("X_lum_tr", shape_X_lum_tr, np.uint8)
    hdfile.create_dataset("Y_lum_tr", shape_Y_lum_tr, np.uint8)
    hdfile.create_dataset("X_chr_tr", shape_X_chr_tr, np.uint8)
    hdfile.create_dataset("Y_ycrcb_tr", shape_Y_ycrcb_tr, np.uint8)

    hdfile.create_dataset("X_lum0_eval", shape_X_lum0_eval, np.uint8)
    hdfile.create_dataset("X_lum_eval", shape_X_lum_eval, np.uint8)
    hdfile.create_dataset("Y_lum_eval", shape_Y_lum_eval, np.uint8)
    hdfile.create_dataset("X_chr_eval", shape_X_chr_eval, np.uint8)
    hdfile.create_dataset("Y_ycrcb_eval", shape_Y_ycrcb_eval, np.uint8)

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
            imgxl           = cv2.imread(fdataxl[j], flags=cv2.IMREAD_COLOR)
            imgxl_rs        = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=INTERPOLATION)
            imgxl_rs_ycrcb  = cv2.cvtColor(imgxl_rs, cv2.COLOR_BGR2YCR_CB)
            
            if LL_en != 1:
                imgxr           = cv2.imread(fdataxr[j], flags=cv2.IMREAD_COLOR)
                imgxr_rs        = cv2.resize(imgxr, (int(imgxr.shape[1]/2), int(imgxr.shape[0]/2)), interpolation=INTERPOLATION)
                imgxr_rs_ycrcb  = cv2.cvtColor(imgxr_rs, cv2.COLOR_BGR2YCR_CB)

            im_h, im_w, _ = imgxl_rs_ycrcb.shape
            for mode in MODE: # data augmentation: [0,1]
                for x in range(0+STEPH, im_h-PATCH_SIZE, STRIDE):
                    for y in range(0+STEPW, im_w-PATCH_SIZE, STRIDE):
                        xx_lum0 = np.zeros((1,PATCH_SIZE, PATCH_SIZE, 1))
                        xx_lum0[0,:,:,0] = data_augmentation(imgxl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)
                        
                        xx_lum  = np.zeros((1,PATCH_SIZE, PATCH_SIZE, PARALLAX))
                        pp = 0
                        for p in range(0,PARALLAX, 1):
                            if LL_en == 1:
                                xx_lum[0,:,:,p] = data_augmentation(imgxl_rs_ycrcb[x:x+PATCH_SIZE,y-pp:y+PATCH_SIZE-pp,0], mode)
                            else:
                                xx_lum[0,:,:,p] = data_augmentation(imgxr_rs_ycrcb[x:x+PATCH_SIZE,y-pp:y+PATCH_SIZE-pp,0], mode)
                            pp += SHIFT_FACTOR

                        yy_lum = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                        yy_lum[0,:,:,0] = data_augmentation(imgyl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,0], mode)

                        xx_chr = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 2))
                        xx_chr[0,:,:,:] = data_augmentation(imgxl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,1:], mode)
                        yy_ycrcb = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
                        yy_ycrcb[0,:,:,:] = data_augmentation(imgyl_rs_ycrcb[x:x+PATCH_SIZE,y:y+PATCH_SIZE,:], mode)
                        
                        hdfile["X_lum0_tr"][c, ...] = xx_lum0
                        hdfile["X_lum_tr"][c, ...] = xx_lum
                        hdfile["Y_lum_tr"][c, ...] = yy_lum
                        hdfile["X_chr_tr"][c, ...] = xx_chr
                        hdfile["Y_ycrcb_tr"][c, ...] = yy_ycrcb
   
                        if random.random() > SAVEPROB:
                            cv2.imwrite(CHKDIR + ('%d_lum_in_0.png' % c),xx_lum0[0,:,:,0])
                            for p in range(0,PARALLAX,1):
                                cv2.imwrite(CHKDIR + ('%d_lum_in_%d.png' % (c, p)),xx_lum[0,:,:,p])
                            cv2.imwrite(CHKDIR + ('%d_lum_out.png' % c), yy_lum[0,:,:,:])
                        c += 1
    print('%d patches saved.' % c)


    print("[*] Processing Eval Images")
    
    c = 0
    for i in range(numEvalY):
        print("\t Eval scene [%2d/%2d]" % (i+1, numEvalY))
        sys.stdout.flush()

        imgyl = cv2.imread(SRCDIR_EVAL + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)  # BGR
        imgyl_rs = cv2.resize(imgyl, (width_eval, height_eval), interpolation=INTERPOLATION)
        imgyl_rs_ycrcb = cv2.cvtColor(imgyl_rs, cv2.COLOR_BGR2YCR_CB) # Y Cr Cb [0,1,2]

        fdataxl = sorted(glob.glob(SRCDIR_EVAL + 'X_left/im%d_*.png' % i))
        fdataxr = sorted(glob.glob(SRCDIR_EVAL + 'X_right/im%d_*.png' % i))
        for j in range(len(fdataxl)):
            assert fdataxl[j][-5] == fdataxr[j][-5]
            imgxl           = cv2.imread(fdataxl[j], flags=cv2.IMREAD_COLOR)
            imgxl_rs        = cv2.resize(imgxl, (width_eval, height_eval), interpolation=INTERPOLATION)
            imgxl_rs_ycrcb  = cv2.cvtColor(imgxl_rs, cv2.COLOR_BGR2YCR_CB)
            
            if LL_en != 1:
                imgxr           = cv2.imread(fdataxr[j], flags=cv2.IMREAD_COLOR)
                imgxr_rs        = cv2.resize(imgxr, (width_eval, height_eval), interpolation=INTERPOLATION)
                imgxr_rs_ycrcb  = cv2.cvtColor(imgxr_rs, cv2.COLOR_BGR2YCR_CB)

            xx_lum0 = np.zeros((1,height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 1))
            xx_lum0[0,:,:,0] = imgxl_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR:,0]

            xx_lum = np.zeros((1,height_eval, width_eval-PARALLAX*SHIFT_FACTOR, PARALLAX))
            pp = 0
            for p in range(0, PARALLAX, 1):
                if LL_en == 1:
                    xx_lum[0,:,:,p] = imgxl_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR-pp:width_eval-pp,0]
                else:
                    xx_lum[0,:,:,p] = imgxr_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR-pp:width_eval-pp,0]
                pp += SHIFT_FACTOR

            yy_lum = np.zeros((1, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 1))
            yy_lum[0,:,:,0] = imgyl_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR:,0]

            xx_chr = np.zeros((1, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 2))
            xx_chr[0,:,:,:] = imgxl_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR:,1:]

            yy_ycrcb = np.zeros((1, height_eval, width_eval-PARALLAX*SHIFT_FACTOR, 3))
            yy_ycrcb[0,:,:,:] = imgyl_rs_ycrcb[:,PARALLAX*SHIFT_FACTOR:,:]
            
            hdfile["X_lum0_eval"][c, ...] = xx_lum0
            hdfile["X_lum_eval"][c, ...] = xx_lum
            hdfile["Y_lum_eval"][c, ...] = yy_lum
            hdfile["X_chr_eval"][c, ...] = xx_chr
            hdfile["Y_ycrcb_eval"][c, ...] = yy_ycrcb
            
            c += 1
    print('%d Images saved.' % c)

if __name__ == '__main__':
    generate_hdf5()

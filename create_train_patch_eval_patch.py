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
import math

def generate_hdf5():
    np.random.seed(42)

    NPATCHES = 17 #number of neighbor patches + itself
    SHIFT_FACTOR = 1
    PATCH_SIZE = 32
    SRCDIR_TR   = '../Data/dark_enh_mb2014/train/'
    SRCDIR_EVAL = '../Data/dark_enh_mb2014/eval/'
    INTERPOLATION = cv2.INTER_CUBIC
    PATCH_PER_IMAGE = 1 # %
    LL_en = 0   # WARNING IMPORTANT!!!!!!

    fdatax  = sorted(glob.glob(SRCDIR_TR + 'X_left/*.png'))
    fdatay  = sorted(glob.glob(SRCDIR_TR + 'Y_left/*.png'))
    numPicsX = len(fdatax)
    numPicsY = len(fdatay)

    fdatax_eval = sorted(glob.glob(SRCDIR_EVAL + 'X_left/*.png'))
    fdatay_eval = sorted(glob.glob(SRCDIR_EVAL + 'Y_left/*.png'))
    numEvalX = len(fdatax_eval)
    numEvalY = len(fdatay_eval)

    DSTDIR = './data/'
    SAVEPROB = 1
    CHKDIR = './data/chk/'
    TMPDIR = './tmp/'  # DO NOT CHANGE!!!

    if not os.path.exists(DSTDIR):
        os.makedirs(DSTDIR)
    if not os.path.exists(CHKDIR):
        os.makedirs(CHKDIR)
    if not os.path.exists(TMPDIR):
        os.makedirs(TMPDIR)
    subprocess.check_call('rm -f {}/*'.format(CHKDIR), shell=True)
    subprocess.check_call('rm -f {}/*'.format(TMPDIR), shell=True)
    assert PATCH_SIZE % 2 == 0
    assert NPATCHES % 2 == 1
   
    count = 0
    fdisp = sorted(glob.glob(SRCDIR_TR + 'Disp/disp*.pfm'))
    numDisp = len(fdisp) / 3
    for i in range(numPicsY):
        imgyl = cv2.imread(SRCDIR_TR + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=INTERPOLATION)
        img_h, img_w, _ = imgyl_rs.shape
        
        numX = len(sorted(glob.glob(SRCDIR_TR + 'X_left/im%d_*.png' % i)))

        disp0, scale0   = load_pfm(SRCDIR_TR + 'Disp/disp%d_0.pfm'  % mapMB(i), True)
        disp0y, scale0y = load_pfm(SRCDIR_TR + 'Disp/disp%d_0y.pfm' % mapMB(i), True)
        disp1, scale1   = load_pfm(SRCDIR_TR + 'Disp/disp%d_1.pfm'  % mapMB(i), True)

        save_pfm(TMPDIR + 'disp0.pfm' , disp0, 1)
        save_pfm(TMPDIR + 'disp1.pfm' , disp1, 1)
        save_pfm(TMPDIR + 'disp0y.pfm', disp0y,1)

        subprocess.check_output('/home/hassanih/MiddEval3/code/computemask tmp/disp0.pfm tmp/disp0y.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread(TMPDIR + 'mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        condx = (x > disp0.max() + math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)) & \
                (x < img_w - (math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)))
        x = x[condx]
        y = y[condx]

        condy = (y > math.ceil(PATCH_SIZE/2)) & (y < img_h - math.ceil(PATCH_SIZE/2)-1)
        x = x[condy]
        y = y[condy]
        
        for kk in range(numX):
            count += int(len(x) * PATCH_PER_IMAGE / 100)
        
    numPatches_tr = count

    count = 0
    fdisp = sorted(glob.glob(SRCDIR_EVAL + 'Disp/disp*.pfm'))
    numDisp = len(fdisp) / 3
    for i in range(numEvalY):
        imgyl = cv2.imread(SRCDIR_EVAL + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=INTERPOLATION)
        img_h, img_w, _ = imgyl_rs.shape
        
        numX = len(sorted(glob.glob(SRCDIR_EVAL + 'X_left/im%d_*.png' % i)))

        disp0, scale0   = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_0.pfm'  % mapMB(i), True)
        disp0y, scale0y = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_0y.pfm' % mapMB(i), True)
        disp1, scale1   = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_1.pfm'  % mapMB(i), True)

        save_pfm(TMPDIR + 'disp0.pfm' , disp0, 1)
        save_pfm(TMPDIR + 'disp1.pfm' , disp1, 1)
        save_pfm(TMPDIR + 'disp0y.pfm', disp0y,1)

        subprocess.check_output('/home/hassanih/MiddEval3/code/computemask tmp/disp0.pfm tmp/disp0y.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread(TMPDIR + 'mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        condx = (x > disp0.max() + math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)) & \
                (x < img_w - (math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)))
        x = x[condx]
        y = y[condx]

        condy = (y > math.ceil(PATCH_SIZE/2)) & (y < img_h - math.ceil(PATCH_SIZE/2)-1)
        x = x[condy]
        y = y[condy]
        
        for kk in range(numX):
            count += int(len(x) * PATCH_PER_IMAGE / 100)
    
    numPatches_eval = count

    if LL_en == 1:
        FDATA = DSTDIR + ('data_evalP_LL_npatch%d_shft%d_ps%d_ppi%.2f_tr%d_eval%d.hdf5' 
                       % (NPATCHES, SHIFT_FACTOR, PATCH_SIZE, PATCH_PER_IMAGE, numPatches_tr, numPatches_eval))
    else:
        FDATA = DSTDIR + ('data_evalP_LR_npatch%d_shft%d_ps%d_ppi%.2f_tr%d_eval%d.hdf5' 
                       % (NPATCHES, SHIFT_FACTOR, PATCH_SIZE, PATCH_PER_IMAGE, numPatches_tr, numPatches_eval))


    print("[*] Info ..")
    print("\t Number of train images = %d" % numPicsX)
    print("\t Number of train scenes = %d" % numPicsY)
    print("\t Number of eval images = %d" % numEvalX)
    print("\t Number of eval scenes = %d" % numEvalY)
    print("\t Number of train patches = %d" % numPatches_tr)
    print("\t Number of eval patches = %d" % numPatches_eval)
    print("\t Number of neighbor patches = %d" % NPATCHES)
    print("\t Shift factor = %d" % SHIFT_FACTOR)
    print("\t Patches per image percentage = %.2f" % PATCH_PER_IMAGE)
    print("\t Patch size = %d" % PATCH_SIZE)
    print("\t Source dir = %s" % SRCDIR_TR)
    print("\t Dest dir = %s" % DSTDIR)
    print("\t Dest file = %s" % FDATA)

    shape_X_lum0_tr =  (numPatches_tr, PATCH_SIZE, PATCH_SIZE, 1)
    shape_X_lum_tr =   (numPatches_tr, PATCH_SIZE, PATCH_SIZE, NPATCHES)
    shape_X_chr_tr =   (numPatches_tr, PATCH_SIZE, PATCH_SIZE, 2)
    
    shape_Y_lum_tr =   (numPatches_tr, PATCH_SIZE, PATCH_SIZE, 1)
    shape_Y_ycrcb_tr = (numPatches_tr, PATCH_SIZE, PATCH_SIZE, 3)

    hdfile = h5py.File(FDATA, mode = 'w')
    hdfile.create_dataset("X_lum0_tr", shape_X_lum0_tr, np.uint8)
    hdfile.create_dataset("X_lum_tr", shape_X_lum_tr, np.uint8)
    hdfile.create_dataset("Y_lum_tr", shape_Y_lum_tr, np.uint8)
    hdfile.create_dataset("X_chr_tr", shape_X_chr_tr, np.uint8)
    hdfile.create_dataset("Y_ycrcb_tr", shape_Y_ycrcb_tr, np.uint8)

    print("[*] Processing Train Images")
    
    c = 0
    for i in range(numPicsY):
        imgyl = cv2.imread(SRCDIR_TR + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)  # BGR
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=INTERPOLATION)
        imgyl_rs_ycrcb = cv2.cvtColor(imgyl_rs, cv2.COLOR_BGR2YCR_CB) # Y Cr Cb [0,1,2
        img_h, img_w, _ = imgyl_rs_ycrcb.shape

        disp0, scale0   = load_pfm(SRCDIR_TR + 'Disp/disp%d_0.pfm'  % mapMB(i), True)
        disp0y, scale0y = load_pfm(SRCDIR_TR + 'Disp/disp%d_0y.pfm' % mapMB(i), True)
        disp1, scale1   = load_pfm(SRCDIR_TR + 'Disp/disp%d_1.pfm'  % mapMB(i), True)

        save_pfm(TMPDIR + 'disp0.pfm' , disp0, 1)
        save_pfm(TMPDIR + 'disp1.pfm' , disp1, 1)
        save_pfm(TMPDIR + 'disp0y.pfm', disp0y,1)

        subprocess.check_output('/home/hassanih/MiddEval3/code/computemask tmp/disp0.pfm tmp/disp0y.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread(TMPDIR + 'mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        condx = (x > disp0.max() + math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)) & \
                (x < img_w - (math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)))
        x = x[condx]
        y = y[condx]
        
        condy = (y > math.ceil(PATCH_SIZE/2)) & (y < img_h - math.ceil(PATCH_SIZE/2)-1)
        x = x[condy]
        y = y[condy]
        
        N = int(PATCH_PER_IMAGE * len(x) / 100)

        fdataxl = sorted(glob.glob(SRCDIR_TR + 'X_left/im%d_*.png' % i))
        fdataxr = sorted(glob.glob(SRCDIR_TR + 'X_right/im%d_*.png' % i))
        
        print("\t Tr scene [%2d/%2d], %d X images, %d patches per X images" % (i+1, numPicsY, len(fdataxl), N))
        
        for j in range(len(fdataxl)):
            assert fdataxl[j][-5] == fdataxr[j][-5]
            imgxl           = cv2.imread(fdataxl[j], flags=cv2.IMREAD_COLOR)
            imgxl_rs        = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=INTERPOLATION)
            imgxl_rs_ycrcb  = cv2.cvtColor(imgxl_rs, cv2.COLOR_BGR2YCR_CB)
            
            if LL_en != 1:
                imgxr           = cv2.imread(fdataxr[j], flags=cv2.IMREAD_COLOR)
                imgxr_rs        = cv2.resize(imgxr, (int(imgxr.shape[1]/2), int(imgxr.shape[0]/2)), interpolation=INTERPOLATION)
                imgxr_rs_ycrcb  = cv2.cvtColor(imgxr_rs, cv2.COLOR_BGR2YCR_CB)
            
            idx = np.random.randint(0, len(x), N)
            for n in range(N):
                xx_lum0  = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                xx_lum   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, NPATCHES))
                xx_chr   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 2))
                yy_lum   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                yy_ycrcb = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))
                
                xx_lum0[0,:,:,0]  = imgxl_rs_ycrcb[y[idx[n]] - PATCH_SIZE/2:y[idx[n]] + PATCH_SIZE/2,x[idx[n]] - PATCH_SIZE/2:x[idx[n]] + PATCH_SIZE/2,0]

                pp = SHIFT_FACTOR * int(NPATCHES/2)
                xtmp = x[idx[n]] - int(round(disp0[y[idx[n]],x[idx[n]]]))

                k = NPATCHES - 1
                for p in range(int(NPATCHES/2)):
                    if LL_en == 1:
                        xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]+pp-PATCH_SIZE/2:x[idx[n]]+pp+PATCH_SIZE/2,0]
                    else:
                        xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp+pp-PATCH_SIZE/2:xtmp+pp+PATCH_SIZE/2,0]
                    k -= 1
                    pp -= SHIFT_FACTOR
                if LL_en == 1:
                    xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,0]
                else:
                    xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp-PATCH_SIZE/2:xtmp+PATCH_SIZE/2,0]
                k -= 1
                pp -= SHIFT_FACTOR
                for p in range(int(NPATCHES/2)):
                    if LL_en == 1:
                        xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]+pp-PATCH_SIZE/2:x[idx[n]]+pp+PATCH_SIZE/2,0]
                    else:
                        xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp+pp-PATCH_SIZE/2:xtmp+pp+PATCH_SIZE/2,0]
                    k -= 1
                    pp -= SHIFT_FACTOR

                xx_chr[0,:,:,:] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,1:]

                yy_lum[0,:,:,0] = imgyl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,0]

                yy_ycrcb[0,:,:,:] = imgyl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,:]
                
                hdfile["X_lum0_tr"][c, ...] = xx_lum0
                hdfile["X_lum_tr"][c, ...] = xx_lum
                hdfile["Y_lum_tr"][c, ...] = yy_lum
                hdfile["X_chr_tr"][c, ...] = xx_chr
                hdfile["Y_ycrcb_tr"][c, ...] = yy_ycrcb
                
                c += 1


    print('%d patches saved.' % c)

    ##################################################################################################
    print("[*] Processing Evaluation Images")
    ##################################################################################################
    shape_X_lum0_eval  = (numPatches_eval, PATCH_SIZE, PATCH_SIZE, 1)
    shape_X_lum_eval   = (numPatches_eval, PATCH_SIZE, PATCH_SIZE, NPATCHES)
    shape_X_chr_eval   = (numPatches_eval, PATCH_SIZE, PATCH_SIZE, 2)
    
    shape_Y_lum_eval   = (numPatches_eval, PATCH_SIZE, PATCH_SIZE, 1)
    shape_Y_ycrcb_eval = (numPatches_eval, PATCH_SIZE, PATCH_SIZE, 3)

    hdfile.create_dataset("X_lum0_eval", shape_X_lum0_eval, np.uint8)
    hdfile.create_dataset("X_lum_eval", shape_X_lum_eval, np.uint8)
    hdfile.create_dataset("Y_lum_eval", shape_Y_lum_eval, np.uint8)
    hdfile.create_dataset("X_chr_eval", shape_X_chr_eval, np.uint8)
    hdfile.create_dataset("Y_ycrcb_eval", shape_Y_ycrcb_eval, np.uint8)

    c = 0
    for i in range(numEvalY):
        imgyl = cv2.imread(SRCDIR_EVAL + 'Y_left/im%d.png' % i, flags=cv2.IMREAD_COLOR)  # BGR
        imgyl_rs = cv2.resize(imgyl, (int(imgyl.shape[1]/2), int(imgyl.shape[0]/2)), interpolation=INTERPOLATION)
        imgyl_rs_ycrcb = cv2.cvtColor(imgyl_rs, cv2.COLOR_BGR2YCR_CB) # Y Cr Cb [0,1,2]
        img_h, img_w, _ = imgyl_rs_ycrcb.shape

        disp0, scale0   = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_0.pfm'  % mapMB(i), True)
        disp0y, scale0y = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_0y.pfm' % mapMB(i), True)
        disp1, scale1   = load_pfm(SRCDIR_EVAL + 'Disp/disp%d_1.pfm'  % mapMB(i), True)

        save_pfm(TMPDIR + 'disp0.pfm' , disp0, 1)
        save_pfm(TMPDIR + 'disp1.pfm' , disp1, 1)
        save_pfm(TMPDIR + 'disp0y.pfm', disp0y,1)

        subprocess.check_output('/home/hassanih/MiddEval3/code/computemask tmp/disp0.pfm tmp/disp0y.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread(TMPDIR + 'mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        condx = (x > disp0.max() + math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)) & \
                (x < img_w - (math.ceil(PATCH_SIZE/2)+SHIFT_FACTOR*int(NPATCHES/2)))
        x = x[condx]
        y = y[condx]

        condy = (y > math.ceil(PATCH_SIZE/2)) & (y < img_h - math.ceil(PATCH_SIZE/2)-1)
        x = x[condy]
        y = y[condy]

        N = int(PATCH_PER_IMAGE * len(y) / 100)

        fdataxl = sorted(glob.glob(SRCDIR_EVAL + 'X_left/im%d_*.png' % i))
        fdataxr = sorted(glob.glob(SRCDIR_EVAL + 'X_right/im%d_*.png' % i))
        
        print("\t Eval scene [%2d/%2d], %d X images, %d patches per X images" % (i+1, numEvalY, len(fdataxl), N))
        
        for j in range(len(fdataxl)):
            assert fdataxl[j][-5] == fdataxr[j][-5]
            imgxl           = cv2.imread(fdataxl[j], flags=cv2.IMREAD_COLOR)
            imgxl_rs        = cv2.resize(imgxl, (int(imgxl.shape[1]/2), int(imgxl.shape[0]/2)), interpolation=INTERPOLATION)
            imgxl_rs_ycrcb  = cv2.cvtColor(imgxl_rs, cv2.COLOR_BGR2YCR_CB)
            
            if LL_en != 1:
                imgxr           = cv2.imread(fdataxr[j], flags=cv2.IMREAD_COLOR)
                imgxr_rs        = cv2.resize(imgxr, (int(imgxr.shape[1]/2), int(imgxr.shape[0]/2)), interpolation=INTERPOLATION)
                imgxr_rs_ycrcb  = cv2.cvtColor(imgxr_rs, cv2.COLOR_BGR2YCR_CB)

            idx = np.random.randint(0, len(y), N)
            for n in range(N):
                xx_lum0  = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                xx_lum   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, NPATCHES))
                xx_chr   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 2))
                yy_lum   = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 1))
                yy_ycrcb = np.zeros((1, PATCH_SIZE, PATCH_SIZE, 3))

                xx_lum0[0,:,:,0]  = imgxl_rs_ycrcb[y[idx[n]] - PATCH_SIZE/2:y[idx[n]] + PATCH_SIZE/2,x[idx[n]] - PATCH_SIZE/2:x[idx[n]] + PATCH_SIZE/2,0]

                pp = SHIFT_FACTOR * int(NPATCHES/2)
                xtmp = x[idx[n]] - int(round(disp0[y[idx[n]],x[idx[n]]]))

                k = NPATCHES - 1
                for p in range(int(NPATCHES/2)):
                    if LL_en == 1:
                        xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]+pp-PATCH_SIZE/2:x[idx[n]]+pp+PATCH_SIZE/2,0]
                    else:
                        xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp+pp-PATCH_SIZE/2:xtmp+pp+PATCH_SIZE/2,0]
                    k -= 1
                    pp -= SHIFT_FACTOR
                if LL_en == 1:
                    xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,0]
                else:
                    xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp-PATCH_SIZE/2:xtmp+PATCH_SIZE/2,0]
                k -= 1
                pp -= SHIFT_FACTOR
                for p in range(int(NPATCHES/2)):
                    if LL_en == 1:
                        xx_lum[0,:,:,k] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]+pp-PATCH_SIZE/2:x[idx[n]]+pp+PATCH_SIZE/2,0]
                    else:
                        xx_lum[0,:,:,k] = imgxr_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,xtmp+pp-PATCH_SIZE/2:xtmp+pp+PATCH_SIZE/2,0]
                    k -= 1
                    pp -= SHIFT_FACTOR

                xx_chr[0,:,:,:] = imgxl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,1:]

                yy_lum[0,:,:,0] = imgyl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,0]

                yy_ycrcb[0,:,:,:] = imgyl_rs_ycrcb[y[idx[n]]-PATCH_SIZE/2:y[idx[n]]+PATCH_SIZE/2,x[idx[n]]-PATCH_SIZE/2:x[idx[n]]+PATCH_SIZE/2,:]
                
                hdfile["X_lum0_eval"][c, ...] = xx_lum0
                hdfile["X_lum_eval"][c, ...] = xx_lum
                hdfile["Y_lum_eval"][c, ...] = yy_lum
                hdfile["X_chr_eval"][c, ...] = xx_chr
                hdfile["Y_ycrcb_eval"][c, ...] = yy_ycrcb
                
                c += 1


    print('%d patches saved.' % c)

    subprocess.check_call('rm -r -f {}/'.format(TMPDIR), shell=True)
if __name__ == '__main__':
    generate_hdf5()

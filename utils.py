import gc
import os
import sys

import numpy as np
import tensorflow as tf
import cv2
import h5py

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

class train_data():
    def __init__(self, filepath='./data/data_da1_p33_s24_b128_tr60.hdf5'):
        self.filepath = filepath
        assert '.hdf5' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = h5py.File(self.filepath, "r")
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")

def load_data(filepath='./data/data_da1_p33_s24_b128_tr60.hdf5'):
    return train_data(filepath=filepath)

def normalize(x):
    '''
    normalize x to -1 and 1
    '''
    if len(x.shape) == 4:
        c = x.shape[3]
    elif len(x.shape) == 3:
        c = x.shape[2]
    elif len(x.shape) == 2:
        c = 1
    else:
        assert False
    for i in range(c):
        if len(x.shape) == 4:
            xx = x[:,:,:,i]
        elif len(x.shape) == 3:
            xx = x[:,:,i]
        elif len(x.shape) == 2:
            xx = x[:,:]
        M = xx.max()
        m = xx.min()
        xx2 = (2 * xx - m - M) / (M-m+sys.float_info.epsilon)
        #xx2 = (2 * xx - 0 - 1) / (1-0)
        if len(x.shape) == 4:
            x[:,:,:,i] = xx2
        elif len(x.shape) == 3:
            x[:,:,i] = xx2
        elif len(x.shape) == 2:
            x[:,:] = xx2
    return np.clip(x, -1, 1)

def denormalize(x, m, M):
    if len(x.shape) == 4:
        c = x.shape[3]
    elif len(x.shape) == 3:
        c = x.shape[2]
    elif len(x.shape) == 2:
        c = 1
    for i in range(c):
        if len(x.shape) == 4:
            xx = x[:,:,:,i]
        elif len(x.shape) == 3:
            xx = x[:,:,i]
        elif len(x.shape) == 2:
            xx = x[:,:]
        a = xx.min()
        b = xx.max()
        xx2 = ((m-M)/(a-b)) * (xx - b) + M
        if len(x.shape) == 4:
            x[:,:,:,i] = xx2
        elif len(x.shape) == 3:
            x[:,:,i] = xx2
        elif len(x.shape) == 2:
            x[:,:] = xx2
    return np.clip(x, m, M)

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = cv2.imread(filelist, flags=cv2.IMREAD_COLOR)
        im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)    # Y Cr Cb = [0,1,2]
        return im_ycrcb
    data = []
    for file in filelist:
        im = cv2.imread(file, flags=cv2.IMREAD_COLOR)
        im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
        data.append(im_ycrcb)
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not noisy_image.any() and not clean_image.any():
        cat_image = ground_truth
        cv2.imwrite(filepath, cat_image.astype('uint8'))
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
        cat_image = cv2.cvtColor(cat_image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(filepath, cat_image.astype('uint8'))


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

def tf_ycrcb2rgb(x):
    # assert x values are in [0, 255]
    n = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    xform = tf.constant([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]], dtype=tf.float32)
    #c128 = tf.constant(-128*np.ones((n,h,w,c)), dtype=tf.float32)
    y, cr, cb = tf.split(x, 3, 3)
    cr = tf.add(cr, -128)
    cb = tf.add(cb, -128)
    rgb = tf.concat([y, cr, cb],3)
    xformT = tf.transpose(xform)
    rgb = tf.reshape(rgb, shape=[n*h*w,c])
    rgb = tf.matmul(rgb, xformT)
    rgb = tf.reshape(rgb, shape=[n,h,w,c])
    rgb = tf.clip_by_value(rgb, 0, 255)
    return rgb



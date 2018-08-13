import time
from utils import *
import random

def resBlock(x, p1, p2, name, is_training):
    # p1["fs"]: feature map size
    # p1["fw"]: filter size
    with tf.variable_scope(name):
        r = tf.layers.conv2d(x, p1["fs"], p1["fw"], padding='same', name=name+'_conv1', activation=None)
        r = tf.layers.batch_normalization(r, training=is_training, name=name+'_BN1')
        r = tf.nn.leaky_relu(r, alpha=0.2, name=name+'_lrelu')
        r = tf.layers.conv2d(r, p2["fs"], p2["fw"], padding='same', name=name+'_conv2', activation=None)
        r = tf.layers.batch_normalization(r, training=is_training, name=name+'_BN2')
        y = tf.add(r,x)
        return y

def dualenh_lum(x_lum0, x_lum, is_training, L=32):
    fm = 64
    xx  = tf.concat([x_lum0, x_lum], 3)
    res = tf.layers.conv2d(xx, fm, 3, padding='same', name='lum_conv_start', activation=None)
    res = tf.nn.leaky_relu(res, alpha=0.2, name='lum_lrelu_start')
    
    p1 = {"fs": fm, "fw": 3}

    for i in range(L):
        res = resBlock(res, p1, p1, 'lum_resB%d' % (i+1), is_training)
    
    res = tf.layers.conv2d(res, fm, 3, padding='same', name='lum_conv', activation=None)
    res = tf.layers.batch_normalization(res, training=is_training, name='lum_BN')
    
    y = tf.layers.conv2d(res, 1, 3, padding='same', name='lum_conv_last', activation=None)
    y = tf.nn.tanh(y, name='lum_tanh_last')

    return y

def enh_chr(y_hat_lum, x_chr, is_training, L=32):
    fm = 64
    y = tf.concat([y_hat_lum, x_chr], 3)
    y = tf.layers.conv2d(y, fm, 3, padding='same', name='chr_conv_start', activation=None)
    y = tf.nn.leaky_relu(y, alpha=0.2, name='chr_lrelu_start')

    p1 = {"fs": fm, "fw": 3}

    for i in range(L):
        y = resBlock(y, p1, p1, 'chr_resB%d' % (i+1), is_training)

    y = tf.layers.conv2d(y, fm, 3, padding='same', name='chr_conv', activation=None)
    y = tf.layers.batch_normalization(y, training=is_training, name='chr_BN')

    y = tf.layers.conv2d(y, 3, 3, padding='same', name='chr_conv_last', activation=None)
    y = tf.nn.tanh(y, name='chr_tanh_last')

    return y

def singleenh_lum(x_lum0, is_training, L=32):
    fm = 64
    res = tf.layers.conv2d(x_lum0, fm, 3, padding='same', name='lum_conv_start', activation=None)
    res = tf.nn.leaky_relu(res, alpha=0.2, name='lum_relu_start')

    p1 = {"fs": fm, "fw":3}

    for i in range(L):
        res = resBlock(res, p1, p1, 'lum_resB%d' % (i+1), is_training)
    
    res = tf.layers.conv2d(res, fm, 3, padding='same', name='lum_conv', activation=None)
    res = tf.layers.batch_normalization(res, training=is_training, name='lum_BN')

    y = tf.layers.conv2d(res, 1, 3, padding='same', name='lum_conv_last', activation=None)
    y = tf.nn.tanh(y, name='lum_tanh_last')

    return y

class imdualenh(object):
    def __init__(self, sess, batch_size=128, PARALLAX=64, model="dual", logfile="./logs/log.txt", lmbd=0.6):
        self.sess = sess
        self.parallax = PARALLAX
        self.logfile = open(logfile, "w")

        # Labels
        self.Y_LUM_   = tf.placeholder(tf.float32, [None, None, None, 1], name='Y_GT_LUM')
        self.Y_YCRCB_ = tf.placeholder(tf.float32, [None, None, None, 3], name='Y_GY_YCrCb')
        
        # Inputs
        self.X_lum0   = tf.placeholder(tf.float32, [None, None, None, 1], name='X_LUM_LEFT')
        self.X_lum    = tf.placeholder(tf.float32, [None, None, None, self.parallax], name='X_LUM_RIGHT_PATCHES')
        self.X_chr    = tf.placeholder(tf.float32, [None, None, None, 2], name='X_CHR_LEFT')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        L = 32
        if model == "dual":
	    self.Y_LUM = dualenh_lum(self.X_lum0, self.X_lum, self.is_training, L)
            self.Y_YCRCB = enh_chr(self.Y_LUM, self.X_chr, self.is_training, L)
	elif model == "single":
	    self.Y_LUM = singleenh_lum(self.X_lum0, self.is_training, L)
            self.Y_YCRCB = enh_chr(self.Y_LUM, self.X_chr, self.is_training, L)
	else:
	    print("Not recognized the model.")
            self.logfile.write("Not recognized the model.\n")
	    sys.exit(1)		

        tf.summary.image('Y_HAT_LUM'  , self.Y_LUM , 1)
        tf.summary.image('Y_GT_LUM'   , self.Y_LUM_, 1)
        tf.summary.image('X_LUM_LEFT' , self.X_lum0, 1)
        tf.summary.image('Y_HAT_YCrCbr', self.Y_YCRCB, 1)
        
        self.loss_lum = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_LUM_ - self.Y_LUM)
        self.loss_ycrcb = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_YCRCB_ - self.Y_YCRCB)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr_lum = tf_psnr(self.Y_LUM, self.Y_LUM_)
        self.eva_psnr_ycrcb = tf_psnr(self.Y_YCRCB, self.Y_YCRCB_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss_lum + lmbd * self.loss_ycrcb)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
        self.logfile.write("[*] Initialize model successfully...\n")
        sys.stdout.flush()

    def evaluate(self, iter_num, test_data_YL, test_data_XL, test_data_XR,
                 sample_dir, summary_merged, summary_writer):
        print("[*] Evaluating...")
        self.logfile.write("[*] Evaluating...\n")
        sys.stdout.flush()
        
        psnr_sum_lum = 0
        psnr_sum_ycrcb = 0
        for idx in range(len(test_data_YL)):
            im_h, im_w, ch = test_data_YL[idx].shape
            assert ch == 3
            
            # inputs
            X_lum0 = np.zeros((1,im_h,im_w-self.parallax,1))
            X_lum0[0,:,:,0] = (test_data_XL[idx][:,self.parallax:,0].astype(np.float32) / 127.5) - 1

            X_lum  = np.zeros((1,im_h,im_w-self.parallax,self.parallax))
            for p in range(0, self.parallax, 1):
                X_lum[0,:,:,p] = (test_data_XR[idx][:,self.parallax-p:im_w-p,0].astype(np.float32) / 127.5) - 1

            X_chr = np.zeros((1, im_h, im_w-self.parallax, 2))
            X_chr[0,:,:,:] = (test_data_XL[idx][:,self.parallax:,1:].astype(np.float32) / 127.5) - 1
            
            # outpus
            Y_GT_LUM = np.zeros((1,im_h,im_w-self.parallax,1))
            Y_GT_LUM[0,:,:,:] = np.expand_dims((test_data_YL[idx][:,self.parallax:,0].astype(np.float32) / 127.5) - 1, 3)

            Y_GT_YCRCB = np.zeros((1,im_h,im_w-self.parallax,3))
            Y_GT_YCRCB[0,:,:,:] = (test_data_YL[idx][:,self.parallax:,:].astype(np.float32) / 127.5) - 1

            # run the model
            lum_hat_image, ycrcb_hat_image, psnr_summary = self.sess.run(
                       [self.Y_LUM, self.Y_YCRCB, summary_merged],
                       feed_dict={self.X_lum0: X_lum0,
                                  self.X_lum: X_lum,
                                  self.X_chr: X_chr,
                                  self.Y_LUM_: Y_GT_LUM,
                                  self.Y_YCRCB_: Y_GT_YCRCB,
                                  self.is_training: False})

            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = test_data_YL[idx][:,self.parallax:,:].squeeze()
            input_image = test_data_XL[idx][:,self.parallax:,:].squeeze()
            outputimage_lum = (127.5*(lum_hat_image+1)).astype('uint8').squeeze()
            outputimage_ycrcb = (127.5*(ycrcb_hat_image+1)).astype('uint8').squeeze()

            # calculate PSNR
            psnr_lum = cal_psnr(groundtruth[:,:,0].squeeze(), outputimage_lum)
            psnr_ycrcb = cal_psnr(groundtruth, outputimage_ycrcb)
            print("img%d PSNR Lum: %.2f, PSNR YCrCb: %.2f" % (idx + 1, psnr_lum, psnr_ycrcb))
            self.logfile.write("img%d PSNR Lum: %.2f, PSNR YCrCb: %.2f\n" % (idx+1, psnr_lum, psnr_ycrcb))
            sys.stdout.flush()
            psnr_sum_lum += psnr_lum
            psnr_sum_ycrcb += psnr_ycrcb
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, input_image, outputimage_ycrcb)
            save_images(os.path.join(sample_dir, 'test_lum%d_%d.png' % (idx + 1, iter_num)),
                        outputimage_lum)

        avg_psnr_lum = psnr_sum_lum / len(test_data_YL)
        avg_psnr_ycrcb = psnr_sum_ycrcb / len(test_data_YL)
        print("--- Test ---- Average PSNR (Lum, YCrCb) %.2f, %.2f ---" % (avg_psnr_lum, avg_psnr_ycrcb))
        self.logfile.write("--- Test ---- Average PSNR (Lum, YCrCb) %.2f, %.2f ---\n" % (avg_psnr_lum, avg_psnr_ycrcb))
        sys.stdout.flush()

    #def denoise(self, data_gt, data_in):
    #    output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
    #            feed_dict={self.Y_:data_gt, self.X:data_in, self.is_training: False})
    #    return output_clean_image, noisy_image, psnr

    def train(self, data, eval_data_YL, eval_data_XL, eval_data_XR, 
              batch_size, ckpt_dir, epoch, lr, sample_dir,
              log_dir, eval_every_epoch=2):
        data_num = data["X_lum"].shape[0]
        numBatch = int(data_num / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
            self.logfile.write("[*] Model restore success!\n")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
            self.logfile.write("[*] Not find pretrained model\n")
        sys.stdout.flush()
        # make summary
        tf.summary.scalar('loss_lum', self.loss_lum)
        tf.summary.scalar('loss_ycrcb', self.loss_ycrcb)
        tf.summary.scalar('lr'  , self.lr)
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr_lum', self.eva_psnr_lum)
        summary_psnr = tf.summary.scalar('eva_psnr_ycrcb', self.eva_psnr_ycrcb)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        self.logfile.write("[*] Start training, with start epoch %d start iter %d : \n" % (start_epoch, iter_num))
        sys.stdout.flush()
        start_time = time.time()
        self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                      sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)
        for epoch in xrange(start_epoch, epoch):
            blist = random.sample(range(0, numBatch), numBatch)
            for batch_id in xrange(start_step, numBatch):
                i_s = blist[batch_id] * batch_size
                i_e = min((blist[batch_id] + 1 ) * batch_size, data_num)

                batch_X_lum0 = (np.expand_dims(data["X_lum"][i_s:i_e, ..., 0],3).astype(np.float32) / 127.5) - 1
                batch_X_lum  = (data["X_lum"][i_s:i_e, ..., 1:].astype(np.float32) / 127.5) - 1
                batch_X_chr  = (data["X_chr"][i_s:i_e, ...].astype(np.float32) / 127.5) - 1

                batch_Y_LUM = (data["Y_lum"][i_s:i_e, ...].astype(np.float32) / 127.5) - 1
                batch_Y_LUM = np.expand_dims(batch_Y_LUM[:,:,:,0], 3)
                batch_Y_YCRCB = (data["Y_chr"][i_s:i_e, ...].astype(np.float32) / 127.5) - 1

                _, loss_lum, loss_ycrcb, summary = self.sess.run(
                        [self.train_op, self.loss_lum, self.loss_ycrcb, merged],
                        feed_dict={self.X_lum: batch_X_lum,
                                   self.X_lum0: batch_X_lum0,
                                   self.X_chr: batch_X_chr,
                                   self.Y_LUM_: batch_Y_LUM,
                                   self.Y_YCRCB_: batch_Y_YCRCB,
                                   self.lr: lr[epoch], self.is_training: True})
                
                if (batch_id+1) % 1000 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_lum: %.6f, loss_ycrcb: %.6f"
                          % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss_lum, loss_ycrcb))
                    self.logfile.write("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_lum: %.6f, loss_ycrcb: %.6f\n" 
                          % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss_lum, loss_ycrcb))
                    sys.stdout.flush()
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                              sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")
        self.logfile.write("[*] Finish training.\n")
        self.logfile.close()
        sys.stdout.flush()

    def save(self, iter_num, ckpt_dir, model_name='dualenh'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.logfile.write("[*] Saving model...\n")
        sys.stdout.flush()
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        self.logfile.write("[*] Reading checkpoint...\n")
        sys.stdout.flush()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

#    def test(self, test_files, ckpt_dir, save_dir):
#        """Test CNN_PAR"""
#        # init variables
#        tf.initialize_all_variables().run()
#        assert len(test_files) != 0, 'No testing data!'
#        load_model_status, global_step = self.load(ckpt_dir)
#        assert load_model_status == True, '[!] Load weights FAILED...'
#        print(" [*] Load weights SUCCESS...")
#        sys.stdout.flush()
#        psnr_sum = 0
#        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
#        sys.stdout.flush()
#        for idx in xrange(len(test_files)):
#            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
#            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
#                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
#            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
#            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
#            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
#            # calculate PSNR
#            psnr = cal_psnr(groundtruth, outputimage)
#            print("img%d PSNR: %.2f" % (idx, psnr))
#            sys.stdout.flush()
#            psnr_sum += psnr
#            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
#            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
#        avg_psnr = psnr_sum / len(test_files)
#        print("--- Average PSNR %.2f ---" % avg_psnr)
#        sys.stdout.flush()

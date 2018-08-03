import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import os
from glob import glob
from ops import *
from utils import *
import cv2

class SRGAN:
    model_name = 'SRGAN'
    def __init__(self, config, batch_size=1, input_size=256, output_size=256, input_channels=3, sess=None):
        self.input_size = input_size
        self.output_size = output_size
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.images_norm = True
        self.config = config
        self.sess = sess
        
    def generator(self, input_x, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            # down_sample here
            # input_x = down_sample_layer(input_x)

            with slim.arg_scope([slim.conv2d_transpose],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer=None,
                                activation_fn=None,
                                normalizer_fn=None,
                                padding='SAME'):
                conv1 = tf.nn.relu(slim.conv2d_transpose(input_x, 64, 3, 1, scope='g_conv1'))
                print(conv1)
                shortcut = conv1
                # res_block(input_x, out_channels=64, k=3, s=1, scope='res_block'):
                res1 = res_block(conv1, 64, 3, 1, scope='g_res1')
                res2 = res_block(res1, 64, 3, 1, scope='g_res2')
                res3 = res_block(res2, 64, 3, 1, scope='g_res3')
                res4 = res_block(res3, 64, 3, 1, scope='g_res4')
                res5 = res_block(res4, 64, 3, 1, scope='g_res5')
                
                conv2 = slim.batch_norm(slim.conv2d_transpose(res5, 64, 3, 1, scope='g_conv2'), scope='g_bn_conv2')
                print(conv2)
                conv2_out = shortcut+conv2
                print(conv2_out) 
                # pixel_shuffle_layer(x, r, n_split):
                # conv3 = slim.conv2d_transpose(conv2_out, 256, 3, 1, scope='g_conv3')
                # print(conv3)
                # shuffle1 = tf.nn.relu(pixel_shuffle_layer(conv3, 2, 64)) #64*2*2
                # print(shuffle1)
                # conv4 = slim.conv2d_transpose(shuffle1, 256, 3, 1, scope='g_conv4')
                # shuffle2 = tf.nn.relu(pixel_shuffle_layer(conv4, 2, 64))
                # print(shuffle2)

                # conv3 = tf.nn.relu(slim.conv2d_transpose(conv2_out, 256, 3, 1, scope='g_conv3'))
                # conv4 = tf.nn.relu(slim.conv2d_transpose(conv3, 256, 3, 1, scope='g_conv4'))
                # conv5 = slim.conv2d_transpose(conv4, 3, 3, 1, scope='g_conv5')

                conv6 = tf.nn.relu(slim.conv2d(conv2_out, 64, 9, 1, padding='VALID', scope='g_conv6'))
                conv7 = tf.nn.relu(slim.conv2d(conv6, 32, 1, 1, padding='VALID', scope='g_conv7'))
                conv8 = slim.conv2d(conv7, 3, 5, 1, padding='VALID', scope='g_conv8')

                self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                return tf.nn.tanh(conv8)
            
    def discriminator(self, input_x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer = None,
                                activation_fn=None,
                                normalizer_fn=None):
                                
                conv1 = leaky_relu(slim.conv2d(input_x, 64, 3, 1, scope='d_conv1'))
                conv1_1 = leaky_relu(slim.batch_norm(slim.conv2d(conv1, 64, 3, 2, scope='d_conv1_1'), scope='d_bn_conv1_1'))

                conv2 = leaky_relu(slim.batch_norm(slim.conv2d(conv1_1, 128, 3, 1, scope='d_conv2'), scope='d_bn_conv2'))
                conv2_1 = leaky_relu(slim.batch_norm(slim.conv2d(conv2, 128, 3, 2, scope='d_conv2_1'), scope='d_bn_conv2_1'))
                
                conv3 = leaky_relu(slim.batch_norm(slim.conv2d(conv2_1, 256, 3, 1, scope='d_conv3'), scope='d_bn_conv3'))
                conv3_1 = leaky_relu(slim.batch_norm(slim.conv2d(conv3, 256, 3, 2, scope='d_conv3_1'), scope='d_bn_conv3_1'))

                conv4 = leaky_relu(slim.batch_norm(slim.conv2d(conv3_1, 512, 3, 1, scope='d_conv4'), scope='d_bn_conv4'))
                conv4_1 = leaky_relu(slim.batch_norm(slim.conv2d(conv4, 512, 3, 2, scope='d_conv4_1'), scope='d_bn_conv4_1'))

                conv_flat = tf.reshape(conv4_1, [-1, 1])
                dense1 = leaky_relu(slim.fully_connected(conv_flat, 1024, scope='d_dense1'))
                dense2 = slim.fully_connected(dense1, 1, scope='d_dense2')
                
                self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
                return dense2, tf.nn.sigmoid(dense2)


    def build_model(self):
        self.input_target = tf.placeholder(tf.float32, [None, self.output_size, self.output_size, self.input_channels], name='input_target')
        self.input_source = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.input_channels], name='input_source')
        
        # self.input_source = down_sample_layer(self.input_target)

        self.real = self.input_target
        # warning

        self.fake = self.generator(self.input_source, reuse=False)
        self.psnr = PSNR(self.real, self.fake)
        self.d_loss, self.g_loss, self.content_loss = self.inference_loss(self.real, self.fake)
        print('d, g_loss')
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.d_loss, var_list=self.d_vars)
        print('d_optim')
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.g_loss, var_list=self.g_vars)
        print('g_optim')
        self.srres_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.content_loss, var_list=self.g_vars)
        print('srres_optim')
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.content_loss_summary = tf.summary.scalar('content_loss', self.content_loss)
        self.psnr_summary = tf.summary.scalar('psnr', self.psnr)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph) 
        self.saver = tf.train.Saver()
        print('builded model...') 

    def inference_loss(self, real, fake):
        # MSE content loss
        def inference_mse_content_loss(real, fake):
            return tf.reduce_mean(tf.square(real-fake))
            
        def inference_adversarial_loss(x, y, w=1, type_='gan'):
            if type_=='gan':
                try:
                    return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
                except:
                    return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            elif type_=='lsgan':
                return w*(x-y)**2
            else:
                raise ValueError('no {} loss type'.format(type_))
        
        content_loss = inference_mse_content_loss(real, fake)
        d_real_logits, d_real_sigmoid = self.discriminator(real, reuse=False)
        d_fake_logits, d_fake_sigmoid = self.discriminator(fake, reuse=True)
        d_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_real_logits, tf.ones_like(d_real_sigmoid)))
        d_real_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.zeros_like(d_fake_sigmoid)))
        g_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.ones_like(d_fake_sigmoid)))
        
        d_loss =  self.config.lambd*(d_fake_loss+d_real_loss)
        g_loss = content_loss + self.config.lambd*g_fake_loss
        
        return d_loss, g_loss, content_loss
        
    def train(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # data/train/*.*
        data = glob(os.path.join(self.config.dataset_dir, 'train', self.config.train_set, '*.*'))

        bool_check, counter = self.load_model(self.config.checkpoint_dir)
        if bool_check:
            print('[!!!] load model successfully')
            counter = counter + 1
        else:
            print('[***] fail to load model')
            counter = 1

        start_time = time.time()
        for epoch in range(self.config.epoches):
            np.random.shuffle(data)
            for file_name in data:
                sav_images = get_images(file_name, self.config.is_crop, self.config.fine_size, images_norm=False)
                images = np.asarray(sav_images)
                images = images[:images.shape[0]//8 * 8, :, :, :]
                batches = images.reshape([-1, self.batch_size, self.input_size, self.input_size, 3])

                for batch_x in batches:
                    batch_x = [blur_images(imgs, self.images_norm, self.output_size) for imgs in batch_x]
                    batch_x_input = [input_x[0] for input_x in batch_x]
                    batch_x_sample = [sample_x[1] for sample_x in batch_x]
                    batch_x_input = np.array(batch_x_input).astype(np.float32)
                    batch_x_sample = np.array(batch_x_sample).astype(np.float32)

                    if counter < 2e4:
                        _, content_loss, psnr = self.sess.run([self.srres_optim, self.content_loss, self.psnr],
                                                              feed_dict={self.input_target: batch_x_sample,
                                                                         self.input_source: batch_x_input})
                        end_time = time.time()
                        print(
                            'epoch{}:total_time:{:.4f},content_loss:{:4f},psnr:{:.4f}'.format(epoch, end_time - start_time,
                                                                                                     content_loss, psnr))
                    else:
                        _, d_loss, summaries = self.sess.run([self.d_optim, self.d_loss, self.summaries],
                                                             feed_dict={self.input_target: batch_x_sample,
                                                                        self.input_source: batch_x_input})
                        _, g_loss, psnr, summaries = self.sess.run([self.g_optim, self.g_loss, self.psnr, self.summaries],
                                                                   feed_dict={self.input_target: batch_x_sample,
                                                                              self.input_source: batch_x_input})
                        end_time = time.time()
                        print('epoch{}:total_time:{:.4f},d_loss:{:.4f},g_loss:{:4f},psnr:{:.4f}'.format(epoch, end_time - start_time,
                                                                                                               d_loss,
                                                                                                               g_loss,
                                                                                                               psnr))

                    if np.mod(counter, 100) == 0:
                        self.sample(epoch)
                    if np.mod(counter, 500) == 0:
                        self.save_model(self.config.checkpoint_dir, counter)
                    counter = counter + 1
            
    def sample(self,epoch):
        # here I use set5 as the valuation sets
        data = glob(os.path.join(self.config.dataset_dir, 'val', self.config.val_set, '*.*'))
        data = data[0]

        h_, w_, input_, sample_ = get_sample_image(data, self.input_size, self.output_size, self.images_norm)

        # save_images(input_, [h_, w_],
        #             'input.png'.format(self.config.sample_dir, self.config.val_set, epoch, idx),
        #             self.images_norm)
        # save_images(sample_, [h_, w_],
        #             'output.png'.format(self.config.sample_dir, self.config.val_set, epoch, idx),
        #             self.images_norm)

        sample_images, psnr, input_source = self.sess.run([self.fake, self.psnr, self.input_source], feed_dict={self.input_target:sample_, self.input_source:input_})
        
        save_images(sample_images, [h_,w_], './{}/{}_sample_{}.png'.format(self.config.sample_dir, self.config.val_set,epoch), self.images_norm)
        save_images(input_source, [h_,w_], './{}/{}_input_{}.png'.format(self.config.sample_dir, self.config.val_set,epoch), self.images_norm)
        print('---------------------------------------')
        print('epoch{}:psnr{:.4f}'.format(epoch, psnr))
        print('---------------------------------------')
    
    def test(self):
        print('testing')
        bool_check, counter = self.load_model(self.config.checkpoint_dir)
        if bool_check:
            print('[!!!] load model successfully')
            counter = counter + 1
        else:
            print('[***] fail to load model')
            counter = 1
        
        test = glob(os.path.join(self.config.dataset_dir, 'test', self.config.test_set, '*.*'))
        batch_files = test[:self.batch_size]
        batch_x = [get_images(batch_file, True, self.config.fine_size, self.images_norm) for batch_file in batch_files]
        batchs = np.array(batch_x).astype(np.float32)

        batch_x = [blur_images(imgs, self.images_norm, self.output_size) for imgs in batch_x]
        batch_x_input = [input_x[0] for input_x in batch_x]
        batch_x_sample = [sample_x[1] for sample_x in batch_x]
        batch_x_input = np.array(batch_x_input).astype(np.float32)
        batch_x_sample = np.array(batch_x_sample).astype(np.float32)
        
        sample_images, input_sources = self.sess.run([self.fake, self.input_source], feed_dict={self.input_target:batch_x_sample, self.input_source:batch_x_input})
        #images = np.concatenate([sample_images, batchs], 2)
        for i in range(len(batch_x)):
            batch = np.expand_dims(batchs[i],0)
            sample_image = np.expand_dims(sample_images[i],0)
            input_source = np.expand_dims(input_sources[i],0)
            save_images(batch, [1,1], './{}/{}_gt_hr_{}.png'.format(self.config.test_dir, self.config.test_set,i))
            save_images(sample_image, [1,1], './{}/{}_test_hr_{}.png'.format(self.config.test_dir, self.config.test_set,i))
            save_images(input_source, [1,1], './{}/{}_gt_lr_{}.png'.format(self.config.test_dir, self.config.test_set,i))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.config.dataset_name,
            self.batch_size)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.config.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_model(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.config.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__=='__main__':
    srgan = SRGAN(None)
    # a = tf.random_normal([8,64,64,3])
    # out = srgan.generator(a)
    # out,_ = srgan.discriminator(a)
    # print(out)

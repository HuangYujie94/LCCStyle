# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import model
import time
import vlib.plot as plot
import vlib.save_images as save_img
import vlib.load_data as load_data
import vgg_simple as vgg
import scipy.misc as scm

import model

slim = tf.contrib.slim

def load_test_img(img_path):
    style_img = tf.read_file(img_path)

    style_img = tf.image.decode_jpeg(style_img, 3)
    shape = tf.shape(style_img)

    style_img = tf.image.resize_images(style_img, [shape[0], shape[1]])

    images = tf.expand_dims(style_img, 0)
    return images


class Train(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = 1
        self.img_size = 300
        self.args = args

    def build_model(self):
        data_path = self.args.train_content_path

        imgs = load_data.get_loader(data_path, self.batch_size, self.img_size)

        style_imgs=load_data.get_loader1(self.args.train_style_path, self.batch_size, self.img_size)


        with slim.arg_scope(model.arg_scope()):

            gen_img, variables = model.LCCStyleFC(imgs, style_imgs, reuse=False, name='LCCStyleFC')

            with slim.arg_scope(vgg.vgg_arg_scope()):
                #histogram loss
                hist_loss = model.histloss(gen_img, style_imgs)
                
                gen_img_processed = [load_data.img_process(image, True) for image in tf.unstack(gen_img, axis=0, num=self.batch_size)]
                imgs_processed = [load_data.img_process(image, True) for image in tf.unstack(imgs, axis=0, num=self.batch_size)]
                style_imgs_processed = [load_data.img_process(image, True) for image in tf.unstack(style_imgs, axis=0, num=self.batch_size)]
                f1, f2, f3, f4, f5, exclude = vgg.vgg_19(tf.concat([gen_img_processed, imgs_processed, style_imgs_processed], axis=0))
                gen_f, img_f, _ = tf.split(f3, 3, 0)
                
                #content loss
                content_loss = tf.nn.l2_loss(gen_f - img_f) / tf.to_float(tf.size(gen_f))
                
                #style loss
                style_loss = model.styleloss(f1, f2, f3, f4,f5)

                # load vgg model
                vgg_model_path = self.args.vgg_model19
                vgg_vars = slim.get_variables_to_restore(include=['vgg_19'], exclude=exclude)
                init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
                init_fn(self.sess)
                print('vgg19 s weights load done')

            self.gen_img = gen_img
            self.imgs = imgs
            self.style_imgs = style_imgs
            self.hist_loss = hist_loss

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.content_loss = content_loss
            self.style_loss = style_loss*self.args.style_w
            self.loss = (self.content_loss + self.style_loss) + (self.hist_loss)
            self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss, global_step=self.global_step, var_list=variables)

        all_var = tf.global_variables()
        init_var = [v for v in all_var if 'vgg_16' not in v.name and 'vgg_19' not in v.name]
        init = tf.variables_initializer(var_list=init_var)
        self.sess.run(init)

        self.save = tf.train.Saver(var_list=variables)

    def train(self):
        print ('start to training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        
        try:
            while not coord.should_stop():
                _, loss, step, cl, sl, hl = self.sess.run([self.opt, self.loss, self.global_step, self.content_loss, self.style_loss, self.hist_loss])
                    
                if step%10000 == 0:
                    all_img = tf.concat([self.imgs, self.style_imgs, self.gen_img],0)
                    gen_img = self.sess.run(all_img)
                    if not os.path.exists('gen_img_LCCStyleFC'):
                        os.mkdir('gen_img_LCCStyleFC')
                    save_img.save_images(gen_img, './gen_img_LCCStyleFC/{0}.jpg'.format(step/10000))

                print ('[{}/1600000],loss:{}, content:{},style:{},hist:{}'.format(step, loss, cl, sl, hl))

                if step % 80000 ==0:
                    if not os.path.exists('model_LCCStyleFC'):
                        os.mkdir('model_LCCStyleFC')
                    self.save.save(self.sess, './model_LCCStyleFC/general{}.ckpt'.format(step/80000))
                if step >= 1600000:
                    break

        
        except tf.errors.OutOfRangeError:
                self.save.save(sess, os.path.join(os.getcwd(), 'fast-style-model.ckpt-done'))
        finally:
            coord.request_stop()
        coord.join(threads)

    def test(self):
        print ('test model')
        test_img_path = self.args.test_data_path
        test_style_path=self.args.test_style_path
        test_img_c = load_test_img(test_img_path)
        test_img_s = load_test_img(test_style_path)
        with slim.arg_scope(model.arg_scope()):

            gen_img, _ = model.LCCStyleFC(test_img_c, test_img_s, reuse=False, name='LCCStyleFC')

            # load model
            model_path = self.args.transfer_model

            vars = slim.get_variables_to_restore(include=['LCCStyleFC'])
            init_fn = slim.assign_from_checkpoint_fn(model_path, vars)
            init_fn(self.sess)
            print('network weights load done')

            gen_img = self.sess.run(gen_img)
            save_img.save_images(gen_img, self.args.new_img_name)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-is_training', help='train or test', type=bool, default=False)
parser.add_argument('-vgg_model19', help='the path of pretrained vgg model19', type=str,
                    default='VGG19/vgg_19.ckpt')
parser.add_argument('-transfer_model', help='the path of the model', type=str,
                    default='model_saved/general800.0.ckpt')
parser.add_argument('-train_content_path', help='the path of the content image folder for train', type=str,
                    default='content_train/')
parser.add_argument('-train_style_path', help='the path of the style image folder for train', type=str, default='style_train/')
parser.add_argument('-test_data_path', help='the path of test content image', type=str, default='test_content.jpg')
parser.add_argument('-new_img_name', help='the path of stylized image', type=str, default='transfer.jpg')
parser.add_argument('-style_w', help='the weight of style loss', type=float, default=5)
parser.add_argument('-test_style_path', help='the path of the style image when test', type=str, default='style_test.jpg')

args = parser.parse_args()

if __name__ == '__main__':

    with tf.Session() as sess:
        Model = Train(sess, args)
        is_training = args.is_training

        if is_training:
            Model.build_model()
            Model.train()
        else:
            Model.test()
            print('the test')

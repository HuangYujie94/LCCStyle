from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


def vgg_16(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=False):
  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    out = []
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

      out1 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

      out2 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      out3 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')

      out4 = net
      
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

      out5 = net
      exclude = ['vgg_16/fc6', 'vgg_16/pool5','vgg_16/fc7','vgg_16/global_pool','vgg_16/fc8/squeezed','vgg_16/fc8']

      return out1, out2, out3, out4, out5, exclude

def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=False):
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    out = []
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

      out1 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

      out2 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')

      out3 = net

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')

      out4 = net
      
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')

      out5 = net
      exclude = ['vgg_19/fc6','vgg_19/pool5','vgg_19/fc7','vgg_19/global_pool','vgg_19/fc8/squeezed','vgg_19/fc8']

      return out1, out2, out3, out4, out5, exclude

def vgg_191(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False,
           reuse=False):
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    out = []
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.conv2d(inputs, 64, [3,3], scope='conv1/conv1_1')

      out1 = net
      
      net = slim.conv2d(net, 64, [3,3], scope='conv1/conv1_2')

      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.conv2d(net, 128, [3,3], scope='conv2/conv2_1')
      out2 = net
      net = slim.conv2d(net, 128, [3,3], scope='conv2/conv2_2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      #net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.conv2d(net, 256, [3,3], scope='conv3/conv3_1')
      out3 = net
      net = slim.conv2d(net, 256, [3,3], scope='conv3/conv3_2')
      net = slim.conv2d(net, 256, [3,3], scope='conv3/conv3_3')
      net = slim.conv2d(net, 256, [3,3], scope='conv3/conv3_4')
      outc = net

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      #net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.conv2d(net, 512, [3,3], scope='conv4/conv4_1')
      out4 = net
      net = slim.conv2d(net, 512, [3,3], scope='conv4/conv4_2')
      net = slim.conv2d(net, 512, [3,3], scope='conv4/conv4_3')
      net = slim.conv2d(net, 512, [3,3], scope='conv4/conv4_4')
      
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      #net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.conv2d(net, 512, [3,3], scope='conv5/conv5_1')
      out5 = net
      net = slim.conv2d(net, 512, [3,3], scope='conv5/conv5_2')
      net = slim.conv2d(net, 512, [3,3], scope='conv5/conv5_3')
      net = slim.conv2d(net, 512, [3,3], scope='conv5/conv5_4')
      exclude = ['vgg_19/fc6','vgg_19/pool5','vgg_19/fc7','vgg_19/global_pool','vgg_19/fc8/squeezed','vgg_19/fc8']

      return out1, out2, out3, outc, out4, out5, exclude


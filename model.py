import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME') as arg_sc:
            return arg_sc

def img_scale(x, scale):
    weight = x.get_shape()[1].value
    height = x.get_shape()[2].value

    try:
        out = tf.image.resize_nearest_neighbor(x, size=(weight*scale, height*scale))
    except:
        out = tf.image.resize_images(x, size=[weight*scale, height*scale])
    return out

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
  
  
def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
  
  
def relu(x):
    return tf.nn.relu(x)


def LCCStyleFC(imgs,styles, reuse, name, is_train=True):
    #reflect padding
    imgs = tf.pad(imgs, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope(name, reuse=reuse) as vs:
        
        # Style Encoder
        out11 = slim.conv2d(styles, 32, [3, 3], scope='conv11')
        out11 = relu(out11)
        out11_1 = slim.conv2d(out11, 32, [3, 3], scope='conv11_1')
        out11_1 = relu(out11_1)
        out11_2 = slim.conv2d(out11_1, 32, [3, 3], scope='conv11_2')
        out11_2 = relu(out11_2)

        out21_1 = slim.conv2d(out11_2, 64, [3, 3], stride=2, scope='conv21')
        out21_1 = relu(out21_1)
        out21_1_1 = slim.conv2d(out21_1, 64, [3, 3], stride=1, scope='conv21_1')
        out21_1_1 = relu(out21_1_1)
        out21_1_2 = slim.conv2d(out21_1_1, 64, [3, 3], stride=1, scope='conv21_2')
        out21_1_2 = relu(out21_1_2)

        out21_2 = slim.conv2d(out21_1_2, 128, [3, 3], stride=2, scope='conv31')
        out21_2 = relu(out21_2)
        out21_2_1 = slim.conv2d(out21_2, 128, [3, 3], stride=1, scope='conv31_1')
        out21_2_1 = relu(out21_2_1)
        out21_2_2 = slim.conv2d(out21_2_1, 128, [3, 3], stride=1, scope='conv31_2')
        out21_2_2 = relu(out21_2_2)
        
        #Skill Learning Network
        #third layer
        gap = gram(out21_2_2)
        gap = tf.layers.flatten(gap)
        batch_size = gap.get_shape().as_list()[0]
        w = slim.fully_connected(gap, 4000, activation_fn = None, scope = 'fc1')
        w = relu(w)
        w = slim.fully_connected(w, 4000, activation_fn = None, scope = 'fc2')
        w = relu(w)
        num1 = int(3*3*32*128)
        num = int(num1*2)
        w = slim.fully_connected(w, num, activation_fn = None, scope = 'fc3')
        wconv1 = tf.slice(w,[0, 0], [batch_size, num1])
        wconv1 = tf.reshape(wconv1,[3, 3, 128, 32])
        wconv2 = tf.slice(w,[0, num1], [batch_size, num1])
        wconv2 = tf.reshape(wconv2,[3, 3, 32, 128])
        #second layer
        gap = gram(out21_1_2)
        gap = tf.layers.flatten(gap)
        batch_size = gap.get_shape().as_list()[0]
        w = slim.fully_connected(gap, 1000, activation_fn = None, scope = 'fc1_1')
        w = relu(w)
        w = slim.fully_connected(w, 1000, activation_fn = None, scope = 'fc2_1')
        w = relu(w)
        num1 = int(3*3*16*64)
        num = int(num1*2)
        w = slim.fully_connected(w, num, activation_fn = None, scope = 'fc3_1')
        wconv1_1 = tf.slice(w,[0, 0], [batch_size, num1])
        wconv1_1 = tf.reshape(wconv1_1,[3, 3, 64, 16])
        wconv2_1 = tf.slice(w,[0, num1], [batch_size, num1])
        wconv2_1 = tf.reshape(wconv2_1,[3, 3, 16, 64])
        #first layer
        gap = gram(out11_2)
        gap = tf.layers.flatten(gap)
        batch_size = gap.get_shape().as_list()[0]
        w = slim.fully_connected(gap, 250, activation_fn = None, scope = 'fc1_2')
        w = relu(w)
        w = slim.fully_connected(w, 250, activation_fn = None, scope = 'fc2_2')
        w = relu(w)
        num1 = int(3*3*8*32)
        num = int(num1*2)
        w = slim.fully_connected(w, num, activation_fn = None, scope = 'fc3_2')
        wconv1_2 = tf.slice(w,[0, 0], [batch_size, num1])
        wconv1_2 = tf.reshape(wconv1_2,[3, 3, 32, 8])
        wconv2_2 = tf.slice(w,[0, num1], [batch_size, num1])
        wconv2_2 = tf.reshape(wconv2_2,[3, 3, 8, 32])
        
        # Content Encoder 
        out1 = slim.conv2d(imgs, 32, [3, 3], scope='conv1')
        out1 = relu(instance_norm(out1))

        out2_1 = slim.conv2d(out1, 64, [3, 3], stride=2, scope='conv2')
        out2_1 = relu(instance_norm(out2_1))

        out2_2 = slim.conv2d(out2_1, 128, [3, 3], stride=2, scope='conv3')
        out2_2 = relu(instance_norm(out2_2))
        
        # The first Drawing Network
        out32 = conv2d(out2_2,wconv1)
        out32 = relu(out32)
        out32 = conv2d(out32,wconv2)
        out32 = relu(out32)
        
        # The Decoder embeded with two Drawing Network
        #The first convolution layer of the Decoder
        out4 = slim.conv2d(out32, 64, [3, 3], scope='conv3_1')
        out4 = relu(out4)
        
        #The second Drawing Network
        out4 = conv2d(out4,wconv1_1)
        out4 = relu(out4)
        out4 = conv2d(out4,wconv2_1)
        out4 = relu(out4)

        #The first upsample layer of the Decoder
        out4 = img_scale(out4, 2)
        
        #The second convolution layer of the Decoder
        out4 = slim.conv2d(out4, 32, [3, 3], scope='conv2_1')
        out4 = relu(out4)
        
        #The third Drawing Network
        out4 = conv2d(out4,wconv1_2)
        out4 = relu(out4)
        out4 = conv2d(out4,wconv2_2)
        out4 = relu(out4)

        #The second upsample layer of the Decoder
        out4 = img_scale(out4, 2)
        
        #The third convolution layer of the Decoder
        out = slim.conv2d(out4, 3, [3, 3], scope='conv1_1')
        
        #0-255
        maxval = tf.reduce_max(out, axis=[1,2], keep_dims=True)
        dela = 1e-9
        minval = tf.reduce_min(out, axis=[1,2], keep_dims=True)
        out = tf.div(out-minval, maxval-minval)
        out=out*255.0

        #restore the image size
        height = out.get_shape()[1].value
        width = out.get_shape()[2].value
        out = tf.image.crop_to_bounding_box(out, 10, 10, height-20, width-20)
        
        variables = tf.contrib.framework.get_variables(vs)

    return out, variables



def LCCStyleConv1d(imgs,styles, reuse, name, is_train=True):
    #reflect padding
    imgs = tf.pad(imgs, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope(name, reuse=reuse) as vs:
        
        # Style Encoder
        out11 = slim.conv2d(styles, 32, [3, 3], activation_fn = None, scope='conv11')
        out11 = relu(out11)
        out11_1 = slim.conv2d(out11, 32, [3, 3], activation_fn = None, scope='conv11_1')
        out11_1 = relu(out11_1)
        out11_2 = slim.conv2d(out11_1, 32, [3, 3], activation_fn = None, scope='conv11_2')
        out11_2 = relu(out11_2)

        out21_1 = slim.conv2d(out11_2, 64, [3, 3], stride=2, activation_fn = None, scope='conv21')
        out21_1 = relu(out21_1)
        out21_1_1 = slim.conv2d(out21_1, 64, [3, 3], stride=1, activation_fn = None, scope='conv21_1')
        out21_1_1 = relu(out21_1_1)
        out21_1_2 = slim.conv2d(out21_1_1, 64, [3, 3], stride=1, activation_fn = None, scope='conv21_2')
        out21_1_2 = relu(out21_1_2)

        out21_2 = slim.conv2d(out21_1_2, 128, [3, 3], stride=2, activation_fn = None, scope='conv31')
        out21_2 = relu(out21_2)
        out21_2_1 = slim.conv2d(out21_2, 128, [3, 3], stride=1, activation_fn = None, scope='conv31_1')
        out21_2_1 = relu(out21_2_1)
        out21_2_2 = slim.conv2d(out21_2_1, 128, [3, 3], stride=1, activation_fn = None, scope='conv31_2')
        out21_2_2 = relu(out21_2_2)
        
        #Skill Learning Network
        #third layer
        gap = gram(out21_2_2)
        w = tf.layers.conv1d(gap, 128*9, 1, name='conv1d')
        wconv = tf.reshape(w,[3, 3, 128, 128])
        #second layer
        gap = gram(out21_1_2)
        w = tf.layers.conv1d(gap, 64*9, 1, name='conv1d_1')
        wconv_1 = tf.reshape(w,[3, 3, 64, 64])
        #first layer
        gap = gram(out11_2)
        w = tf.layers.conv1d(gap, 32*9, 1, name='conv1d_2')
        wconv_2 = tf.reshape(w,[3, 3, 32, 32])
        
        # Content Encoder 
        out1 = slim.conv2d(imgs, 32, [3, 3], activation_fn = None, scope='conv1')
        out1 = relu(instance_norm(out1))

        out2_1 = slim.conv2d(out1, 64, [3, 3], stride=2, activation_fn = None, scope='conv2')
        out2_1 = relu(instance_norm(out2_1))

        out2_2 = slim.conv2d(out2_1, 128, [3, 3], stride=2, activation_fn = None, scope='conv3')
        out2_2 = relu(instance_norm(out2_2))
        
        # The first Drawing Network
        out32 = conv2d(out2_2,wconv)
        out32 = relu(out32)
        
        # The Decoder embeded with two Drawing Network
        #The first convolution layer of the Decoder
        out4 = slim.conv2d(out32, 64, [3, 3], activation_fn = None, scope='conv3_1')
        out4 = relu(out4)
        
        #The second Drawing Network
        out4 = conv2d(out4,wconv_1)
        out4 = relu(out4)

        #The first upsample layer of the Decoder
        out4 = img_scale(out4, 2)
        
        #The second convolution layer of the Decoder
        out4 = slim.conv2d(out4, 32, [3, 3], activation_fn = None, scope='conv2_1')
        out4 = relu(out4)
        
        #The third Drawing Network
        out4 = conv2d(out4,wconv_2)
        out4 = relu(out4)

        #The second upsample layer of the Decoder
        out4 = img_scale(out4, 2)
        
        #The third convolution layer of the Decoder
        out = slim.conv2d(out4, 3, [3, 3], activation_fn = None, scope='conv1_1')
        out = tf.nn.tanh(out)
        #0-255

        out = (out + 1) * 127.5

        #restore the image size
        height = out.get_shape()[1].value
        width = out.get_shape()[2].value
        out = tf.image.crop_to_bounding_box(out, 10, 10, height-20, width-20)
        
        variables = tf.contrib.framework.get_variables(vs)

    return out, variables 

def styleloss(f1, f2, f3, f4, f5):
    gen_f, _, style_f = tf.split(f1, 3, 0)
    gmean, gvar = tf.nn.moments(gen_f, [1, 2])
    smean, svar = tf.nn.moments(style_f, [1, 2])
    size = tf.size(gmean)
    style_loss = tf.nn.l2_loss(gmean - smean)*2 / tf.to_float(size)+tf.nn.l2_loss(tf.sqrt(gvar) - tf.sqrt(svar))*2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f2, 3, 0)
    gmean, gvar = tf.nn.moments(gen_f, [1, 2])
    smean, svar = tf.nn.moments(style_f, [1, 2])
    size = tf.size(gmean)
    style_loss = style_loss + tf.nn.l2_loss(gmean - smean)*2 / tf.to_float(size)+tf.nn.l2_loss(tf.sqrt(gvar) - tf.sqrt(svar))*2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f3, 3, 0)
    gmean, gvar = tf.nn.moments(gen_f, [1, 2])
    smean, svar = tf.nn.moments(style_f, [1, 2])
    size = tf.size(gmean)
    style_loss = style_loss + tf.nn.l2_loss(gmean - smean)*2 / tf.to_float(size)+tf.nn.l2_loss(tf.sqrt(gvar) - tf.sqrt(svar))*2 / tf.to_float(size)

    gen_f, _, style_f = tf.split(f4, 3, 0)
    gmean, gvar = tf.nn.moments(gen_f, [1, 2])
    smean, svar = tf.nn.moments(style_f, [1, 2])
    size = tf.size(gmean)
    style_loss = style_loss + tf.nn.l2_loss(gmean - smean)*2 / tf.to_float(size)+tf.nn.l2_loss(tf.sqrt(gvar) - tf.sqrt(svar))*2 / tf.to_float(size)
    
    gen_f, _, style_f = tf.split(f5, 3, 0)
    gmean, gvar = tf.nn.moments(gen_f, [1, 2])
    smean, svar = tf.nn.moments(style_f, [1, 2])
    size = tf.size(gmean)
    style_loss = style_loss + tf.nn.l2_loss(gmean - smean)*2 / tf.to_float(size)+tf.nn.l2_loss(tf.sqrt(gvar) - tf.sqrt(svar))*2 / tf.to_float(size)

    return style_loss


def histloss(gen_img, style):
    gen_img_r, gen_img_g, gen_img_b = tf.split(gen_img, 3, 3)
    gen_img_histr = tf.histogram_fixed_width(gen_img_r, [0.0, 255.0], nbins = 256)
    gen_img_histg = tf.histogram_fixed_width(gen_img_g, [0.0, 255.0], nbins = 256)
    gen_img_histb = tf.histogram_fixed_width(gen_img_b, [0.0, 255.0], nbins = 256)
    gen_img_hist = tf.concat([gen_img_histr, gen_img_histg, gen_img_histb], axis = 0)
    gen_img_hist = tf.to_float(gen_img_hist, name='ToFlaot')
    style_r, style_g, style_b = tf.split(style, 3, 3)
    style_histr = tf.histogram_fixed_width(style_r, [0.0, 255.0], nbins = 256)
    style_histg = tf.histogram_fixed_width(style_g, [0.0, 255.0], nbins = 256)
    style_histb = tf.histogram_fixed_width(style_b, [0.0, 255.0], nbins = 256)
    style_hist = tf.concat([style_histr, style_histg, style_histb], axis = 0)
    style_hist = tf.to_float(style_hist, name='ToFlaot')
    hist_loss = tf.reduce_mean(tf.multiply(gen_img_hist, style_hist)) #/ (256*3)
    return hist_loss
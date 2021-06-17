from __future__ import print_function, division
import os
import pathlib
import scipy.io as scio
from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from PIL import Image

from utils import load_images_and_flow_1clip, generate_mask
from SE_Net import squeeze_excitation_layer as SElayer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

p_keep = 0.7



def sample_images(dataset_name, in_flows, in_frames, out_flows, out_frames, epoch, batch_i, train):
    def scale_range(img):
        for i in range(img.shape[-1]):
            img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        return img

    assert len(np.unique([len(in_flows), len(in_frames), len(out_flows), len(out_frames)])) == 1
    saved_sample_path = 'generated/%s/%s' % (dataset_name, 'train' if train else 'test')
    os.makedirs(saved_sample_path, exist_ok=True)
    r, c = 4, len(in_flows)

    gen_imgs = np.concatenate([0.5*in_frames+0.5, 0.5*out_frames+0.5, 0.5*in_flows+0.5, 0.5*out_flows+0.5])

    titles = ['in_frame', 'out_frame', 'in_flow', 'out_flow']
    assert len(titles) == r
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if i < 2:
                axs[i, j].imshow(np.clip(gen_imgs[cnt], 0., 1.))
            else:
                axs[i, j].imshow(scale_range(gen_imgs[cnt]))
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("%s/%d_%d.png" % (saved_sample_path, epoch, batch_i))
    plt.close()


# ¶þÎ¬¾í»ý²ã
def conv2d(x, out_channel, filter_size=3, stride=1, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        in_channel = x.get_shape()[-1]
        w = tf.get_variable('w', [filter_size[0], filter_size[1], in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        result = tf.nn.conv2d(x, w, [1, stride, stride, 1], 'SAME') + b
        if return_filters:
            return result, w, b
        return result


# ¶þÎ¬·´¾í»ý²ã
def conv_transpose(x, output_shape, filter_size=3, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size[0], filter_size[1], output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 2, 2, 1]), b)
        if return_filters:
            return convt, w, b
        return convt


def conv2d_Inception(x, out_channel, max_filter_size=7, scope=None):
    assert max_filter_size % 2 == 1 and max_filter_size < 8
    n_branch = (max_filter_size+1) // 2
    assert out_channel % n_branch == 0
    nf_branch = out_channel // n_branch

    with tf.variable_scope(scope):

        s1_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s1_11')
        if n_branch == 1:
            return s1_11
        # 3x3
        s3_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s3_11')
        s3_1n = conv2d(s3_11, nf_branch, filter_size=(1, 3), scope='s3_1n')
        s3_n1 = conv2d(s3_1n, nf_branch, filter_size=(3, 1), scope='s3_n1')
        if n_branch == 2:
            return tf.concat([s1_11, s3_n1], -1)            # ÒÔÍ¨µÀ¼¶Áª²¢·µ»Ø
        # 5x5
        s5_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s5_11')
        s5_1n = conv2d(s5_11, nf_branch, filter_size=(1, 3), scope='s5_1n_1')
        s5_n1 = conv2d(s5_1n, nf_branch, filter_size=(3, 1), scope='s5_n1_1')
        s5_1n = conv2d(s5_n1, nf_branch, filter_size=(1, 3), scope='s5_1n_2')
        s5_n1 = conv2d(s5_1n, nf_branch, filter_size=(3, 1), scope='s5_n1_2')
        if n_branch == 3:
            return tf.concat([s1_11, s3_n1, s5_n1], -1)
        # 7x7
        s7_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s7_11')
        s7_1n = conv2d(s7_11, nf_branch, filter_size=(1, 3), scope='s7_1n_1')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_1')
        s7_1n = conv2d(s7_n1, nf_branch, filter_size=(1, 3), scope='s7_1n_2')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_2')
        s7_1n = conv2d(s7_n1, nf_branch, filter_size=(1, 3), scope='s7_1n_3')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_3')
        return tf.concat([s1_11, s3_n1, s5_n1, s7_n1], -1)


def G_conv_bn_lrelu(x, out_channel, filter_size, stride=2, training=False, bn=True, scope=None):
    with tf.variable_scope(scope):
        d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
        if bn:
            d = tf.layers.batch_normalization(d, training=training)
        d = tf.nn.leaky_relu(d)
        return d


def G_deconv_bn_dr_relu_concat(layer_input, skip_input, out_shape, filter_size, p_keep_drop, training=False, scope=None):
    with tf.variable_scope(scope):
        """Layers used during upsampling"""
        u = conv_transpose(layer_input, out_shape, filter_size=filter_size, scope='deconv')
        u = tf.layers.batch_normalization(u, training=training)
        u = tf.nn.dropout(u, p_keep_drop)
        u = tf.nn.relu(u)
        if skip_input is not None:
            u = tf.concat([u, skip_input], -1)
        return u


def Generator_Appearance(input_data, is_training, keep_prob, return_layers=False):

    with tf.variable_scope('generator_appearance'):
        b_size = tf.shape(input_data)[0]
        h = tf.shape(input_data)[1]
        w = tf.shape(input_data)[2]

        h0 = input_data
        filters = 64
        filter_size = (4, 4)
        '''COMMON ENCODER'''
        h0 = conv2d_Inception(h0, filters, max_filter_size=7, scope='gen_h0')
        h1 = G_conv_bn_lrelu(h0, filters, filter_size, stride=1, training=is_training, bn=False, scope='gen_h1')
        h2 = G_conv_bn_lrelu(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='gen_h2')
        h3 = G_conv_bn_lrelu(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='gen_h3')
        h4 = G_conv_bn_lrelu(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h4')
        h5 = G_conv_bn_lrelu(h4, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h5')

        '''Unet DECODER for FRAME'''
        h4fr = G_deconv_bn_dr_relu_concat(h5, None, [b_size, h//8, w//8, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h4fr')
        h3fr = G_deconv_bn_dr_relu_concat(h4fr, None, [b_size, h//4, w//4, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h3fr')
        h2fr = G_deconv_bn_dr_relu_concat(h3fr, None, [b_size, h//2, w//2, filters*2], filter_size, keep_prob, training=is_training, scope='gen_h2fr')
        h1fr = G_deconv_bn_dr_relu_concat(h2fr, None, [b_size, h, w, filters], filter_size, keep_prob, training=is_training, scope='gen_h1fr')
        out_frame = conv2d(h1fr, input_data.get_shape()[-1], filter_size=3, stride=1, scope='gen_frame')
        #
        if return_layers:
            # return out_frame, [h0, h1, h2, h3, h4, h5, h4fr, h3fr, h2fr, h1fr]
            return out_frame, h1, h1fr, h5
        return out_frame


def Generator_Flow(input_data, is_training, keep_prob, return_layers=False):

    with tf.variable_scope('generator_flow'):
        b_size = tf.shape(input_data)[0]
        h = tf.shape(input_data)[1]
        w = tf.shape(input_data)[2]

        h0 = input_data
        filters = 64
        filter_size = (4, 4)
        '''COMMON ENCODER'''
        h0 = conv2d_Inception(h0, filters, max_filter_size=7, scope='gen_h0')
        h1 = G_conv_bn_lrelu(h0, filters, filter_size, stride=1, training=is_training, bn=False, scope='gen_h1')
        h2 = G_conv_bn_lrelu(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='gen_h2')
        h3 = G_conv_bn_lrelu(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='gen_h3')
        h4 = G_conv_bn_lrelu(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h4')
        h5 = G_conv_bn_lrelu(h4, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h5')

        '''Unet DECODER for OPTICAL FLOW MAGITUDE'''
        h4fl = G_deconv_bn_dr_relu_concat(h5, SElayer(h4, out_dim=filters*8), [b_size, h//8, w//8, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h4fl')
        h3fl = G_deconv_bn_dr_relu_concat(h4fl, SElayer(h3, out_dim=filters*4), [b_size, h//4, w//4, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h3fl')
        h2fl = G_deconv_bn_dr_relu_concat(h3fl, SElayer(h2, out_dim=filters*2), [b_size, h//2, w//2, filters*2], filter_size, keep_prob, training=is_training, scope='gen_h2fl')
        h1fl = G_deconv_bn_dr_relu_concat(h2fl, SElayer(h1, out_dim=filters), [b_size, h, w, filters], filter_size, keep_prob, training=is_training, scope='gen_h1fl')
        out_flow = conv2d(h1fl, input_data.get_shape()[-1], filter_size=3, stride=1, scope='gen_flow')

        '''Unet DECODER for OPTICAL FLOW ANGLES'''
        h4fla = G_deconv_bn_dr_relu_concat(h5, SElayer(h4, out_dim=filters*8), [b_size, h // 8, w // 8, filters * 4], filter_size, keep_prob,
                                          training=is_training, scope='gen_h4fla')
        h3fla = G_deconv_bn_dr_relu_concat(h4fla, SElayer(h3, out_dim=filters*4), [b_size, h // 4, w // 4, filters * 4], filter_size, keep_prob,
                                          training=is_training, scope='gen_h3fla')
        h2fla = G_deconv_bn_dr_relu_concat(h3fla, SElayer(h2, out_dim=filters*2), [b_size, h // 2, w // 2, filters * 2], filter_size, keep_prob,
                                          training=is_training, scope='gen_h2fla')
        h1fla = G_deconv_bn_dr_relu_concat(h2fla, SElayer(h1, out_dim=filters), [b_size, h, w, filters], filter_size, keep_prob,
                                          training=is_training, scope='gen_h1fla')
        out_flow_ang = conv2d(h1fla, input_data.get_shape()[-1], filter_size=3, stride=1, scope='gen_flow_ang')

        #
        if return_layers:
            return out_flow, out_flow_ang, [h0, h1, h2, h3, h4, h5, h4fl, h3fl, h2fl, h1fl]
        return out_flow, out_flow_ang


def Discriminator_image(image_hat, is_training, reuse=False, return_middle_layers=False):

    def D_conv_bn_active(x, out_channel, filter_size, stride=2, training=False, bn=True, active=tf.nn.leaky_relu, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            if active is not None:
                d = active(d)
            return d

    with tf.variable_scope('discriminator') as var_scope:
        if reuse:
            var_scope.reuse_variables()

        filters = 64
        filter_size = (4, 4)

        # h0 = tf.concat([frame_true, flow_hat], -1)
        h0 = image_hat
        h1 = D_conv_bn_active(h0, filters, filter_size, stride=2, training=is_training, bn=False, scope='dis_h1')
        h2 = D_conv_bn_active(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='dis_h2')
        h3 = D_conv_bn_active(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='dis_h3')
        h4 = D_conv_bn_active(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, active=None, scope='dis_h4')

        if return_middle_layers:
            return tf.nn.sigmoid(h4), h4, [h1, h2, h3]
        return tf.nn.sigmoid(h4), h4


def Discriminator_flow(flow_hat, is_training, reuse=False, return_middle_layers=False):

    def D_conv_bn_active(x, out_channel, filter_size, stride=2, training=False, bn=True, active=tf.nn.leaky_relu, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            if active is not None:
                d = active(d)
            return d

    with tf.variable_scope('discriminator') as var_scope:
        if reuse:
            var_scope.reuse_variables()

        filters = 64
        filter_size = (4, 4)

        # h0 = tf.concat([frame_true, flow_hat], -1)
        h0 = flow_hat
        h1 = D_conv_bn_active(h0, filters, filter_size, stride=2, training=is_training, bn=False, scope='dis_h1')
        h2 = D_conv_bn_active(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='dis_h2')
        h3 = D_conv_bn_active(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='dis_h3')
        h4 = D_conv_bn_active(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, active=None, scope='dis_h4')

        if return_middle_layers:
            return tf.nn.sigmoid(h4), h4, [h1, h2, h3]
        return tf.nn.sigmoid(h4), h4


def train_model(training_images, training_flows, training_flows_ang, max_epoch, dataset_name='', start_model_idx=0, batch_size=16, channel=1):
    print('no. of images = %s' % len(training_images))

    assert len(training_images) == len(training_flows)
    h, w = training_images.shape[1:3]
    assert h < w

    training_images /= 0.5
    training_images -= 1.

    training_flows /= 0.5
    training_flows -= 1.

    training_flows_ang /= 0.5
    training_flows_ang -= 1.

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_flow_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_flow_ang_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_appe, h1_f_appe, h1_b_appe, encoder_vec = Generator_Appearance(plh_frame_true, plh_is_training, plh_dropout_prob, True)
    output_opt, output_opt_ang = Generator_Flow(output_appe, plh_is_training, plh_dropout_prob)

    # discriminator for true flow and fake flow
    D_real_flow, D_real_logits_flow = Discriminator_flow(plh_flow_true, plh_is_training, reuse=False)
    D_fake_flow, D_fake_logits_flow = Discriminator_flow(output_opt, plh_is_training, reuse=True)

    # appearance loss
    dy1, dx1 = tf.image.image_gradients(output_appe)
    dy0, dx0 = tf.image.image_gradients(plh_frame_true)
    loss_inten = tf.reduce_mean((output_appe - plh_frame_true)**2)
    loss_gradi = tf.reduce_mean(tf.abs(tf.abs(dy1)-tf.abs(dy0)) + tf.abs(tf.abs(dx1)-tf.abs(dx0)))
    loss_appe = loss_inten + loss_gradi
    # feature map loss
    loss_perceptual = tf.reduce_mean((h1_f_appe - h1_b_appe)**2)

    # optical loss
    loss_opt_mag = tf.reduce_mean(tf.abs(output_opt - plh_flow_true))
    loss_opt_ang = tf.reduce_mean(tf.abs(output_opt_ang - plh_flow_ang_true))
    loss_opt = loss_opt_mag + loss_opt_ang


    # GAN loss
    D_loss = 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits_flow, labels=tf.ones_like(D_real_flow))) + \
             0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_flow, labels=tf.zeros_like(D_fake_flow)))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits_flow, labels=tf.ones_like(D_fake_flow)))
    # G_loss = tf.reduce_mean((D_fake_logits_flow - D_real_logits_flow) ** 2)
    G_loss_total = 0.25 * G_loss + loss_appe + 2 * loss_opt + 0.2 * loss_perceptual


    # optimizers
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'gen_' in var.name]
    d_vars = [var for var in t_vars if 'dis_' in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        D_optimizer = tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.9, beta2=0.999, name='AdamD').minimize(D_loss, var_list=d_vars)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.9, beta2=0.999, name='AdamG').minimize(G_loss_total, var_list=g_vars)

    init_op = tf.global_variables_initializer()

    # tensorboard
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('appe_loss', loss_appe)
    tf.summary.scalar('opt_loss', loss_opt)
    merge = tf.summary.merge_all()

    #
    saver = tf.train.Saver(max_to_keep=50)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # maximun alloc gpu 90% of MEM
    config.gpu_options.allow_growth = True  # allocate dynamically
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        loss_anomaly = []
        loss_normal = []
        loss_recoder = []
        if start_model_idx > 0:
            saver.restore(sess, './training_saver/%s/model_ckpt_%d.ckpt' % (dataset_name, start_model_idx))
            loss_anomaly = scio.loadmat('./training_saver/%s/loss_anomaly_%d.mat' % (dataset_name, start_model_idx))['loss'][0]
            loss_normal = scio.loadmat('./training_saver/%s/loss_normal_%d.mat' % (dataset_name, start_model_idx))['loss'][0]
            loss_recoder = scio.loadmat('./training_saver/%s/loss_recoder_%d.mat' % (dataset_name, start_model_idx))['loss'][0]

        # define log path for tensorboard
        tensorboard_path = './training_saver/%s/logs/2/train' % (dataset_name)
        if not os.path.exists(tensorboard_path):
            pathlib.Path(tensorboard_path).mkdir(parents=True, exist_ok=True)

        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        print('Run: tensorboard --logdir logs/2')
        # executive training stage
        is_mask =True
        for i in range(start_model_idx, max_epoch):
            if is_mask:
                mask = generate_mask(h, w)

            tf.set_random_seed(i)
            np.random.seed(i)

            batch_idx = np.array_split(np.random.permutation(len(training_images)), np.ceil(len(training_images)/batch_size))
            for j in range(len(batch_idx)):
                # discriminator
                _, curr_D_loss, summary = sess.run([D_optimizer, D_loss, merge],
                                                   feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                                              plh_flow_true: training_flows[batch_idx[j]],
                                                              plh_flow_ang_true: training_flows_ang[batch_idx[j]],
                                                              plh_is_training: True})

                # genarator
                if j % len(batch_idx) == 0:
                    _, curr_G_loss, curr_loss_appe, curr_loss_opt, curr_loss_opt_ang, curr_gen_frames, curr_gen_flows, summary = \
                        sess.run([G_optimizer, G_loss, loss_appe, loss_opt_mag, loss_opt_ang, output_appe[:4],
                                  output_opt[:4], merge],
                                 feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                            plh_flow_true: training_flows[batch_idx[j]],
                                            plh_flow_ang_true: training_flows_ang[batch_idx[j]],
                                            plh_dropout_prob: p_keep,
                                            plh_is_training: True,
                                            })
                    sample_images(dataset_name, training_flows[batch_idx[j][:4]], training_images[batch_idx[j][:4]],
                                  curr_gen_flows, curr_gen_frames, i, j, train=True)

                else:
                    _, curr_G_loss, curr_loss_appe, curr_loss_opt, curr_loss_opt_ang, summary = \
                        sess.run([G_optimizer, G_loss, loss_appe, loss_opt_mag, loss_opt_ang, merge],
                                 feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                            plh_flow_true: training_flows[batch_idx[j]],
                                            plh_flow_ang_true: training_flows_ang[batch_idx[j]],
                                            plh_dropout_prob: p_keep,
                                            plh_is_training: True,
                                            })

                    # normal
                    curr_loss_appe_n = sess.run(loss_appe, feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                                                      plh_dropout_prob: 1.0,
                                                                      plh_is_training: False})

                    if is_mask:
                        training_images_a = training_images[batch_idx[j]].transpose(0, 3, 1, 2) * mask
                        training_images_a = training_images_a.transpose(0, 2, 3, 1)

                    # pseudo-anomaly
                    curr_loss_appe_a, re_pseudo = sess.run([loss_appe, output_appe],
                                                           feed_dict={plh_frame_true: training_images_a,
                                                                      plh_dropout_prob: 1.0,
                                                                      plh_is_training: False})

                    loss_anomaly = np.append(loss_anomaly, curr_loss_appe_a)  # 伪异常的外观重构损失
                    loss_normal = np.append(loss_normal, curr_loss_appe_n)  # 正常的外观重构损失
                    loss_recoder = np.append(loss_recoder, curr_loss_appe)  # 训练重构损失

                    if j % 50 == 0:
                        print(
                            'Epoch: %d/%d, Batch: %3d/%d,  D = %.4f, G = %.4f, appe = %.4f, '
                            'flow = %.4f, ang = %.4f, anomaly = %.4f, normal = %.4f'
                            % (i + 1, max_epoch, j + 1, len(batch_idx), curr_D_loss, curr_G_loss, curr_loss_appe,
                               curr_loss_opt, curr_loss_opt_ang, curr_loss_appe_a, curr_loss_appe_n))

                    if np.isnan(curr_D_loss) or np.isnan(curr_G_loss) or np.isnan(curr_loss_appe) or np.isnan(
                            curr_loss_opt) \
                            or np.isnan(curr_loss_opt_ang):
                        return

                train_writer.add_summary(summary, i)
                train_writer.flush()  # 刷新缓冲区，立即写入文件

                # save the model in chkt form
                if (i + 1) % 10 == 0:
                    saver.save(sess, './training_saver/%s/model_ckpt_%d.ckpt' % (dataset_name, i + 1))
                    scio.savemat('./training_saver/%s/loss_anomaly_%d.mat' % (dataset_name, i + 1),
                                 {'loss': loss_anomaly})
                    scio.savemat('./training_saver/%s/loss_normal_%d.mat' % (dataset_name, i + 1),
                                 {'loss': loss_normal})
                    scio.savemat('./training_saver/%s/loss_recoder_%d.mat' % (dataset_name, i + 1),
                                 {'loss': loss_recoder})


def test_model(h, w, dataset, sequence_n_frame, batch_size=32, model_idx=20, using_test_data=True, channel=1):

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_flow_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_flow_ang_true = tf.placeholder(tf.float32, shape=[None, h, w, channel])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_appe = Generator_Appearance(plh_frame_true, plh_is_training, plh_dropout_prob)
    output_opt, output_opt_ang = Generator_Flow(output_appe, plh_is_training, plh_dropout_prob)

    # appearance loss
    dy1, dx1 = tf.image.image_gradients(output_appe)
    dy0, dx0 = tf.image.image_gradients(plh_frame_true)
    loss_inten = tf.reduce_mean((output_appe - plh_frame_true) ** 2)
    loss_gradi = tf.reduce_mean(tf.abs(tf.abs(dy1) - tf.abs(dy0)) + tf.abs(tf.abs(dx1) - tf.abs(dx0)))
    loss_appe = loss_inten + loss_gradi

    # optical loss
    loss_opt_mag = tf.reduce_mean(tf.abs(output_opt - plh_flow_true))
    loss_opt_ang = tf.reduce_mean(tf.abs(output_opt_ang - plh_flow_ang_true))
    loss_opt = loss_opt_mag + loss_opt_ang

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #
        saved_model_file = './training_saver/%s/model_ckpt_%d.ckpt' % (dataset['name'], model_idx)
        saver.restore(sess, saved_model_file)
        #
        saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'test' if using_test_data else 'train', model_idx)
        if not os.path.exists(saved_data_path):
            pathlib.Path(saved_data_path).mkdir(parents=True, exist_ok=True)
        for clip_idx in range(dataset['n_clip_train'] if not using_test_data else dataset['n_clip_test']):
            # clip_idx = 4
            saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
            if os.path.isfile(saved_data_file):
                print('File existed! Return!')
                continue

            #
            test_images, test_flows, test_flows_ang = load_images_and_flow_1clip(dataset, clip_idx, train=not using_test_data)
            print(test_images.shape, test_flows.shape, np.sum(sequence_n_frame))
            assert len(test_images) == len(test_flows)
            assert len(test_images) == sequence_n_frame[clip_idx]
            #
            saved_out_appes = np.zeros(test_images.shape)
            saved_out_flows = np.zeros(test_flows.shape)
            saved_out_flows_ang = np.zeros(test_flows_ang.shape)

            #
            test_images /= 0.5
            test_images -= 1.

            test_flows /= 0.5
            test_flows -= 1.

            test_flows_ang /= 0.5
            test_flows_ang -= 1.

            batch_idx = np.array_split(np.arange(len(test_images)), np.ceil(len(test_images) / batch_size))
            for j in range(len(batch_idx)):
                saved_out_appes[batch_idx[j]], saved_out_flows[batch_idx[j]], saved_out_flows_ang[batch_idx[j]], curr_loss_appe, curr_loss_opt = \
                    sess.run([output_appe, output_opt, output_opt_ang, loss_appe, loss_opt_mag],
                             feed_dict={plh_frame_true: test_images[batch_idx[j]],
                                        plh_flow_true: test_flows[batch_idx[j]],
                                        plh_flow_ang_true: test_flows_ang[batch_idx[j]],
                                        plh_is_training: False,
                                        plh_dropout_prob: 1.0})
                saved_out_appes[batch_idx[j]] = 0.5 * (saved_out_appes[batch_idx[j]] + 1)
                saved_out_flows[batch_idx[j]] = 0.5 * (saved_out_flows[batch_idx[j]] + 1)
                saved_out_flows_ang[batch_idx[j]] = 0.5 * (saved_out_flows_ang[batch_idx[j]] + 1)

            np.savez_compressed(saved_data_file, image=saved_out_appes, flow=saved_out_flows, flow_ang=saved_out_flows_ang)




#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Network structure design for SketchCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from SketchCNN.utils.util_func import cropconcat_layer
import tensorflow.contrib.slim as slim
import logging

# network logger initialization
net_logger = logging.getLogger('main.network')


class SKETCHNET(object):
    """CNN networks for normal, depth, confidence and direction field regression.
    """

    # Cook raw input into different tensors.
    @staticmethod
    def cook_raw_inputs(raw_input):
        """
        Args:
            :param raw_input: raw input after custom decoder.

        Returns:
            :return: divided tensors.
        """
        with tf.name_scope("cook_raw_input") as _:
            input_data, label_data = raw_input

            npr_line = tf.slice(input_data, [0, 0, 0, 0], [-1, -1, -1, 1])
            depth_sample = tf.slice(input_data, [0, 0, 0, 3], [-1, -1, -1, 1])
            distance_field = tf.slice(input_data, [0, 0, 0, 1], [-1, -1, -1, 2])
            feature_mask = tf.slice(input_data, [0, 0, 0, 4], [-1, -1, -1, 1])
            feature_mask_inv = tf.slice(input_data, [0, 0, 0, 5], [-1, -1, -1, 1])
            selLine_mask = tf.slice(label_data, [0, 0, 0, 15], [-1, -1, -1, 1])
            vdotn_scalar = tf.slice(label_data, [0, 0, 0, 16], [-1, -1, -1, 1])
            label_normal = tf.slice(label_data, [0, 0, 0, 0], [-1, -1, -1, 3])
            label_depth = tf.slice(label_data, [0, 0, 0, 3], [-1, -1, -1, 1])
            mask_shape = tf.slice(label_data, [0, 0, 0, 8], [-1, -1, -1, 1])
            mask_depth_sample = tf.slice(label_data, [0, 0, 0, 9], [-1, -1, -1, 1])
            label_field = tf.slice(label_data, [0, 0, 0, 4], [-1, -1, -1, 4])
            cmask_line = tf.slice(label_data, [0, 0, 0, 10], [-1, -1, -1, 1])
            mask_line_inv = tf.slice(label_data, [0, 0, 0, 12], [-1, -1, -1, 1])
            mask_line = tf.slice(label_data, [0, 0, 0, 11], [-1, -1, -1, 1])
            mask2d = tf.slice(label_data, [0, 0, 0, 13], [-1, -1, -1, 1])

            return npr_line, depth_sample, distance_field, feature_mask, feature_mask_inv, label_normal, label_depth, \
                   label_field, cmask_line, mask_shape, mask_depth_sample, mask_line, mask_line_inv, mask2d, \
                   selLine_mask, vdotn_scalar

    # Direction field regression network.
    @staticmethod
    def load_field_net(lines, mask2d, ds, fm, selm, vdotn, root_feature=32, is_training=True, padding='SAME',
                       reuse=None, d_rate=1,
                       l2_reg=0.0005):
        """Direction field regression network. U shape network.

        Args:
            :param lines: npr lines.
            :param mask2d: 2d mask, consistent with interface.
            :param ds: depth samples.
            :param fm: feature mask.
            :param selm: selected line mask.
            :param vdotn: normal change scalar.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: regressed field(4), field regression variable list.
        """
        with tf.variable_scope('SASFieldNet', reuse=reuse) as f_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                reg_f_input = tf.concat([lines, mask2d, ds, fm, selm, vdotn], axis=3, name='concat_field_input')
                f_input = tf.identity(reg_f_input, name='reg_f_input')
                conv1 = slim.conv2d(f_input, root_feature, scope='f_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='f_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='f_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='f_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='f_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='f_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='f_conv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='f_conv6')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    f_deconv3 = slim.conv2d_transpose(conv6, root_feature * 2, scope='f_deconv3_1')
                    f_concat3 = cropconcat_layer(conv4, f_deconv3, 3, name='f_concat3')
                    f_deconv3_2 = slim.conv2d(f_concat3, root_feature * 2, scope='f_deconv3_2')
                    f_deconv3_3 = slim.conv2d(f_deconv3_2, root_feature * 2, scope='f_deconv3_3')
                    f_deconv4 = slim.conv2d_transpose(f_deconv3_3, root_feature, scope='f_deconv4_1')
                    f_concat4 = cropconcat_layer(conv2, f_deconv4, 3, name='f_concat4')
                    f_deconv4_2 = slim.conv2d(f_concat4, root_feature, scope='f_deconv4_2')
                    f_deconv4_3 = slim.conv2d(f_deconv4_2, root_feature, scope='f_deconv4_3')

                    f_res = slim.conv2d(f_deconv4_3, 4, kernel_size=[1, 1], activation_fn=None,
                                        scope='f_output')  # direction field
                    logit_f = tf.identity(f_res, name='output_f')

        f_net_variables = tf.contrib.framework.get_variables(f_vs)

        return logit_f, f_net_variables

    # Geometry regression network.
    @staticmethod
    def load_GeomNet(lines, ds, mask2d, fm, selm, vdotn, field, cl_mask, root_feature=32,
                     is_training=True, padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """Geometry regression network.
            U shape network with shared encoder and 3 individual decoders.

        Args:
            :param lines: npr lines.
            :param ds: depth samples.
            :param mask2d: 2d mask, consistent with interface.
            :param fm: feature mask.
            :param selm: seleted line mask.
            :param vdotn: normal change scalar.
            :param field: direction field.
            :param cl_mask: contour line mask.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: depth(1), normal(3), confidence map(1), GeomNet variable list.
        """
        with tf.variable_scope('SASMFGeoNet', reuse=reuse) as g_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                full_mask = tf.tile(mask2d * cl_mask, [1, 1, 1, 4])
                field = field * full_mask
                reg_geo_input = tf.concat([lines, ds, mask2d, fm, selm, vdotn, field], axis=3, name='concat_geo_input')
                conv1 = slim.conv2d(reg_geo_input, root_feature, scope='geo_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='geo_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='geo_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='geo_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='geo_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='geo_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='geo_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='geo_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='geo_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='geo_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='geo_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='geo_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='geo_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='geo_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    d_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='d_geo_deconv1_1')
                    n_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='n_geo_deconv1_1')
                    c_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='c_geo_deconv1_1')

                    d_concat1 = cropconcat_layer(conv8, d_deconv1, 3, name='d_geo_concat1')
                    n_concat1 = cropconcat_layer(conv8, n_deconv1, 3, name='n_geo_concat1')
                    c_concat1 = cropconcat_layer(conv8, c_deconv1, 3, name='c_geo_concat1')

                    d_deconv1_2 = slim.conv2d(d_concat1, root_feature * 8, scope='d_geo_deconv1_2')
                    n_deconv1_2 = slim.conv2d(n_concat1, root_feature * 8, scope='n_geo_deconv1_2')
                    c_deconv1_2 = slim.conv2d(c_concat1, root_feature * 8, scope='c_geo_deconv1_2')

                    d_deconv1_3 = slim.conv2d(d_deconv1_2, root_feature * 8, scope='d_geo_deconv1_3')
                    n_deconv1_3 = slim.conv2d(n_deconv1_2, root_feature * 8, scope='n_geo_deconv1_3')
                    c_deconv1_3 = slim.conv2d(c_deconv1_2, root_feature * 8, scope='c_geo_deconv1_3')

                    d_deconv2 = slim.conv2d_transpose(d_deconv1_3, root_feature * 4, scope='d_geo_deconv2_1')
                    n_deconv2 = slim.conv2d_transpose(n_deconv1_3, root_feature * 4, scope='n_geo_deconv2_1')
                    c_deconv2 = slim.conv2d_transpose(c_deconv1_3, root_feature * 4, scope='c_geo_deconv2_1')

                    d_concat2 = cropconcat_layer(conv6, d_deconv2, 3, name='d_geo_concat2')
                    n_concat2 = cropconcat_layer(conv6, n_deconv2, 3, name='n_geo_concat2')
                    c_concat2 = cropconcat_layer(conv6, c_deconv2, 3, name='c_geo_concat2')

                    d_deconv2_2 = slim.conv2d(d_concat2, root_feature * 4, scope='d_geo_deonv2_2')
                    n_deconv2_2 = slim.conv2d(n_concat2, root_feature * 4, scope='n_geo_deonv2_2')
                    c_deconv2_2 = slim.conv2d(c_concat2, root_feature * 4, scope='c_geo_deonv2_2')

                    d_deconv2_3 = slim.conv2d(d_deconv2_2, root_feature * 4, scope='d_geo_deconv2_3')
                    n_deconv2_3 = slim.conv2d(n_deconv2_2, root_feature * 4, scope='n_geo_deconv2_3')
                    c_deconv2_3 = slim.conv2d(c_deconv2_2, root_feature * 4, scope='c_geo_deconv2_3')

                    d_deconv3 = slim.conv2d_transpose(d_deconv2_3, root_feature * 2, scope='d_geo_deconv3_1')
                    n_deconv3 = slim.conv2d_transpose(n_deconv2_3, root_feature * 2, scope='n_geo_deconv3_1')
                    c_deconv3 = slim.conv2d_transpose(c_deconv2_3, root_feature * 2, scope='c_geo_deconv3_1')

                    d_concat3 = cropconcat_layer(conv4, d_deconv3, 3, name='d_geo_concat3')
                    n_concat3 = cropconcat_layer(conv4, n_deconv3, 3, name='n_geo_concat3')
                    c_concat3 = cropconcat_layer(conv4, c_deconv3, 3, name='c_geo_concat3')

                    d_deconv3_2 = slim.conv2d(d_concat3, root_feature * 2, scope='d_geo_deconv3_2')
                    n_deconv3_2 = slim.conv2d(n_concat3, root_feature * 2, scope='n_geo_deconv3_2')
                    c_deconv3_2 = slim.conv2d(c_concat3, root_feature * 2, scope='c_geo_deconv3_2')

                    d_deconv3_3 = slim.conv2d(d_deconv3_2, root_feature * 2, scope='d_geo_deconv3_3')
                    n_deconv3_3 = slim.conv2d(n_deconv3_2, root_feature * 2, scope='n_geo_deconv3_3')
                    c_deconv3_3 = slim.conv2d(c_deconv3_2, root_feature * 2, scope='c_geo_deconv3_3')

                    d_deconv4 = slim.conv2d_transpose(d_deconv3_3, root_feature, scope='d_geo_deconv4_1')
                    n_deconv4 = slim.conv2d_transpose(n_deconv3_3, root_feature, scope='n_geo_deconv4_1')
                    c_deconv4 = slim.conv2d_transpose(c_deconv3_3, root_feature, scope='c_geo_deconv4_1')

                    d_concat4 = cropconcat_layer(conv2, d_deconv4, 3, name='d_geo_concat4')
                    n_concat4 = cropconcat_layer(conv2, n_deconv4, 3, name='n_geo_concat4')
                    c_concat4 = cropconcat_layer(conv2, c_deconv4, 3, name='c_geo_concat4')

                    d_deconv4_2 = slim.conv2d(d_concat4, root_feature, scope='d_geo_deconv4_2')
                    n_deconv4_2 = slim.conv2d(n_concat4, root_feature, scope='n_geo_deconv4_2')
                    c_deconv4_2 = slim.conv2d(c_concat4, root_feature, scope='c_geo_deconv4_2')

                    d_deconv4_3 = slim.conv2d(d_deconv4_2, root_feature, scope='d_geo_deconv4_3')
                    n_deconv4_3 = slim.conv2d(n_deconv4_2, root_feature, scope='n_geo_deconv4_3')
                    c_deconv4_3 = slim.conv2d(c_deconv4_2, root_feature, scope='c_geo_deconv4_3')

                    res_d = slim.conv2d(d_deconv4_3, 1, kernel_size=[1, 1], scope='d_geo_output')  # depth
                    res_n = slim.conv2d(n_deconv4_3, 3, kernel_size=[1, 1], scope='n_geo_output')  # normal
                    res_c = slim.conv2d(c_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                        scope='c_geo_output')  # confidence map

                    logit_d = tf.identity(res_d, name='output_d')
                    logit_n = tf.identity(res_n, name='output_n')
                    logit_c = tf.identity(res_c, name='output_c')

        geom_net_variables = tf.contrib.framework.get_variables(g_vs)

        return logit_d, logit_n, logit_c, geom_net_variables

    # Baseline network.
    @staticmethod
    def load_baseline_net(lines, ds, mask2d, fm, selm, vdotn, root_feature=32, is_training=True,
                          padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """Baseline network without direction field input.
            U shape network with shared encoder and 3 individual decoders.

        Args:
            :param lines: npr lines.
            :param ds: depth samples.
            :param mask2d: 2d mask consistent with interface.
            :param fm: feature mask.
            :param selm: selected line mask.
            :param vdotn: vdotn.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: depth(1), normal(3), confidence map(1), baseline network variable list.
        """
        with tf.variable_scope('SASBLNet', reuse=reuse) as g_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                reg_geo_input = tf.concat([lines, ds, mask2d, fm, selm, vdotn], axis=3, name='concat_geo_input')
                geo_input = tf.identity(reg_geo_input, name='reg_geo_input')
                conv1 = slim.conv2d(geo_input, root_feature, scope='geo_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='geo_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='geo_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='geo_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='geo_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='geo_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='geo_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='geo_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='geo_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='geo_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='geo_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='geo_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='geo_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='geo_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    d_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='d_geo_deconv1_1')
                    n_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='n_geo_deconv1_1')
                    c_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='c_geo_deconv1_1')

                    d_concat1 = cropconcat_layer(conv8, d_deconv1, 3, name='d_geo_concat1')
                    n_concat1 = cropconcat_layer(conv8, n_deconv1, 3, name='n_geo_concat1')
                    c_concat1 = cropconcat_layer(conv8, c_deconv1, 3, name='c_geo_concat1')

                    d_deconv1_2 = slim.conv2d(d_concat1, root_feature * 8, scope='d_geo_deconv1_2')
                    n_deconv1_2 = slim.conv2d(n_concat1, root_feature * 8, scope='n_geo_deconv1_2')
                    c_deconv1_2 = slim.conv2d(c_concat1, root_feature * 8, scope='c_geo_deconv1_2')

                    d_deconv1_3 = slim.conv2d(d_deconv1_2, root_feature * 8, scope='d_geo_deconv1_3')
                    n_deconv1_3 = slim.conv2d(n_deconv1_2, root_feature * 8, scope='n_geo_deconv1_3')
                    c_deconv1_3 = slim.conv2d(c_deconv1_2, root_feature * 8, scope='c_geo_deconv1_3')

                    d_deconv2 = slim.conv2d_transpose(d_deconv1_3, root_feature * 4, scope='d_geo_deconv2_1')
                    n_deconv2 = slim.conv2d_transpose(n_deconv1_3, root_feature * 4, scope='n_geo_deconv2_1')
                    c_deconv2 = slim.conv2d_transpose(c_deconv1_3, root_feature * 4, scope='c_geo_deconv2_1')

                    d_concat2 = cropconcat_layer(conv6, d_deconv2, 3, name='d_geo_concat2')
                    n_concat2 = cropconcat_layer(conv6, n_deconv2, 3, name='n_geo_concat2')
                    c_concat2 = cropconcat_layer(conv6, c_deconv2, 3, name='c_geo_concat2')

                    d_deconv2_2 = slim.conv2d(d_concat2, root_feature * 4, scope='d_geo_deonv2_2')
                    n_deconv2_2 = slim.conv2d(n_concat2, root_feature * 4, scope='n_geo_deonv2_2')
                    c_deconv2_2 = slim.conv2d(c_concat2, root_feature * 4, scope='c_geo_deonv2_2')

                    d_deconv2_3 = slim.conv2d(d_deconv2_2, root_feature * 4, scope='d_geo_deconv2_3')
                    n_deconv2_3 = slim.conv2d(n_deconv2_2, root_feature * 4, scope='n_geo_deconv2_3')
                    c_deconv2_3 = slim.conv2d(c_deconv2_2, root_feature * 4, scope='c_geo_deconv2_3')

                    d_deconv3 = slim.conv2d_transpose(d_deconv2_3, root_feature * 2, scope='d_geo_deconv3_1')
                    n_deconv3 = slim.conv2d_transpose(n_deconv2_3, root_feature * 2, scope='n_geo_deconv3_1')
                    c_deconv3 = slim.conv2d_transpose(c_deconv2_3, root_feature * 2, scope='c_geo_deconv3_1')

                    d_concat3 = cropconcat_layer(conv4, d_deconv3, 3, name='d_geo_concat3')
                    n_concat3 = cropconcat_layer(conv4, n_deconv3, 3, name='n_geo_concat3')
                    c_concat3 = cropconcat_layer(conv4, c_deconv3, 3, name='c_geo_concat3')

                    d_deconv3_2 = slim.conv2d(d_concat3, root_feature * 2, scope='d_geo_deconv3_2')
                    n_deconv3_2 = slim.conv2d(n_concat3, root_feature * 2, scope='n_geo_deconv3_2')
                    c_deconv3_2 = slim.conv2d(c_concat3, root_feature * 2, scope='c_geo_deconv3_2')

                    d_deconv3_3 = slim.conv2d(d_deconv3_2, root_feature * 2, scope='d_geo_deconv3_3')
                    n_deconv3_3 = slim.conv2d(n_deconv3_2, root_feature * 2, scope='n_geo_deconv3_3')
                    c_deconv3_3 = slim.conv2d(c_deconv3_2, root_feature * 2, scope='c_geo_deconv3_3')

                    d_deconv4 = slim.conv2d_transpose(d_deconv3_3, root_feature, scope='d_geo_deconv4_1')
                    n_deconv4 = slim.conv2d_transpose(n_deconv3_3, root_feature, scope='n_geo_deconv4_1')
                    c_deconv4 = slim.conv2d_transpose(c_deconv3_3, root_feature, scope='c_geo_deconv4_1')

                    d_concat4 = cropconcat_layer(conv2, d_deconv4, 3, name='d_geo_concat4')
                    n_concat4 = cropconcat_layer(conv2, n_deconv4, 3, name='n_geo_concat4')
                    c_concat4 = cropconcat_layer(conv2, c_deconv4, 3, name='c_geo_concat4')

                    d_deconv4_2 = slim.conv2d(d_concat4, root_feature, scope='d_geo_deconv4_2')
                    n_deconv4_2 = slim.conv2d(n_concat4, root_feature, scope='n_geo_deconv4_2')
                    c_deconv4_2 = slim.conv2d(c_concat4, root_feature, scope='c_geo_deconv4_2')

                    d_deconv4_3 = slim.conv2d(d_deconv4_2, root_feature, scope='d_geo_deconv4_3')
                    n_deconv4_3 = slim.conv2d(n_deconv4_2, root_feature, scope='n_geo_deconv4_3')
                    c_deconv4_3 = slim.conv2d(c_deconv4_2, root_feature, scope='c_geo_deconv4_3')

                    res_d = slim.conv2d(d_deconv4_3, 1, kernel_size=[1, 1], scope='d_geo_output')  # depth
                    res_n = slim.conv2d(n_deconv4_3, 3, kernel_size=[1, 1], scope='n_geo_output')  # normal
                    res_c = slim.conv2d(c_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                        scope='c_geo_output')  # confidence map

                    logit_d = tf.identity(res_d, name='output_d')
                    logit_n = tf.identity(res_n, name='output_n')
                    logit_c = tf.identity(res_c, name='output_c')

        baseline_net_variables = tf.contrib.framework.get_variables(g_vs)

        return logit_d, logit_n, logit_c, baseline_net_variables

    # Naive network.
    @staticmethod
    def load_dn_naive_net(lines, ds, mask2d, fm, selm, vdotn, root_feature=32, is_training=True,
                          padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """Naive dn regression network.
            U shape network with shared encoder and 2 individual decoders.

        Args:
            :param lines: npr lines.
            :param ds: depth samples.
            :param mask2d: mask along 2D lines, consistent with interface.
            :param fm: feature mask.
            :param selm: selected line mask.
            :param vdotn: vdotn scalar.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: depth(1), normal(3), naiveNet variable list.
        """
        with tf.variable_scope('SASDNNet', reuse=reuse) as g_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                reg_geo_input = tf.concat([lines, ds, mask2d, fm, selm, vdotn], axis=3, name='concat_geo_input')
                geo_input = tf.identity(reg_geo_input, name='reg_geo_input')
                conv1 = slim.conv2d(geo_input, root_feature, scope='geo_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='geo_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='geo_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='geo_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='geo_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='geo_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='geo_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='geo_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='geo_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='geo_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='geo_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='geo_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='geo_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='geo_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    d_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='d_geo_deconv1_1')
                    n_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='n_geo_deconv1_1')

                    d_concat1 = cropconcat_layer(conv8, d_deconv1, 3, name='d_geo_concat1')
                    n_concat1 = cropconcat_layer(conv8, n_deconv1, 3, name='n_geo_concat1')

                    d_deconv1_2 = slim.conv2d(d_concat1, root_feature * 8, scope='d_geo_deconv1_2')
                    n_deconv1_2 = slim.conv2d(n_concat1, root_feature * 8, scope='n_geo_deconv1_2')

                    d_deconv1_3 = slim.conv2d(d_deconv1_2, root_feature * 8, scope='d_geo_deconv1_3')
                    n_deconv1_3 = slim.conv2d(n_deconv1_2, root_feature * 8, scope='n_geo_deconv1_3')

                    d_deconv2 = slim.conv2d_transpose(d_deconv1_3, root_feature * 4, scope='d_geo_deconv2_1')
                    n_deconv2 = slim.conv2d_transpose(n_deconv1_3, root_feature * 4, scope='n_geo_deconv2_1')

                    d_concat2 = cropconcat_layer(conv6, d_deconv2, 3, name='d_geo_concat2')
                    n_concat2 = cropconcat_layer(conv6, n_deconv2, 3, name='n_geo_concat2')

                    d_deconv2_2 = slim.conv2d(d_concat2, root_feature * 4, scope='d_geo_deonv2_2')
                    n_deconv2_2 = slim.conv2d(n_concat2, root_feature * 4, scope='n_geo_deonv2_2')

                    d_deconv2_3 = slim.conv2d(d_deconv2_2, root_feature * 4, scope='d_geo_deconv2_3')
                    n_deconv2_3 = slim.conv2d(n_deconv2_2, root_feature * 4, scope='n_geo_deconv2_3')

                    d_deconv3 = slim.conv2d_transpose(d_deconv2_3, root_feature * 2, scope='d_geo_deconv3_1')
                    n_deconv3 = slim.conv2d_transpose(n_deconv2_3, root_feature * 2, scope='n_geo_deconv3_1')

                    d_concat3 = cropconcat_layer(conv4, d_deconv3, 3, name='d_geo_concat3')
                    n_concat3 = cropconcat_layer(conv4, n_deconv3, 3, name='n_geo_concat3')

                    d_deconv3_2 = slim.conv2d(d_concat3, root_feature * 2, scope='d_geo_deconv3_2')
                    n_deconv3_2 = slim.conv2d(n_concat3, root_feature * 2, scope='n_geo_deconv3_2')

                    d_deconv3_3 = slim.conv2d(d_deconv3_2, root_feature * 2, scope='d_geo_deconv3_3')
                    n_deconv3_3 = slim.conv2d(n_deconv3_2, root_feature * 2, scope='n_geo_deconv3_3')

                    d_deconv4 = slim.conv2d_transpose(d_deconv3_3, root_feature, scope='d_geo_deconv4_1')
                    n_deconv4 = slim.conv2d_transpose(n_deconv3_3, root_feature, scope='n_geo_deconv4_1')

                    d_concat4 = cropconcat_layer(conv2, d_deconv4, 3, name='d_geo_concat4')
                    n_concat4 = cropconcat_layer(conv2, n_deconv4, 3, name='n_geo_concat4')

                    d_deconv4_2 = slim.conv2d(d_concat4, root_feature, scope='d_geo_deconv4_2')
                    n_deconv4_2 = slim.conv2d(n_concat4, root_feature, scope='n_geo_deconv4_2')

                    d_deconv4_3 = slim.conv2d(d_deconv4_2, root_feature, scope='d_geo_deconv4_3')
                    n_deconv4_3 = slim.conv2d(n_deconv4_2, root_feature, scope='n_geo_deconv4_3')

                    res_d = slim.conv2d(d_deconv4_3, 1, kernel_size=[1, 1], scope='d_geo_output')  # depth
                    res_n = slim.conv2d(n_deconv4_3, 3, kernel_size=[1, 1], scope='n_geo_output')  # normal

                    logit_d = tf.identity(res_d, name='output_d')
                    logit_n = tf.identity(res_n, name='output_n')

        naive_net_variables = tf.contrib.framework.get_variables(g_vs)

        return logit_d, logit_n, naive_net_variables

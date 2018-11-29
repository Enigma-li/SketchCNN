#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Naive network and ablation networks training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from random import randint
import argparse

import tensorflow as tf
from SketchCNN.script.loader import SketchReader
from SketchCNN.script.network import SKETCHNET
from SketchCNN.utils.util_func import slice_tensor, make_dir, dump_params

# Hyper Parameters
hyper_params = {
    'maxIter': 50000000,
    'batchSize': 16,
    'dbTrain': '',
    'dbEval': '',
    'outDir': '',
    'device': '0',
    'nb_gpus': 1,
    'rootFt': 32,
    'dispLossStep': 200,
    'exeValStep': 2000,
    'saveModelStep': 2000,
    'nbDispImg': 4,
    'nbThreads': 64,
    'lossId': 0,
    'regWeight': 0.0005,
    'dsWeight': 5.0,
}


# TensorBoard: collect training images
def collect_vis_img(logit_d, logit_n, npr, normal, depth, shape_mask, ds, cl_inv_mask, fm,
                    fm_inv, selm, vdotn, mask2d):
    with tf.name_scope('collect_train_img') as _:
        mask_crop = slice_tensor(shape_mask, logit_n)
        mask_crop3 = tf.tile(mask_crop, [1, 1, 1, 3])
        mask3 = tf.tile(shape_mask, [1, 1, 1, 3])

        logit_n = logit_n * mask_crop3
        logit_d = logit_d * mask_crop

        gt_normal = slice_tensor(normal * mask3, logit_n)
        gt_depth = slice_tensor(depth * shape_mask, logit_d)

        npr_lines = slice_tensor(npr, logit_n)
        cl_inv_mask = slice_tensor(cl_inv_mask, logit_n)

        feature_mask = slice_tensor(fm, logit_n)
        feature_mask_inv = slice_tensor(fm_inv, logit_n)
        sel_mask = slice_tensor(selm, logit_n)
        vdotn_scalar = slice_tensor(vdotn, logit_n)

    train_npr_proto = tf.summary.image('train_npr_lines', npr_lines, hyper_params['nbDispImg'])
    train_gtn_proto = tf.summary.image('train_gt_normal', gt_normal, hyper_params['nbDispImg'])
    train_gtd_proto = tf.summary.image('train_gt_depth', gt_depth, hyper_params['nbDispImg'])
    train_fn_proto = tf.summary.image('train_out_normal', logit_n, hyper_params['nbDispImg'])
    train_fd_proto = tf.summary.image('train_out_depth', logit_d, hyper_params['nbDispImg'])
    train_gtds_proto = tf.summary.image('train_gt_ds', ds, hyper_params['nbDispImg'])
    train_cl_mask_inv_proto = tf.summary.image('train_clmask_inv', cl_inv_mask, hyper_params['nbDispImg'])
    train_fm_proto = tf.summary.image('train_feature_mask', feature_mask, hyper_params['nbDispImg'])
    train_fm_inv_proto = tf.summary.image('train_feature_mask_inv', feature_mask_inv, hyper_params['nbDispImg'])
    train_selm_proto = tf.summary.image('train_sel_mask', sel_mask, hyper_params['nbDispImg'])
    train_vdotn_proto = tf.summary.image('train_vdotn_scalar', vdotn_scalar, hyper_params['nbDispImg'])
    train_mask_proto = tf.summary.image('train_shapeMask', shape_mask, hyper_params['nbDispImg'])
    train_mask2d_proto = tf.summary.image('train_2dMask', mask2d, hyper_params['nbDispImg'])

    return [train_npr_proto, train_gtn_proto, train_gtd_proto, train_fn_proto, train_fd_proto, train_gtds_proto,
            train_cl_mask_inv_proto, train_fm_proto, train_fm_inv_proto, train_selm_proto,
            train_vdotn_proto, train_mask_proto, train_mask2d_proto]


# TensorBoard: collect evaluating images
def collect_vis_img_val(logit_d, logit_n, npr, normal, depth, shape_mask, ds, cl_inv_mask, fm,
                        fm_inv, selm, vdotn, mask2d):
    with tf.name_scope('collect_val_img') as _:
        mask_crop = slice_tensor(shape_mask, logit_n)
        mask_crop3 = tf.tile(mask_crop, [1, 1, 1, 3])
        mask3 = tf.tile(shape_mask, [1, 1, 1, 3])

        logit_n = logit_n * mask_crop3
        logit_d = logit_d * mask_crop

        gt_normal = slice_tensor(normal * mask3, logit_n)
        gt_depth = slice_tensor(depth * shape_mask, logit_d)

        npr_lines = slice_tensor(npr, logit_n)
        cl_inv_mask = slice_tensor(cl_inv_mask, logit_n)

        feature_mask = slice_tensor(fm, logit_n)
        feature_mask_inv = slice_tensor(fm_inv, logit_n)
        sel_mask = slice_tensor(selm, logit_n)
        vdotn_scalar = slice_tensor(vdotn, logit_n)

    val_npr_proto = tf.summary.image('val_npr_lines', npr_lines, hyper_params['nbDispImg'])
    val_gtn_proto = tf.summary.image('val_gt_normal', gt_normal, hyper_params['nbDispImg'])
    val_gtd_proto = tf.summary.image('val_gt_depth', gt_depth, hyper_params['nbDispImg'])
    val_fn_proto = tf.summary.image('val_out_normal', logit_n, hyper_params['nbDispImg'])
    val_fd_proto = tf.summary.image('val_out_depth', logit_d, hyper_params['nbDispImg'])
    val_gtds_proto = tf.summary.image('val_gt_ds', ds, hyper_params['nbDispImg'])
    val_cl_mask_inv_proto = tf.summary.image('val_clmask_inv', cl_inv_mask, hyper_params['nbDispImg'])
    val_fm_proto = tf.summary.image('val_feature_mask', feature_mask, hyper_params['nbDispImg'])
    val_fm_inv_proto = tf.summary.image('val_feature_mask_inv', feature_mask_inv, hyper_params['nbDispImg'])
    val_selm_proto = tf.summary.image('val_sel_mask', sel_mask, hyper_params['nbDispImg'])
    val_vdotn_proto = tf.summary.image('val_vdotn_scalar', vdotn_scalar, hyper_params['nbDispImg'])
    val_mask_proto = tf.summary.image('val_shapeMask', shape_mask, hyper_params['nbDispImg'])
    val_mask2d_proto = tf.summary.image('val_2dMask', mask2d, hyper_params['nbDispImg'])

    return [val_npr_proto, val_gtn_proto, val_gtd_proto, val_fn_proto, val_fd_proto, val_gtds_proto,
            val_cl_mask_inv_proto, val_fm_proto, val_fm_inv_proto, val_selm_proto, val_vdotn_proto,
            val_mask_proto, val_mask2d_proto]


# depth, normal regularization term
def reg_loss(logit_n, logit_d, shape_mask, cl_mask_inverse, fl_mask_inv, scope='reg_loss'):
    with tf.name_scope(scope) as _:
        # convert normal signal back to [-1, 1]
        converted_n = (logit_n * 2.0) - 1.0

        img_shape = logit_d.get_shape().as_list()
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        K = 0.007843137254902

        shape_mask_crop = slice_tensor(shape_mask, logit_d)
        l_mask_crop = slice_tensor(cl_mask_inverse, logit_d)
        fl_mask_inv_crop = slice_tensor(fl_mask_inv, logit_d)
        combined_mask = shape_mask_crop * l_mask_crop * fl_mask_inv_crop
        mask_shift_x = tf.slice(combined_mask, [0, 0, 0, 0], [-1, -1, W - 1, -1])
        mask_shift_y = tf.slice(combined_mask, [0, 0, 0, 0], [-1, H - 1, -1, -1])

        c0 = tf.constant(K, shape=[N, H, W - 1, 1])
        c1 = tf.zeros(shape=[N, H, W - 1, 1])
        cx = logit_d[:, :, 1:, :] - logit_d[:, :, :-1, :]
        t_x = tf.concat([c0, c1, cx], axis=3)
        # approximate normalization
        t_x /= K

        c2 = tf.zeros(shape=[N, H - 1, W, 1])
        c3 = tf.constant(K, shape=[N, H - 1, W, 1])
        cy = logit_d[:, 1:, :, :] - logit_d[:, :-1, :, :]
        t_y = tf.concat([c2, c3, cy], axis=3)
        # approximate normalization
        t_y /= K

        normal_shift_x = tf.slice(converted_n, [0, 0, 0, 0], [-1, -1, W - 1, -1])
        normal_shift_y = tf.slice(converted_n, [0, 0, 0, 0], [-1, H - 1, -1, -1])

        reg_loss1_diff = tf.reduce_sum(t_x * normal_shift_x, 3)
        reg_loss1 = tf.losses.mean_squared_error(tf.zeros(shape=[N, H, W - 1]), reg_loss1_diff,
                                                 weights=tf.squeeze(mask_shift_x, [3]))

        reg_loss2_diff = tf.reduce_sum(t_y * normal_shift_y, 3)
        reg_loss2 = tf.losses.mean_squared_error(tf.zeros(shape=[N, H - 1, W]), reg_loss2_diff,
                                                 weights=tf.squeeze(mask_shift_y, [3]))

        return reg_loss1 + reg_loss2


# total loss
def loss(logit_d, logit_n, normal, depth, shape_mask, ds_mask, cl_mask_inv, reg_weight, fl_mask_inv,
         mode=0, scope='loss'):
    with tf.name_scope(scope) as _:
        mask_crop = slice_tensor(shape_mask, logit_n)
        mask_crop3 = tf.tile(mask_crop, [1, 1, 1, 3])

        # normal loss (l2)
        gt_normal = slice_tensor(normal, logit_n)
        n_loss = tf.losses.mean_squared_error(gt_normal, logit_n, weights=mask_crop3)

        real_n_loss = tf.losses.absolute_difference(gt_normal, logit_n, weights=mask_crop3)

        # depth loss (l2)
        gt_depth = slice_tensor(depth, logit_n)
        d_loss = tf.losses.mean_squared_error(gt_depth, logit_d, weights=mask_crop)

        real_d_loss = tf.losses.absolute_difference(gt_depth, logit_d, weights=mask_crop)

        # depth sample loss (l2)
        d_mask_crop = slice_tensor(ds_mask, logit_n)
        ds_loss = tf.constant(0.0, dtype=tf.float32, shape=[])
        if mode == 1 or mode == 3:
            ds_loss = tf.losses.mean_squared_error(gt_depth, logit_d, weights=d_mask_crop)

        # regularization loss
        r_loss = tf.constant(0.0, dtype=tf.float32, shape=[])
        if mode == 2 or mode == 3:
            r_loss = reg_loss(logit_n, logit_d, shape_mask, cl_mask_inv, fl_mask_inv)

        total_loss = 0.76 * d_loss + n_loss + hyper_params['dsWeight'] * ds_loss + reg_weight * r_loss

        return total_loss, d_loss, n_loss, real_d_loss, real_n_loss, r_loss, ds_loss


# multiple GPUs training
def average_gradient(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses_full):
    return_averaged_losses = []
    for per_loss_in_one_tower in tower_losses_full:
        losses = []
        for tower_loss in per_loss_in_one_tower:
            expand_loss = tf.expand_dims(tower_loss, 0)
            losses.append(expand_loss)

        average_loss = tf.concat(losses, axis=0)
        average_loss = tf.reduce_mean(average_loss, 0)

        return_averaged_losses.append(average_loss)

    return return_averaged_losses


# training process
def train_procedure(net, train_records, reg_weight):
    nb_gpus = hyper_params['nb_gpus']

    # Load data
    with tf.name_scope('train_inputs') as _:
        bSize = hyper_params['batchSize'] * nb_gpus
        nbThreads = hyper_params['nbThreads'] * nb_gpus
        reader = SketchReader(tfrecord_list=train_records, raw_size=[256, 256, 25], shuffle=True,
                              num_threads=nbThreads, batch_size=bSize)
        raw_input = reader.next_batch()

        npr_lines, ds, _, fm, fm_inv, gt_normal, gt_depth, _, mask_cline_inv, mask_shape, mask_ds, _, _, \
        mask_2d, sel_mask, vdotn_scalar = net.cook_raw_inputs(raw_input)

    # initialize optimizer
    opt = tf.train.AdamOptimizer()

    # split data
    with tf.name_scope('divide_data'):
        gpu_npr_lines = tf.split(npr_lines, nb_gpus, axis=0)
        gpu_ds = tf.split(ds, nb_gpus, axis=0)
        gpu_mask2d = tf.split(mask_2d, nb_gpus, axis=0)
        gpu_fm = tf.split(fm, nb_gpus, axis=0)
        gpu_fm_inv = tf.split(fm_inv, nb_gpus, axis=0)
        gpu_gt_normal = tf.split(gt_normal, nb_gpus, axis=0)
        gpu_gt_depth = tf.split(gt_depth, nb_gpus, axis=0)
        gpu_mask_shape = tf.split(mask_shape, nb_gpus, axis=0)
        gpu_mask_ds = tf.split(mask_ds, nb_gpus, axis=0)
        gpu_cl_mask_inv = tf.split(mask_cline_inv, nb_gpus, axis=0)
        gpu_sel_mask = tf.split(sel_mask, nb_gpus, axis=0)
        gpu_vdotn_scalar = tf.split(vdotn_scalar, nb_gpus, axis=0)

    tower_grads = []
    tower_loss_collected = []
    tower_total_losses = []
    tower_d_losses = []
    tower_n_losses = []
    tower_abs_d_losses = []
    tower_abs_n_losses = []
    tower_r_losses = []
    tower_ds_losses = []

    # TensorBoard: images
    gpu0_npr_lines_imgs = None
    gpu0_gt_ds_imgs = None
    gpu0_logit_d_imgs = None
    gpu0_logit_n_imgs = None
    gpu0_gt_normal_imgs = None
    gpu0_gt_depth_imgs = None
    gpu0_shape_mask_imgs = None
    gpu0_clinv_mask_imgs = None
    gpu0_fm_imgs = None
    gpu0_fm_inv_imgs = None
    gpu0_sel_mask_imgs = None
    gpu0_vdotn_scalar_imgs = None
    gpu0_mask2d_imgs = None

    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(nb_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower_%s' % gpu_id) as _:
                    # Network forward
                    logit_d, logit_n, _ = net.load_dn_naive_net(gpu_npr_lines[gpu_id],
                                                                gpu_ds[gpu_id],
                                                                gpu_mask2d[gpu_id],
                                                                gpu_fm[gpu_id],
                                                                gpu_sel_mask[gpu_id],
                                                                gpu_vdotn_scalar[gpu_id],
                                                                hyper_params['rootFt'],
                                                                is_training=True)

                    # Training loss
                    train_loss, train_d_loss, train_n_loss, train_real_dloss, train_real_nloss, train_r_loss, \
                    train_ds_loss = loss(logit_d,
                                         logit_n,
                                         gpu_gt_normal[gpu_id],
                                         gpu_gt_depth[gpu_id],
                                         gpu_mask_shape[gpu_id],
                                         gpu_mask_ds[gpu_id],
                                         gpu_cl_mask_inv[gpu_id],
                                         reg_weight,
                                         gpu_fm_inv[gpu_id],
                                         mode=hyper_params['lossId'],
                                         scope='train_loss')

                    # reuse variables
                    tf.get_variable_scope().reuse_variables()

                    # collect gradients and every loss
                    tower_grads.append(opt.compute_gradients(train_loss))
                    tower_total_losses.append(train_loss)
                    tower_d_losses.append(train_d_loss)
                    tower_n_losses.append(train_n_loss)
                    tower_abs_d_losses.append(train_real_dloss)
                    tower_abs_n_losses.append(train_real_nloss)
                    tower_r_losses.append(train_r_loss)
                    tower_ds_losses.append(train_ds_loss)

                    # TensorBoard: collect images from GPU 0
                    if gpu_id == 0:
                        gpu0_npr_lines_imgs = gpu_npr_lines[gpu_id]
                        gpu0_gt_ds_imgs = gpu_ds[gpu_id]
                        gpu0_logit_d_imgs = logit_d
                        gpu0_logit_n_imgs = logit_n
                        gpu0_gt_normal_imgs = gpu_gt_normal[gpu_id]
                        gpu0_gt_depth_imgs = gpu_gt_depth[gpu_id]
                        gpu0_shape_mask_imgs = gpu_mask_shape[gpu_id]
                        gpu0_clinv_mask_imgs = gpu_cl_mask_inv[gpu_id]
                        gpu0_fm_imgs = gpu_fm[gpu_id]
                        gpu0_fm_inv_imgs = gpu_fm_inv[gpu_id]
                        gpu0_sel_mask_imgs = gpu_sel_mask[gpu_id]
                        gpu0_vdotn_scalar_imgs = gpu_vdotn_scalar[gpu_id]
                        gpu0_mask2d_imgs = gpu_mask2d[gpu_id]

        tower_loss_collected.append(tower_total_losses)
        tower_loss_collected.append(tower_d_losses)
        tower_loss_collected.append(tower_n_losses)
        tower_loss_collected.append(tower_abs_d_losses)
        tower_loss_collected.append(tower_abs_n_losses)
        tower_loss_collected.append(tower_r_losses)
        tower_loss_collected.append(tower_ds_losses)

    # Solver
    with tf.name_scope('solve') as _:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = average_gradient(tower_grads)
            averaged_losses = average_losses(tower_loss_collected)
            apply_gradient_op = opt.apply_gradients(grads)
            train_op = tf.group(apply_gradient_op)

    # TensorBoard: visualization
    train_diff_proto = tf.summary.scalar('Training_TotalLoss', averaged_losses[0])
    train_diff_d_proto = tf.summary.scalar('Training_DepthL2Loss', averaged_losses[1])
    train_diff_n_proto = tf.summary.scalar('Training_NormalL2Loss', averaged_losses[2])
    train_diff_reald_proto = tf.summary.scalar('Training_RealDLoss', averaged_losses[3])
    train_diff_realn_proto = tf.summary.scalar('Training_RealNLoss', averaged_losses[4])
    train_diff_reg_proto = tf.summary.scalar('Training_RegL2Loss', averaged_losses[5])
    train_diff_ds_protp = tf.summary.scalar('Training_DsL2Loss', averaged_losses[6])

    proto_list = collect_vis_img(gpu0_logit_d_imgs,
                                 gpu0_logit_n_imgs,
                                 gpu0_npr_lines_imgs,
                                 gpu0_gt_normal_imgs,
                                 gpu0_gt_depth_imgs,
                                 gpu0_shape_mask_imgs,
                                 gpu0_gt_ds_imgs,
                                 gpu0_clinv_mask_imgs,
                                 gpu0_fm_imgs,
                                 gpu0_fm_inv_imgs,
                                 gpu0_sel_mask_imgs,
                                 gpu0_vdotn_scalar_imgs,
                                 gpu0_mask2d_imgs)

    proto_list.append(train_diff_proto)
    proto_list.append(train_diff_d_proto)
    proto_list.append(train_diff_n_proto)
    proto_list.append(train_diff_reald_proto)
    proto_list.append(train_diff_realn_proto)
    proto_list.append(train_diff_reg_proto)
    proto_list.append(train_diff_ds_protp)
    merged_train = tf.summary.merge(proto_list)

    return merged_train, train_op, averaged_losses[0]


# validation process
def validation_procedure(net, val_records, reg_weight):
    # Load data
    with tf.name_scope('eval_inputs') as _:
        reader = SketchReader(tfrecord_list=val_records, raw_size=[256, 256, 25], shuffle=False,
                              num_threads=hyper_params['nbThreads'], batch_size=hyper_params['batchSize'])
        raw_input = reader.next_batch()

        npr_lines, ds, _, fm, fm_inv, gt_normal, gt_depth, _, mask_cline_inv, mask_shape, mask_ds, _, _, \
        mask_2d, sel_mask, vdotn_scalar = net.cook_raw_inputs(raw_input)

    # Network forward
    logit_d, logit_n, _ = net.load_dn_naive_net(npr_lines,
                                                ds,
                                                mask_2d,
                                                fm,
                                                sel_mask,
                                                vdotn_scalar,
                                                hyper_params['rootFt'],
                                                is_training=False,
                                                reuse=True)

    # Validate loss
    val_loss, val_d_loss, val_n_loss, val_real_dloss, val_real_nloss, val_r_loss, val_ds_loss \
        = loss(logit_d,
               logit_n,
               gt_normal,
               gt_depth,
               mask_shape,
               mask_ds,
               mask_cline_inv,
               reg_weight,
               fm_inv,
               mode=hyper_params['lossId'],
               scope='test_loss')

    # TensorBoard
    proto_list = collect_vis_img_val(logit_d,
                                     logit_n,
                                     npr_lines,
                                     gt_normal,
                                     gt_depth,
                                     mask_shape,
                                     ds,
                                     mask_cline_inv,
                                     fm,
                                     fm_inv,
                                     sel_mask,
                                     vdotn_scalar,
                                     mask_2d)
    merged_val = tf.summary.merge(proto_list)

    return merged_val, val_loss, val_d_loss, val_n_loss, val_real_dloss, val_real_nloss, val_r_loss, val_ds_loss


def train_net():
    # Set logging
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # Load network
    net = SKETCHNET()

    # regularization weight
    reg_weight_value = tf.placeholder(tf.float32, name='reg_weight')

    # Train
    train_data_records = [item for item in os.listdir(hyper_params['dbTrain']) if item.endswith('.tfrecords')]
    train_records = [os.path.join(hyper_params['dbTrain'], item) for item in train_data_records if
                     item.find('train') != -1]
    train_summary, train_step, train_loss = train_procedure(net, train_records, reg_weight=reg_weight_value)

    # Validation
    val_data_records = [item for item in os.listdir(hyper_params['dbEval']) if item.endswith('.tfrecords')]
    val_records = [os.path.join(hyper_params['dbEval'], item) for item in val_data_records if
                   item.find('eval') != -1]
    num_eval_samples = sum(1 for _ in tf.python_io.tf_record_iterator(val_records[0]))
    num_eval_itr = num_eval_samples // hyper_params['batchSize']
    num_eval_itr += 1

    val_proto, val_loss, val_d_loss, val_n_loss, val_real_dloss, val_real_nloss, val_reg_loss, val_ds_loss \
        = validation_procedure(net, val_records, reg_weight=reg_weight_value)

    valid_loss = tf.placeholder(tf.float32, name='val_loss')
    valid_loss_proto = tf.summary.scalar('Validating_TotalLoss', valid_loss)
    valid_d_loss = tf.placeholder(tf.float32, name='val_d_loss')
    valid_d_loss_proto = tf.summary.scalar('Validating_DepthL2Loss', valid_d_loss)
    valid_n_loss = tf.placeholder(tf.float32, name='val_n_loss')
    valid_n_loss_proto = tf.summary.scalar('Validating_NormalL2Loss', valid_n_loss)
    valid_real_dloss = tf.placeholder(tf.float32, name='val_real_dloss')
    valid_real_dloss_proto = tf.summary.scalar('Validating_RealDLoss', valid_real_dloss)
    valid_real_nloss = tf.placeholder(tf.float32, name='val_real_nloss')
    valid_real_nloss_proto = tf.summary.scalar('Validating_RealNLoss', valid_real_nloss)
    valid_reg_loss = tf.placeholder(tf.float32, name='val_reg_loss')
    valid_reg_loss_proto = tf.summary.scalar('Validating_RegL2Loss', valid_reg_loss)
    valid_ds_loss = tf.placeholder(tf.float32, name='val_ds_loss')
    valid_ds_loss_proto = tf.summary.scalar('Validating_DsL2Loss', valid_ds_loss)
    valid_value_merge = tf.summary.merge(
        [valid_loss_proto, valid_d_loss_proto, valid_n_loss_proto, valid_real_dloss_proto,
         valid_real_nloss_proto, valid_reg_loss_proto, valid_ds_loss_proto])

    # Saver
    tf_saver = tf.train.Saver(max_to_keep=100)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        # TF summary
        train_writer = tf.summary.FileWriter(output_folder + '/train', sess.graph)

        # initialize
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_logger.info('pre-load data to fill data buffer...')

        # reg_weight init
        cur_weight = hyper_params['regWeight']
        for titr in range(hyper_params['maxIter']):

            # update regularization weight
            # # 4 gpus
            # if titr % 5000 == 0 and titr > 0:
            #     cur_weight *= 3.985
            #     cur_weight = min(cur_weight, 2.0)
            # 2 gpus
            if titr % 10000 == 0 and titr > 0:
                cur_weight *= 3.275
                cur_weight = min(cur_weight, 2.0)
            # # 1 gup
            # if titr % 10000 == 0 and titr > 0:
            #     cur_weight *= 1.738
            #     cur_weight = min(cur_weight, 2.0)

            # Validation
            if titr % hyper_params['exeValStep'] == 0:
                idx = randint(0, num_eval_itr - 1)
                avg_loss = 0.0
                avg_d_loss = 0.0
                avg_n_loss = 0.0
                avg_real_dloss = 0.0
                avg_real_nloss = 0.0
                avg_reg_loss = 0.0
                avg_ds_loss = 0.0
                for eitr in range(num_eval_itr):
                    if eitr == idx:
                        val_merge, cur_v_loss, cur_vd_loss, cur_vn_loss, cur_real_dloss, cur_real_nloss, cur_reg_loss, \
                        cur_ds_loss = sess.run([val_proto, val_loss, val_d_loss, val_n_loss, val_real_dloss,
                                                val_real_nloss, val_reg_loss, val_ds_loss],
                                               feed_dict={'reg_weight:0': cur_weight})
                        train_writer.add_summary(val_merge, titr)
                    else:
                        cur_v_loss, cur_vd_loss, cur_vn_loss, cur_real_dloss, cur_real_nloss, cur_reg_loss, cur_ds_loss \
                            = sess.run([val_loss, val_d_loss, val_n_loss, val_real_dloss, val_real_nloss,
                                        val_reg_loss, val_ds_loss],
                                       feed_dict={'reg_weight:0': cur_weight})

                    avg_loss += cur_v_loss
                    avg_d_loss += cur_vd_loss
                    avg_n_loss += cur_vn_loss
                    avg_real_dloss += cur_real_dloss
                    avg_real_nloss += cur_real_nloss
                    avg_reg_loss += cur_reg_loss
                    avg_ds_loss += cur_ds_loss

                avg_loss /= num_eval_itr
                avg_d_loss /= num_eval_itr
                avg_n_loss /= num_eval_itr
                avg_real_dloss /= num_eval_itr
                avg_real_nloss /= num_eval_itr
                avg_reg_loss /= num_eval_itr
                avg_ds_loss /= num_eval_itr

                valid_summary = sess.run(valid_value_merge,
                                         feed_dict={'val_loss:0': avg_loss,
                                                    'val_d_loss:0': avg_d_loss,
                                                    'val_n_loss:0': avg_n_loss,
                                                    'val_real_dloss:0': avg_real_dloss,
                                                    'val_real_nloss:0': avg_real_nloss,
                                                    'val_reg_loss:0': avg_reg_loss,
                                                    'val_ds_loss:0': avg_ds_loss})
                train_writer.add_summary(valid_summary, titr)
                train_logger.info('Validation loss at step {} is: {}'.format(titr, avg_loss))

            # Save model
            if titr % hyper_params['saveModelStep'] == 0:
                tf_saver.save(sess, hyper_params['outDir'] + '/savedModel/my_model{:d}.ckpt'.format(titr))
                train_logger.info('Save model at step: {:d}, reg weight: {}'.format(titr, cur_weight))

            # Training
            t_summary, _, t_loss = sess.run([train_summary, train_step, train_loss],
                                            feed_dict={'reg_weight:0': cur_weight})

            # Display
            if titr % hyper_params['dispLossStep'] == 0:
                train_writer.add_summary(t_summary, titr)
                train_logger.info('Training loss at step {} is: {}'.format(titr, t_loss))

        # Finish training
        coord.request_stop()
        coord.join(threads)

        # Release resource
        train_writer.close()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbTrain', required=True, help='training dataset directory', type=str)
    parser.add_argument('--dbEval', required=True, help='evaluation dataset directory', type=str)
    parser.add_argument('--outDir', required=True, help='otuput directory', type=str)
    parser.add_argument('--nb_gpus', help='GPU number', type=int, default=1)
    parser.add_argument('--devices', help='GPU device indices', type=str, default='0')
    parser.add_argument('--lossId', help='training loss types - 0: naiveNet, Edn; 3: ablationNet, Edn_Eds_Ereg',
                        type=int, default=0)

    args = parser.parse_args()
    hyper_params['dbTrain'] = args.dbTrain
    hyper_params['dbEval'] = args.dbEval
    hyper_params['outDir'] = args.outDir
    hyper_params['nb_gpus'] = args.nb_gpus
    hyper_params['device'] = args.devices
    hyper_params['lossId'] = args.lossId

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = hyper_params['device']

    # Set output folder
    output_folder = hyper_params['outDir']
    make_dir(output_folder)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Training preparation
    logger.info('---Training preparation: ---')

    # Dump parameters
    dump_params(hyper_params['outDir'], hyper_params)

    # Begin training
    train_net()

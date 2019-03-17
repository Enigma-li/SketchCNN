#
# Project SketchCNN
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2018. All Rights Reserved.
#
# ==============================================================================
"""Direction field regression network training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
from random import randint
import argparse

import tensorflow as tf
from SketchCNN.script.network import SKETCHNET
from SketchCNN.script.loader import SketchReader
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
    'smoothWeight': 0.1,
}

nprLine_input = tf.placeholder(tf.float32, [None, None, None, 1], name='npr_input')
ds_input = tf.placeholder(tf.float32, [None, None, None, 1], name='ds_input')
fm_input = tf.placeholder(tf.float32, [None, None, None, 1], name='fLMask_input')
fmInv_input = tf.placeholder(tf.float32, [None, None, None, 1], name='fLInvMask_input')
gtField_input = tf.placeholder(tf.float32, [None, None, None, 3], name='gtField_input')
clineInvMask_input = tf.placeholder(tf.float32, [None, None, None, 1], name='clIMask_input')
maskShape_input = tf.placeholder(tf.float32, [None, None, None, 1], name='shapeMask_input')
mask2D_input = tf.placeholder(tf.float32, [None, None, None, 1], name='2dMask_input')
selLineMask_input = tf.placeholder(tf.float32, [None, None, None, 1], name='sLMask_input')
vdotnScalar_input = tf.placeholder(tf.float32, [None, None, None, 1], name='curvMag_input')


# TensorBoard: collect training images
def collect_vis_img(logit_f, npr_lines, gt_field, shape_mask, line_inv, ds, fm, sel_m, vdotn, fm_inv):
    with tf.name_scope('collect_train_img') as _:
        mask_crop = slice_tensor(shape_mask, logit_f)
        l_mask_crop = slice_tensor(line_inv, logit_f)
        combined_mask = mask_crop * l_mask_crop
        combined_mask = tf.tile(combined_mask, [1, 1, 1, 4])

        logit_f = logit_f * combined_mask
        gt_field = slice_tensor(gt_field, logit_f) * combined_mask

        cur_shape = logit_f.get_shape().as_list()
        lc = tf.zeros([cur_shape[0], cur_shape[1], cur_shape[2], 1], tf.float32)

        f_coeff_a = tf.concat([tf.slice(logit_f, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
        f_coeff_b = tf.concat([tf.slice(logit_f, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)

        gt_coeff_a = tf.concat([tf.slice(gt_field, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
        gt_coeff_b = tf.concat([tf.slice(gt_field, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)

        npr_lines = slice_tensor(npr_lines, logit_f)
        line_inv = slice_tensor(line_inv, logit_f)

        depth_sample = slice_tensor(ds, logit_f)
        feature_mask = slice_tensor(fm, logit_f)
        feature_mask_inv = slice_tensor(fm_inv, logit_f)
        sel_mask = slice_tensor(sel_m, logit_f)
        vdotn_scalar = slice_tensor(vdotn, logit_f)

    train_npr_proto = tf.summary.image('train_npr_lines', npr_lines, hyper_params['nbDispImg'])
    train_gt_coeff_a_proto = tf.summary.image('train_gt_a', gt_coeff_a, hyper_params['nbDispImg'])
    train_gt_coeff_b_proto = tf.summary.image('train_gt_b', gt_coeff_b, hyper_params['nbDispImg'])
    train_f_coeff_a_proto = tf.summary.image('train_f_a', f_coeff_a, hyper_params['nbDispImg'])
    train_f_coeff_b_proto = tf.summary.image('train_f_b', f_coeff_b, hyper_params['nbDispImg'])
    train_mask_proto = tf.summary.image('train_mask', line_inv, hyper_params['nbDispImg'])
    train_ds_proto = tf.summary.image('train_ds', depth_sample, hyper_params['nbDispImg'])
    train_fm_proto = tf.summary.image('train_feature_mask', feature_mask, hyper_params['nbDispImg'])
    train_fm_inv_proto = tf.summary.image('train_feature_mask_inv', feature_mask_inv, hyper_params['nbDispImg'])
    train_selm_proto = tf.summary.image('train_sel_mask', sel_mask, hyper_params['nbDispImg'])
    train_vdotn_proto = tf.summary.image('train_vdotn_scalar', vdotn_scalar, hyper_params['nbDispImg'])

    return [train_npr_proto, train_gt_coeff_a_proto, train_gt_coeff_b_proto, train_f_coeff_a_proto,
            train_f_coeff_b_proto, train_mask_proto, train_ds_proto, train_fm_proto, train_fm_inv_proto,
            train_selm_proto, train_vdotn_proto]


# TensorBoard: collect evaluating images
def collect_vis_img_val(logit_f, npr_lines, gt_field, shape_mask, line_inv, ds, fm, sel_m, vdotn, fm_inv):
    with tf.name_scope('collect_val_img') as _:
        mask_crop = slice_tensor(shape_mask, logit_f)
        l_mask_crop = slice_tensor(line_inv, logit_f)
        combined_mask = mask_crop * l_mask_crop
        combined_mask = tf.tile(combined_mask, [1, 1, 1, 4])

        logit_f = logit_f * combined_mask
        gt_field = slice_tensor(gt_field, logit_f) * combined_mask

        cur_shape = logit_f.get_shape().as_list()
        lc = tf.zeros([cur_shape[0], cur_shape[1], cur_shape[2], 1], tf.float32)

        f_coeff_a = tf.concat([tf.slice(logit_f, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
        f_coeff_b = tf.concat([tf.slice(logit_f, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)

        gt_coeff_a = tf.concat([tf.slice(gt_field, [0, 0, 0, 0], [-1, -1, -1, 2]), lc], axis=3)
        gt_coeff_b = tf.concat([tf.slice(gt_field, [0, 0, 0, 2], [-1, -1, -1, 2]), lc], axis=3)

        npr_lines = slice_tensor(npr_lines, logit_f)
        line_inv = slice_tensor(line_inv, logit_f)

        depth_sample = slice_tensor(ds, logit_f)
        feature_mask = slice_tensor(fm, logit_f)
        feature_mask_inv = slice_tensor(fm_inv, logit_f)
        sel_mask = slice_tensor(sel_m, logit_f)
        vdotn_scalar = slice_tensor(vdotn, logit_f)

    val_npr_proto = tf.summary.image('val_npr_lines', npr_lines, hyper_params['nbDispImg'])
    val_gt_coeff_a_proto = tf.summary.image('val_gt_a', gt_coeff_a, hyper_params['nbDispImg'])
    val_gt_coeff_b_proto = tf.summary.image('val_gt_b', gt_coeff_b, hyper_params['nbDispImg'])
    val_f_coeff_a_proto = tf.summary.image('val_f_a', f_coeff_a, hyper_params['nbDispImg'])
    val_f_coeff_b_proto = tf.summary.image('val_f_b', f_coeff_b, hyper_params['nbDispImg'])
    val_mask_proto = tf.summary.image('val_mask_img', line_inv, hyper_params['nbDispImg'])
    val_ds_proto = tf.summary.image('val_ds', depth_sample, hyper_params['nbDispImg'])
    val_fm_proto = tf.summary.image('val_feature_mask', feature_mask, hyper_params['nbDispImg'])
    val_fm_inv_proto = tf.summary.image('val_feature_mask_inv', feature_mask_inv, hyper_params['nbDispImg'])
    val_selm_proto = tf.summary.image('val_sel_mask', sel_mask, hyper_params['nbDispImg'])
    val_vdotn_proto = tf.summary.image('val_vdotn_scalar', vdotn_scalar, hyper_params['nbDispImg'])

    return [val_npr_proto, val_gt_coeff_a_proto, val_gt_coeff_b_proto, val_f_coeff_a_proto, val_f_coeff_b_proto,
            val_mask_proto, val_ds_proto, val_fm_proto, val_fm_inv_proto, val_selm_proto, val_vdotn_proto]


# total loss
def loss(logit_f, gt_field, shape_mask, l_mask_inverse, fl_mask_inv, scope='loss'):
    with tf.name_scope(scope) as _:
        mask_crop = slice_tensor(shape_mask, logit_f)
        l_mask_crop = slice_tensor(l_mask_inverse, logit_f)
        fl_mask_inv_crop = slice_tensor(fl_mask_inv, logit_f)
        combined_smooth_mask = mask_crop * l_mask_crop * fl_mask_inv_crop
        combined_smooth_mask = tf.tile(combined_smooth_mask, [1, 1, 1, 4])

        combined_mask = mask_crop * l_mask_crop
        combined_mask = tf.tile(combined_mask, [1, 1, 1, 4])

        with tf.name_scope('data_term'):
            # data term
            gt_field = slice_tensor(gt_field, logit_f)
            f_loss = tf.losses.absolute_difference(gt_field, logit_f, weights=combined_mask)

        with tf.name_scope('smoothness_term'):
            # smoothness term
            img_shape = logit_f.get_shape().as_list()
            H = img_shape[1]
            W = img_shape[2]

            pixel_dif1 = logit_f[:, 1:, :, :] - logit_f[:, :-1, :, :]
            pixel_dif2 = logit_f[:, :, 1:, :] - logit_f[:, :, :-1, :]
            mask_shift1 = tf.slice(combined_smooth_mask, [0, 0, 0, 0], [-1, H - 1, -1, -1])
            mask_shift2 = tf.slice(combined_smooth_mask, [0, 0, 0, 0], [-1, -1, W - 1, -1])

            var_loss1 = tf.losses.compute_weighted_loss(tf.abs(pixel_dif1), weights=mask_shift1)
            var_loss2 = tf.losses.compute_weighted_loss(tf.abs(pixel_dif2), weights=mask_shift2)
            tot_var_mean = var_loss1 + var_loss2

        tot_loss = f_loss + hyper_params['smoothWeight'] * tot_var_mean

    return tot_loss, f_loss, tot_var_mean


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
def train_procedure(net, train_records):
    nb_gpus = hyper_params['nb_gpus']

    # Load data
    with tf.name_scope('train_inputs') as _:
        bSize = hyper_params['batchSize'] * nb_gpus
        nbThreads = hyper_params['nbThreads'] * nb_gpus
        reader = SketchReader(tfrecord_list=train_records, raw_size=[256, 256, 25], shuffle=True,
                              num_threads=nbThreads, batch_size=bSize)
        raw_input = reader.next_batch()

        npr_lines, ds, _, fm, fm_inv, _, _, gt_field, cline_mask_inv, mask_shape, _, _, _, mask_2d, selm, vdotn \
            = net.cook_raw_inputs(raw_input)

    # initialize optimizer
    opt = tf.train.AdamOptimizer()

    # split data
    with tf.name_scope('divide_data'):
        gpu_npr_lines = tf.split(nprLine_input, nb_gpus, axis=0)
        gpu_mask2d = tf.split(mask2D_input, nb_gpus, axis=0)
        gpu_ds = tf.split(ds_input, nb_gpus, axis=0)
        gpu_fm = tf.split(fm_input, nb_gpus, axis=0)
        gpu_fm_inv = tf.split(fmInv_input, nb_gpus, axis=0)
        gpu_selm = tf.split(selLineMask_input, nb_gpus, axis=0)
        gpu_vdotn = tf.split(vdotnScalar_input, nb_gpus, axis=0)
        gpu_gt_field = tf.split(gtField_input, nb_gpus, axis=0)
        gpu_mask_shape = tf.split(maskShape_input, nb_gpus, axis=0)
        gpu_mask_cline = tf.split(clineInvMask_input, nb_gpus, axis=0)

    tower_grads = []
    tower_loss_collected = []
    tower_total_losses = []
    tower_data_losses = []
    tower_smooth_losses = []

    # TensorBoard: images
    gpu0_npr_lines_imgs = None
    gpu0_logit_f_imgs = None
    gpu0_gt_f_imgs = None
    gpu0_shape_mask_imgs = None
    gpu0_shape_cline_imgs = None
    gpu0_ds_imgs = None
    gpu0_fm_imgs = None
    gpu0_fm_inv_imgs = None
    gpu0_selm_imgs = None
    gpu0_vdotn_imgs = None

    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(nb_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower_%s' % gpu_id) as _:
                    # Network forward
                    logit_f, _ = net.load_field_net(gpu_npr_lines[gpu_id],
                                                    gpu_mask2d[gpu_id],
                                                    gpu_ds[gpu_id],
                                                    gpu_fm[gpu_id],
                                                    gpu_selm[gpu_id],
                                                    gpu_vdotn[gpu_id],
                                                    hyper_params['rootFt'],
                                                    is_training=True)

                    # Training loss
                    train_loss, train_data_loss, train_smooth_loss = loss(logit_f,
                                                                          gpu_gt_field[gpu_id],
                                                                          gpu_mask_shape[gpu_id],
                                                                          gpu_mask_cline[gpu_id],
                                                                          gpu_fm_inv[gpu_id],
                                                                          scope='train_loss')

                    # reuse variables
                    tf.get_variable_scope().reuse_variables()

                    # collect gradients and every loss
                    tower_grads.append(opt.compute_gradients(train_loss))
                    tower_total_losses.append(train_loss)
                    tower_data_losses.append(train_data_loss)
                    tower_smooth_losses.append(train_smooth_loss)

                    # TensorBoard: collect images from GPU 0
                    if gpu_id == 0:
                        gpu0_npr_lines_imgs = gpu_npr_lines[gpu_id]
                        gpu0_logit_f_imgs = logit_f
                        gpu0_gt_f_imgs = gpu_gt_field[gpu_id]
                        gpu0_shape_mask_imgs = gpu_mask_shape[gpu_id]
                        gpu0_shape_cline_imgs = gpu_mask_cline[gpu_id]
                        gpu0_ds_imgs = gpu_ds[gpu_id]
                        gpu0_fm_imgs = gpu_fm[gpu_id]
                        gpu0_fm_inv_imgs = gpu_fm_inv[gpu_id]
                        gpu0_selm_imgs = gpu_selm[gpu_id]
                        gpu0_vdotn_imgs = gpu_vdotn[gpu_id]

        tower_loss_collected.append(tower_total_losses)
        tower_loss_collected.append(tower_data_losses)
        tower_loss_collected.append(tower_smooth_losses)

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
    train_data_loss_proto = tf.summary.scalar('Traning_DataL1Loss', averaged_losses[1])
    train_smooth_loss_proto = tf.summary.scalar('Training_SmoothL1Loss', averaged_losses[2])

    proto_list = collect_vis_img(gpu0_logit_f_imgs,
                                 gpu0_npr_lines_imgs,
                                 gpu0_gt_f_imgs,
                                 gpu0_shape_mask_imgs,
                                 gpu0_shape_cline_imgs,
                                 gpu0_ds_imgs,
                                 gpu0_fm_imgs,
                                 gpu0_selm_imgs,
                                 gpu0_vdotn_imgs,
                                 gpu0_fm_inv_imgs)

    proto_list.append(train_diff_proto)
    proto_list.append(train_data_loss_proto)
    proto_list.append(train_smooth_loss_proto)
    merged_train = tf.summary.merge(proto_list)

    return merged_train, train_op, averaged_losses[0], \
           [npr_lines, ds, fm, fm_inv, gt_field, cline_mask_inv, mask_shape, mask_2d, selm, vdotn]


# validation process
def validation_procedure(net, val_records):
    # Load data
    with tf.name_scope('eval_inputs') as _:
        reader = SketchReader(tfrecord_list=val_records, raw_size=[256, 256, 25], shuffle=False,
                              num_threads=hyper_params['nbThreads'], batch_size=hyper_params['batchSize'])
        raw_input = reader.next_batch()

        npr_lines, ds, _, fm, fm_inv, _, _, gt_field, cline_mask_inv, mask_shape, _, _, _, mask2d, selm, vdotn \
            = net.cook_raw_inputs(raw_input)

    # Network forward
    logit_f, _ = net.load_field_net(nprLine_input,
                                    mask2D_input,
                                    ds_input,
                                    fm_input,
                                    selLineMask_input,
                                    vdotnScalar_input,
                                    hyper_params['rootFt'],
                                    is_training=False,
                                    reuse=True)

    # Validate loss
    val_loss, val_data_loss, val_smooth_loss = loss(logit_f,
                                                    gtField_input,
                                                    maskShape_input,
                                                    clineInvMask_input,
                                                    fmInv_input,
                                                    scope='test_loss')

    # TensorBoard
    proto_list = collect_vis_img_val(logit_f,
                                     nprLine_input,
                                     gtField_input,
                                     maskShape_input,
                                     clineInvMask_input,
                                     ds_input,
                                     fm_input,
                                     selLineMask_input,
                                     vdotnScalar_input,
                                     fmInv_input)

    merged_val = tf.summary.merge(proto_list)

    return merged_val, val_loss, val_data_loss, val_smooth_loss, \
           [npr_lines, ds, fm, fm_inv, gt_field, cline_mask_inv, mask_shape, mask2d, selm, vdotn]


def train_net():
    # Set logging
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # Load network
    net = SKETCHNET()

    # Train
    train_data_records = [item for item in os.listdir(hyper_params['dbTrain']) if item.endswith('.tfrecords')]
    train_records = [os.path.join(hyper_params['dbTrain'], item) for item in train_data_records if
                     item.find('train') != -1]
    train_summary, train_step, train_loss, train_inputList = train_procedure(net, train_records)

    # Validation
    val_data_records = [item for item in os.listdir(hyper_params['dbEval']) if item.endswith('.tfrecords')]
    val_records = [os.path.join(hyper_params['dbEval'], item) for item in val_data_records if
                   item.find('eval') != -1]
    num_eval_samples = sum(1 for _ in tf.python_io.tf_record_iterator(val_records[0]))
    num_eval_itr = num_eval_samples // hyper_params['batchSize']
    num_eval_itr += 1

    val_proto, val_total_loss, val_data_loss, val_smooth_loss, val_inputList = validation_procedure(net, val_records)

    valid_loss = tf.placeholder(tf.float32, name='val_loss')
    valid_loss_proto = tf.summary.scalar('Validating_TotalLoss', valid_loss)
    valid_data_loss = tf.placeholder(tf.float32, name='val_data_loss')
    valid_data_loss_proto = tf.summary.scalar('Validating_DataL1Loss', valid_data_loss)
    valid_smooth_loss = tf.placeholder(tf.float32, name='val_smooth_loss')
    valid_smooth_loss_proto = tf.summary.scalar('Validating_SmoothL1Loss', valid_smooth_loss)
    valid_loss_merge = tf.summary.merge([valid_loss_proto, valid_data_loss_proto, valid_smooth_loss_proto])

    # Saver
    tf_saver = tf.train.Saver(max_to_keep=100)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

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

        for titr in range(hyper_params['maxIter']):
            # Validation
            if titr % hyper_params['exeValStep'] == 0:
                idx = randint(0, num_eval_itr - 1)
                avg_loss = 0.0
                avg_data_loss = 0.0
                avg_smooth_loss = 0.0
                for eitr in range(num_eval_itr):

                    # get real input
                    val_real_input = sess.run(val_inputList)

                    if eitr == idx:
                        val_merge, cur_v_loss, cur_v_data_loss, cur_v_smooth_loss = sess.run(
                            [val_proto, val_total_loss, val_data_loss, val_smooth_loss],
                            feed_dict={'npr_input:0': val_real_input[0],
                                       'ds_input:0': val_real_input[1],
                                       'fLMask_input:0': val_real_input[2],
                                       'fLInvMask_input:0': val_real_input[3],
                                       'gtField_input:0': val_real_input[4],
                                       'clIMask_input:0': val_real_input[5],
                                       'shapeMask_input:0': val_real_input[6],
                                       '2dMask_input:0': val_real_input[7],
                                       'sLMask_input:0': val_real_input[8],
                                       'curvMag_input:0': val_real_input[9]
                                       })
                        train_writer.add_summary(val_merge, titr)
                    else:
                        cur_v_loss, cur_v_data_loss, cur_v_smooth_loss = sess.run(
                            [val_total_loss, val_data_loss, val_smooth_loss],
                            feed_dict={'npr_input:0': val_real_input[0],
                                       'ds_input:0': val_real_input[1],
                                       'fLMask_input:0': val_real_input[2],
                                       'fLInvMask_input:0': val_real_input[3],
                                       'gtField_input:0': val_real_input[4],
                                       'clIMask_input:0': val_real_input[5],
                                       'shapeMask_input:0': val_real_input[6],
                                       '2dMask_input:0': val_real_input[7],
                                       'sLMask_input:0': val_real_input[8],
                                       'curvMag_input:0': val_real_input[9]
                                       })
                    avg_loss += cur_v_loss
                    avg_data_loss += cur_v_data_loss
                    avg_smooth_loss += cur_v_smooth_loss

                avg_loss /= num_eval_itr
                avg_data_loss /= num_eval_itr
                avg_smooth_loss /= num_eval_itr
                valid_summary = sess.run(valid_loss_merge,
                                         feed_dict={'val_loss:0': avg_loss, 'val_data_loss:0': avg_data_loss,
                                                    'val_smooth_loss:0': avg_smooth_loss})
                train_writer.add_summary(valid_summary, titr)
                train_logger.info('Validation loss at step {} is: {}'.format(titr, avg_loss))

            # Save model
            if titr % hyper_params['saveModelStep'] == 0:
                tf_saver.save(sess, hyper_params['outDir'] + '/savedModel/my_model{:d}.ckpt'.format(titr))
                train_logger.info('Save model at step: {:d}'.format(titr))

            # Training
            # get real input
            train_real_input = sess.run(train_inputList)

            t_summary, _, t_loss = sess.run([train_summary, train_step, train_loss],
                                            feed_dict={'npr_input:0': train_real_input[0],
                                                       'ds_input:0': train_real_input[1],
                                                       'fLMask_input:0': train_real_input[2],
                                                       'fLInvMask_input:0': train_real_input[3],
                                                       'gtField_input:0': train_real_input[4],
                                                       'clIMask_input:0': train_real_input[5],
                                                       'shapeMask_input:0': train_real_input[6],
                                                       '2dMask_input:0': train_real_input[7],
                                                       'sLMask_input:0': train_real_input[8],
                                                       'curvMag_input:0': train_real_input[9]
                                                       })

            # display
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

    args = parser.parse_args()
    hyper_params['dbTrain'] = args.dbTrain
    hyper_params['dbEval'] = args.dbEval
    hyper_params['outDir'] = args.outDir
    hyper_params['nb_gpus'] = args.nb_gpus
    hyper_params['device'] = args.devices

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

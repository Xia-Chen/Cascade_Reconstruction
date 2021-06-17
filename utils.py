import os
import glob
import pathlib
import copy
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from funcsigs import signature


from scipy.misc import imread
from scipy.io import loadmat, savemat
from skimage.measure import compare_ssim as ssim



def resize(datum, size):
    if len(datum.shape) == 2:
        return cv.resize(datum.astype(float), tuple(size))
    elif len(datum.shape) == 3:
        return np.dstack([cv.resize(datum[:, :, i].astype(float), tuple(size)) for i in range(datum.shape[-1])])
    else:
        print('unexpected data size', datum.shape)
        return None


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def extend_gray_channel(datum):
    if len(datum.shape) < 3 or datum.shape[2] == 1:
        return np.dstack((datum, datum, datum))
    return datum


def load_images_and_flows(dataset, new_size=[128, 192], lamda=0, train=True, force_recalc=False, channel=3):
    img_dir = dataset['path_train' if train else 'path_test']
    flow_dir = dataset['path_flow_train' if train else 'path_flow_test']
    flow_ang_dir = dataset['path_flow_ang_train' if train else 'path_flow_ang_test']
    # check: e.g. 15312 for Avenue
    n_images = np.sum(count_sequence_n_frame(dataset, test=not train) - 1)
    print('number of images: ', n_images)
    resized_image_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    resized_flow_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    resized_flow_ang_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']  # µÃµ½Êý¾Ý¼¯×ÓÊÓÆµÊýÁ¿
    #
    idx = 0

    for i in range(n_clip):
        saved_image_file = '%s/%s_image_clip_%d.npz' % (img_dir, 'training' if train else 'test', i + 1)
        if os.path.isfile(saved_image_file) and not force_recalc:
            image_data = np.load(saved_image_file)['image']
        else:
            clip_path = '%s/%s%s/' % (img_dir, 'Train' if train else 'Test', str(i + 1).zfill(3))
            print(clip_path)
            if dataset['name'] == 'Avenue':
                img_files = sorted(glob.glob(clip_path + '*.jpg'))[:-1]
            else:
                img_files = sorted(glob.glob(clip_path + '*.tif'))[:-1]
            if channel == 3:
                image_data = np.array([extend_gray_channel(resize(imread(img_file) / 255., (new_size[1], new_size[0])))
                                       for img_file in img_files]).astype(np.float32)
            else:
                image_data = np.array(
                    [np.expand_dims(resize(imread(img_file) / 255., (new_size[1], new_size[0])), axis=-1)
                     for img_file in img_files]).astype(np.float32)

            np.savez_compressed(saved_image_file, image=image_data)  # ½«¶ÁÈ¡×ÓÊÓÆµµÄÊý¾Ý±£´æÎª¶þ½øÖÆÎÄ¼þ
        resized_image_data[idx:idx + len(image_data)] = image_data

        saved_flow_file = '%s/%s_flow_clip_%d.npz' % (flow_dir, 'training' if train else 'test', i + 1)
        if os.path.isfile(saved_flow_file) and not force_recalc:
            flow_data = np.load(saved_flow_file)['flow']
        else:
            flow_clip_path = '%s/%s%s/' % (flow_dir, 'Train' if train else 'Test', str(i + 1))

            if dataset['name'] == 'Avenue':
                n_flow_files = len(glob.glob(flow_clip_path + '*.jpg'))
                flow_files = []
                for n in range(n_flow_files):
                    flow_files.append('%s%d.jpg' % (flow_clip_path, n + 1))
            else:
                n_flow_files = len(glob.glob(flow_clip_path + '*.tif'))
                flow_files = []
                for n in range(n_flow_files):
                    flow_files.append('%s%d.tif' % (flow_clip_path, n + 1))

            if channel == 3:
                flow_data = np.array([extend_gray_channel(resize(imread(flow_file) / 255., (new_size[1], new_size[0])))
                                      for flow_file in flow_files]).astype(np.float32)
            else:
                flow_data = np.array(
                    [np.expand_dims(resize(imread(flow_file) / 255., (new_size[1], new_size[0])), axis=-1)
                     for flow_file in flow_files]).astype(np.float32)

            np.savez_compressed(saved_flow_file, flow=flow_data)
        resized_flow_data[idx:idx + len(flow_data)] = flow_data

        saved_flow_ang_file = '%s/%s_flow_clip_%d.npz' % (flow_ang_dir, 'training' if train else 'test', i + 1)
        if os.path.isfile(saved_flow_ang_file) and not force_recalc:
            flow_ang_data = np.load(saved_flow_ang_file)['flow_ang']
        else:
            flow_ang_clip_path = '%s/%s%s/' % (flow_ang_dir, 'Train' if train else 'Test', str(i + 1))

            if dataset['name'] == 'Avenue':
                n_flow_files = len(glob.glob(flow_ang_clip_path + '*.jpg'))
                flow_files = []
                for n in range(n_flow_files):
                    flow_files.append('%s%d.jpg' % (flow_ang_clip_path, n + 1))
            else:
                n_flow_files = len(glob.glob(flow_ang_clip_path + '*.tif'))
                flow_files = []
                for n in range(n_flow_files):
                    flow_files.append('%s%d.tif' % (flow_ang_clip_path, n + 1))

            if channel == 3:
                flow_ang_data = np.array(
                    [extend_gray_channel(resize(imread(flow_file) / 255., (new_size[1], new_size[0])))
                     for flow_file in flow_files]).astype(np.float32)
            else:
                flow_ang_data = np.array(
                    [np.expand_dims(resize(imread(flow_file) / 255., (new_size[1], new_size[0])), axis=-1)
                     for flow_file in flow_files]).astype(np.float32)

            np.savez_compressed(saved_flow_ang_file, flow_ang=flow_ang_data)
        resized_flow_ang_data[idx:idx + len(flow_ang_data)] = flow_ang_data

        assert len(image_data) == len(flow_data) and len(image_data) == len(flow_ang_data)
        idx += len(image_data)

    return resized_image_data, resized_flow_data, resized_flow_ang_data


def load_images_and_flows_umn(dataset, new_size=[128, 192], train=True, force_recalc=False, channel=1):
    img_dir = dataset['path_train' if train else 'path_test']
    flow_dir = dataset['path_flow_train' if train else 'path_flow_test']
    flow_ang_dir = dataset['path_flow_ang_train' if train else 'path_flow_ang_test']
    n_images = dataset['n_images_train' if train else 'n_images_test'] - 1
    print('number of images: ', n_images)
    resized_image_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    resized_flow_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    resized_flow_ang_data = np.empty((n_images, new_size[0], new_size[1], channel), dtype=np.float32)
    n_img_files = len(glob.glob(img_dir + '/*.jpg')) - 1
    assert n_images == n_img_files

    saved_image_file = '%s/%s_image_clip_%d.npz' % (img_dir, 'training' if train else 'test', 1)
    if os.path.isfile(saved_image_file) and not force_recalc:
        image_data = np.load(saved_image_file)['image']
    else:
        img_files = []
        for n in range(n_img_files):
            img_files.append('%s/%d.jpg' % (img_dir, n + 1))

        if channel == 3:
            image_data = np.array([extend_gray_channel(resize(imread(img_file) / 255., (new_size[1], new_size[0])))
                                   for img_file in img_files]).astype(np.float32)
        else:
            image_data = np.array([np.expand_dims(resize(imread(img_file) / 255., (new_size[1], new_size[0])), axis=-1)
                                   for img_file in img_files]).astype(np.float32)
        np.savez_compressed(saved_image_file, image=image_data)
    resized_image_data = image_data

    saved_flow_file = '%s/%s_flow_clip_%d.npz' % (flow_dir, 'training' if train else 'test', 1)
    if os.path.isfile(saved_flow_file) and not force_recalc:
        flow_data = np.load(saved_flow_file)['flow']
    else:
        n_flow_files = len(glob.glob(flow_dir + '/*.jpg'))
        flow_files = []
        for n in range(n_flow_files):
            flow_files.append('%s/%d.jpg' % (flow_dir, n + 1))

        if channel == 3:
            flow_data = np.array([extend_gray_channel(resize(imread(flow_file) / 255., (new_size[1], new_size[0])))
                                  for flow_file in flow_files]).astype(np.float32)
        else:
            flow_data = np.array([np.expand_dims(resize(imread(flow_file) / 255., (new_size[1], new_size[0])), axis=-1)
                                  for flow_file in flow_files]).astype(np.float32)
        np.savez_compressed(saved_flow_file, flow=flow_data)
    resized_flow_data = flow_data

    saved_flow_file = '%s/%s_flow_clip_%d.npz' % (flow_ang_dir, 'training' if train else 'test', 1)
    if os.path.isfile(saved_flow_file) and not force_recalc:
        flow_ang_data = np.load(saved_flow_file)['flow_ang']
    else:
        n_flow_files = len(glob.glob(flow_ang_dir + '/*.jpg'))
        flow_files = []
        for n in range(n_flow_files):
            flow_files.append('%s/%d.jpg' % (flow_ang_dir, n + 1))
        if channel == 3:
            flow_ang_data = np.array([extend_gray_channel(resize(imread(flow_file) / 255., (new_size[1], new_size[0])))
                                  for flow_file in flow_files]).astype(np.float32)
        else:
            flow_ang_data = np.array([np.expand_dims(resize(imread(flow_file) / 255., (new_size[1], new_size[0])), axis=-1)
                                  for flow_file in flow_files]).astype(np.float32)
        np.savez_compressed(saved_flow_file, flow_ang=flow_ang_data)
    resized_flow_ang_data = flow_ang_data
    assert len(image_data) == len(flow_data) and len(flow_data) == len(flow_ang_data)
    print(image_data.shape, flow_data.shape)

    return resized_image_data, resized_flow_data, resized_flow_ang_data


def load_images_and_flow_1clip(dataset, clip_idx, train=False):
    img_dir = dataset['path_train' if train else 'path_test']
    flow_dir = dataset['path_flow_train' if train else 'path_flow_test']
    flow_ang_dir = dataset['path_flow_ang_train' if train else 'path_flow_ang_test']
    saved_image_file = '%s/%s_image_clip_%d.npz' % (img_dir, 'training' if train else 'test', clip_idx + 1)
    saved_flow_file = '%s/%s_flow_clip_%d.npz' % (flow_dir, 'training' if train else 'test', clip_idx + 1)
    saved_flow_ang_file = '%s/%s_flow_clip_%d.npz' % (flow_ang_dir, 'training' if train else 'test', clip_idx + 1)
    print(saved_image_file)
    assert os.path.isfile(saved_image_file)
    assert os.path.isfile(saved_flow_file)
    image_data = np.load(saved_image_file)['image']
    flow_data = np.load(saved_flow_file)['flow']
    flow_ang_data = np.load(saved_flow_ang_file)['flow_ang']
    return image_data, flow_data, flow_ang_data


def count_sequence_n_frame(dataset, test=True):
    sequence_n_frame = np.zeros(dataset['n_clip_test' if test else 'n_clip_train'], dtype=int)
    for i in range(len(sequence_n_frame)):
        clip_path = '%s/%s%s/' % (
        dataset['path_test' if test else 'path_train'], 'Test' if test else 'Train', str(i + 1).zfill(3))
        if dataset['name'] =='Avenue':
            sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.jpg')))
        elif dataset['name'] in ['UMN_scene1', 'UMN_scene2', 'UMN_scene3']:
            clip_path = '%s/' % (dataset['path_test' if test else 'path_train'])
            sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.jpg')))
        else:
            sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.tif')))

    return sequence_n_frame


def count_sequence_n_flow_frame(dataset, test=True):
    sequence_n_frame = np.zeros(dataset['n_clip_test' if test else 'n_clip_train'], dtype=int)
    for i in range(len(sequence_n_frame)):
        clip_path = '%s/%s%s/' % (
        dataset['path_flow_test' if test else 'path_flow_train'], 'Test' if test else 'Train', str(i + 1))
        sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.jpg')))
    return sequence_n_frame


# 1: abnormal, 0: normal
def get_test_frame_labels(ground_truth, sequence_n_frame):
    assert len(ground_truth) == len(sequence_n_frame)
    labels_exclude_last = np.zeros(0, dtype=int)
    labels_exclude_first = np.zeros(0, dtype=int)
    labels_full = np.zeros(0, dtype=int)
    for i in range(len(sequence_n_frame)):
        seg = ground_truth[i]
        tmp_labels = np.zeros(sequence_n_frame[i])
        for j in range(0, len(seg), 2):
            tmp_labels[(seg[j] - 1):seg[j + 1]] = 1
        labels_exclude_last = np.append(labels_exclude_last, tmp_labels[:-1])
        labels_exclude_first = np.append(labels_exclude_first, tmp_labels[1:])
        labels_full = np.append(labels_full, tmp_labels)
    return labels_exclude_last, labels_exclude_first, labels_full


def calc_anomaly_score_one_frame(frame_true, frame_hat, flow_true, flow_hat, angle_true, angle_hat,
                                 thresh_cut_off=0, return_as_map=False, operation=np.mean):
    assert frame_true.shape == frame_hat.shape
    assert flow_true.shape == flow_hat.shape
    loss_appe = (frame_true - frame_hat) ** 2
    loss_flow = (flow_true - flow_hat) ** 2
    loss_angle_flow = (angle_true - angle_hat) ** 2

    # cut-off low scores to check only high scores
    if thresh_cut_off is not None:
        thresh_cut_off = np.mean(loss_flow) * thresh_cut_off
        bi_matrix_flow = loss_flow > thresh_cut_off
        loss_appe = loss_appe * bi_matrix_flow

    # return score map for pixel-wise assessment
    if return_as_map:
        return operation(loss_appe, axis=-1), operation(loss_flow, axis=-1), loss_angle_flow

    def calc_measures_single_item(item_true, item_hat, squared_error, max_val_hat):
        # PSNR
        PSNR_X = 10 * np.log10(np.max(item_hat) ** 2 / np.mean(squared_error))
        PSNR_inv = np.max(item_hat) ** 2 * np.mean(squared_error)
        PSNR = 10 * np.log10(max_val_hat ** 2 / np.mean(squared_error))
        # SSIM
        SSIM = ssim(item_true, item_hat, data_range=np.max([item_true, item_hat]) - np.min([item_true, item_hat]),
                    multichannel=len(item_true.shape) == 3 and item_true.shape[-1] > 1)
        # MSE
        stat_MSE = np.mean(squared_error)
        stat_maxSE = np.max(squared_error)
        stat_std = np.std(squared_error)
        stat_MSE_1channel = np.mean(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_maxSE_1channel = np.max(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_std_1channel = np.std(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        return np.array(
            [PSNR_X, PSNR_inv, PSNR, SSIM, stat_MSE, stat_maxSE, stat_std, stat_MSE_1channel, stat_maxSE_1channel,
             stat_std_1channel])

    scores_appe = calc_measures_single_item(frame_true, frame_hat, loss_appe, 1.0)
    scores_flow = calc_measures_single_item(flow_true, flow_hat, loss_flow, 1.0)
    scores_angle = calc_measures_single_item(angle_true, angle_hat, loss_angle_flow, 1.0)

    return np.array([scores_appe, scores_flow, scores_angle])


def calc_anomaly_score(frames_true, frames_hat, flows_true, flows_hat, flows_ang_true, flows_ang_hat):
    assert frames_true.shape == frames_hat.shape
    assert flows_true.shape == flows_hat.shape
    return np.array([calc_anomaly_score_one_frame(frames_true[i], frames_hat[i], flows_true[i], flows_hat[i],
                                                  flows_ang_true[i], flows_ang_hat[i])
                     for i in range(len(frames_true))])


# suitable for Avenue
def calc_score_one_clip(dataset, epoch, clip_idx, train=False, force_calc=False):
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s/score_epoch_%d_clip_%d.npz' % (saved_data_path, epoch, clip_idx + 1)
    if not force_calc and os.path.isfile(saved_score_file):
        return np.load(saved_score_file)['data']
    # load true data and outputted data
    in_appe, in_flow, in_flow_ang = load_images_and_flow_1clip(dataset, clip_idx, train=train)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    out_loader = np.load(saved_data_file)
    out_appe, out_flow, out_flow_ang = out_loader['image'].astype(np.float32), out_loader['flow'].astype(np.float32), \
                                       out_loader['flow_ang'].astype(np.float32)

    print(in_appe.shape, out_appe.shape, in_flow.shape, out_flow.shape)
    assert in_appe.shape == out_appe.shape
    assert in_flow.shape == out_flow.shape
    assert in_flow_ang.shape == out_flow_ang.shape
    # calc score and save to file
    score_frame = calc_anomaly_score(in_appe, out_appe, in_flow, out_flow, in_flow_ang, out_flow_ang)
    np.savez_compressed(saved_score_file, data=score_frame)
    return score_frame


def calc_score_full_clips(dataset, epoch, train=False, force_calc=True):
    def flip_scores(scores):
        norm_scores = np.zeros_like(scores)
        for i in range(len(norm_scores)):
            norm_scores[i] = scores[i]
            norm_scores[i, :, 0] = 1. / norm_scores[i, :, 0]  # PSNR_X
            norm_scores[i, :, 2] = 1. / norm_scores[i, :, 2]  # PSNR
            norm_scores[i, :, 3] = 1. / norm_scores[i, :, 3]  # SSIM
        return norm_scores

    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s_scores.npz' % saved_data_path
    if not force_calc and os.path.isfile(saved_score_file):
        return flip_scores(np.load(saved_score_file)['data'])
    # calc scores for all clips and save to file
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']
    for i in range(n_clip):
        if i == 0:
            score_frame = calc_score_one_clip(dataset, epoch, i, train=train, force_calc=False)
        else:
            score_frame = np.concatenate((score_frame,
                                          calc_score_one_clip(dataset, epoch, i, train=train, force_calc=False)),
                                         axis=0)
    np.savez_compressed(saved_score_file, data=score_frame)
    return flip_scores(score_frame)


def get_weights(dataset, epoch, bool_future_anomaly=False,force_calc=True):
    if dataset is None:
        return None
    saved_data_path = './training_saver/%s/output_train/%d_epoch' % (dataset['name'], epoch)
    saved_score_file = '%s_scores.npz' % saved_data_path
    if os.path.isfile(saved_score_file) and not force_calc:
        training_scores = np.load(saved_score_file)['data']
    else:
        training_scores = calc_score_full_clips(dataset, epoch, train=True)

    if bool_future_anomaly:
        training_scores = multi_future_frames_to_scores(training_scores)

    return np.mean(training_scores, axis=0)


def basic_assess_AUC(scores, labels, plot_pr_idx=None):
    assert len(scores) == len(labels)
    if plot_pr_idx is not None:
        precision, recall, _ = precision_recall_curve(labels, scores[:, plot_pr_idx])
        print(
        len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]), len(np.unique(precision)), len(np.unique(recall)))
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()
    return np.array([roc_auc_score(labels, scores[:, i]) for i in range(scores.shape[1])]), \
           np.array([average_precision_score(labels, scores[:, i]) for i in range(scores.shape[1])])


def normalize_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return [item / np.max(item, axis=0) for item in scores]     # ÁÐ²éÕÒ×î´ó×îÐ¡ÖµßMÐÐnormalize
    else:
        return [(item - np.min(item, axis=0)) / (np.max(item, axis=0) - np.min(item, axis=0)) for item in scores]


def normalize_one_clip_scores(scores, ver=1):
    assert ver in [1, 2]
    if ver == 1:
        return scores / np.max(scores, axis=0)
    else:
        return (scores - np.min(scores, axis=0)) / (np.max(scores, axis=0) - np.min(scores, axis=0))


def normalize(sequence_n_frame, scores_appe, scores_flow, scores_comb, scores_angle, ver=2, clip_normalize=True):
    if sequence_n_frame is not None:
        if len(sequence_n_frame) > 1:
            accumulated_n_frame = np.cumsum(sequence_n_frame - 1)[:-1]

            scores_appe = np.split(scores_appe, accumulated_n_frame, axis=0)
            scores_flow = np.split(scores_flow, accumulated_n_frame, axis=0)
            scores_comb = np.split(scores_comb, accumulated_n_frame, axis=0)
            scores_angle = np.split(scores_angle, accumulated_n_frame, axis=0)

            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')
                scores_appe = normalize_clip_scores(scores_appe, ver=ver)
                scores_flow = normalize_clip_scores(scores_flow, ver=ver)
                scores_comb = normalize_clip_scores(scores_comb, ver=ver)
                scores_angle = normalize_clip_scores(scores_angle, ver=ver)

            scores_appe = np.concatenate(scores_appe, axis=0)
            scores_flow = np.concatenate(scores_flow, axis=0)
            scores_comb = np.concatenate(scores_comb, axis=0)
            scores_angle = np.concatenate(scores_angle, axis=0)

        else:
            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')

                scores_appe = np.array(normalize_one_clip_scores(scores_appe, ver=ver))
                scores_flow = np.array(normalize_one_clip_scores(scores_flow, ver=ver))
                scores_comb = np.array(normalize_one_clip_scores(scores_comb, ver=ver))
                scores_angle = np.array(normalize_one_clip_scores(scores_angle, ver=1))

    return scores_appe, scores_flow, scores_angle, scores_comb


def full_assess_AUC(dataset, score_frame, frame_labels, w_img=0.5, w_flow=0.5, w_flow_ang=0.5, epoch=0,
                    sequence_n_frame=None, clip_normalize=True, use_pr=False,
                    para_appe=1, para_flow=5, para_angle=0.1):

    scores_appe = score_frame[:, 0, :] / w_img
    scores_flow = score_frame[:, 1, :] / w_flow
    scores_angle = score_frame[:, 2, :] / w_flow_ang

    scores_comb = para_appe * scores_appe + para_flow * scores_flow + para_angle * scores_angle

    # print(scores_appe.shape, scores_flow.shape, scores_comb.shape, scores_angle.shape)
    auc, prc = basic_assess_AUC(scores_appe, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]

    return scores_appe, scores_flow, scores_angle, scores_comb


def find_max_patch(diff_map_flow, diff_map_angle, diff_map_appe, size=16, step=4, plot=False):
    # assert len(diff_map_flow.shape) == 2 and diff_map_flow.shape[0] % size == 0 and diff_map_flow.shape[1] % size == 0
    assert size % step == 0
    max_val_mean, std_1, pos_1, mean_angle_1, std_angle_1, mean_appe_1, std_appe_1 = 0, 0, None, 0, 0, 0, 0
    mean_2, max_val_std, pos_2, mean_angle_2, std_angle_2, mean_appe_2, std_appe_2 = 0, 0, None, 0, 0, 0, 0

    for i in range(0, diff_map_flow.shape[0] - size, step):
        for j in range(0, diff_map_flow.shape[1] - size, step):
            curr_std = np.std(diff_map_flow[i:i + size, j:j + size, :])
            curr_mean = np.mean(diff_map_flow[i:i + size, j:j + size, :])
            curr_std_angle = np.std(diff_map_angle[i:i + size, j:j + size, :])
            curr_mean_angle = np.mean(diff_map_angle[i:i + size, j:j + size, :])
            curr_std_appe = np.std(diff_map_appe[i:i + size, j:j + size, :])
            curr_mean_appe = np.mean(diff_map_appe[i:i + size, j:j + size, :])
            if curr_mean > max_val_mean:
                max_val_mean = curr_mean
                std_1 = curr_std
                pos_1 = [i, j]
                std_angle_1 = curr_std_angle
                mean_angle_1 = curr_mean_angle
                std_appe_1 = curr_std_appe
                mean_appe_1 = curr_mean_appe
            if curr_std > max_val_std:
                max_val_std = curr_std
                mean_2 = curr_mean
                pos_2 = [i, j]
                std_angle_2 = curr_std_angle
                mean_angle_2 = curr_mean_angle
                std_appe_2 = curr_std_appe
                mean_appe_2 = curr_mean_appe

    if plot:
        print(pos_1, max_val_mean, std_1, std_appe_1, mean_appe_1)
        print(pos_2, mean_2, max_val_std, std_appe_2, mean_appe_2)
        rect_mean = Rectangle((pos_1[1], pos_1[0]), size, size, linewidth=2, edgecolor='g', facecolor='none')
        rect_std = Rectangle((pos_2[1], pos_2[0]), size, size, linewidth=2, edgecolor='r', facecolor='none')
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(diff_map_flow)
        ax1.add_patch(rect_mean)
        ax1.add_patch(rect_std)
        rect_mean = Rectangle((pos_1[1], pos_1[0]), size, size, linewidth=2, edgecolor='g', facecolor='none')
        rect_std = Rectangle((pos_2[1], pos_2[0]), size, size, linewidth=2, edgecolor='r', facecolor='none')
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(diff_map_appe)
        ax2.add_patch(rect_mean)
        ax2.add_patch(rect_std)
        plt.show()
    return max_val_mean, std_1, mean_angle_1, std_angle_1, mean_appe_1, std_appe_1,\
           mean_2, max_val_std, mean_angle_2, std_angle_2, mean_appe_2, std_appe_2


def calc_score_max_patch_one_clip(dataset, epoch, clip_idx, step=4, train=False, force_calc=False,
                                  bool_clip_future_anomaly=False, thresh_cut_off=0):
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s/patch_score_epoch_%d_clip_%d_step_%d.npz' % (saved_data_path, epoch, clip_idx + 1, step)
    if not force_calc and os.path.isfile(saved_score_file):
        return np.load(saved_score_file)['data']
    # load true data and outputted data
    in_appe, in_flow, in_flow_ang = load_images_and_flow_1clip(dataset, clip_idx, train=train)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    out_loader = np.load(saved_data_file)
    out_appe, out_flow, out_flow_ang = out_loader['image'].astype(np.float32), out_loader['flow'].astype(np.float32), \
                                       out_loader['flow_ang'].astype(np.float32)

    assert in_flow.shape == out_flow.shape
    # calc score and save to file
    diff_map_appe = (in_appe - out_appe) ** 2
    diff_map_flow = (in_flow - out_flow) ** 2
    diff_map_angle = (in_flow_ang - out_flow_ang) ** 2

    score_seq = np.array([find_max_patch(diff_map_flow[i], diff_map_angle[i],
                                         diff_map_appe[i], step=step)
                          for i in range(len(in_flow))])

    if bool_clip_future_anomaly:
        score_seq = multi_future_frames_to_scores(score_seq)
    np.savez_compressed(saved_score_file, data=score_seq)
    return score_seq


def normalize_full_scores(sequence_n_frame, full_scores, ver=2, clip_normalize=True):
    norm_full_scores = copy.copy(full_scores)
    if sequence_n_frame is not None and len(sequence_n_frame) > 1:
        accumulated_n_frame = np.cumsum(sequence_n_frame - 1)[:-1]
        for i in range(full_scores.shape[1]):
            scores = np.split(full_scores[:, i], accumulated_n_frame, axis=0)
            if clip_normalize:
                np.seterr(divide='ignore', invalid='ignore')
                scores = normalize_clip_scores(scores, ver=ver)

            norm_full_scores[:, i] = np.concatenate(scores, axis=0)
    else:
        if clip_normalize:
            np.seterr(divide='ignore', invalid='ignore')
            norm_full_scores = np.array(normalize_one_clip_scores(full_scores, ver=ver))

    return norm_full_scores


def full_assess_AUC_by_max_patch(dataset, epoch, frame_labels, step=4, sequence_n_frame=None, clip_normalize=True,
                                 bool_future_anomaly=False, force_calc_score=True,
                                 save_roc_pr_idx=12, para_appe=1, para_flow=5, para_angle=0.1):

    def load_patch_scores(dataset, epoch, step, train=False, force_calc_score=False, bool_clip_future_anomaly=False):
        saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (
        dataset['name'], 'train' if train else 'test', epoch)
        saved_score_file = '%s_max_patch_scores_step_%d.npz' % (saved_data_path, step)
        if not force_calc_score and os.path.isfile(saved_score_file):
            loaded_scores = np.load(saved_score_file)['data']
        else:
            loaded_scores = np.concatenate([calc_score_max_patch_one_clip(dataset, epoch, clip_idx, step=step, train=train,
                                            bool_clip_future_anomaly=bool_clip_future_anomaly)
                                            for clip_idx in range(dataset['n_clip_train' if train else 'n_clip_test'])],
                                           axis=0)
            np.savez_compressed(saved_score_file, data=loaded_scores)
        return loaded_scores

    training_scores = load_patch_scores(dataset, epoch, step, train=True, force_calc_score=force_calc_score)
    test_scores = load_patch_scores(dataset, epoch, step, train=False, force_calc_score=force_calc_score)

    # if consider future frames when compute regularity scores
    if bool_future_anomaly:
        training_scores = multi_future_frames_to_scores(training_scores)
        test_scores = multi_future_frames_to_scores(test_scores)

    print('training_scores:', training_scores.shape)
    print('test_scores:', test_scores.shape)
    mean_training_scores = np.mean(training_scores, axis=0)

    new_test_scores = np.array([test_score / mean_training_scores for test_score in test_scores])
    comb_test_scores = np.array([para_flow * new_test_scores[i, [0, 1, 6, 7]] + para_angle * new_test_scores[i, [2, 3, 8, 9]] + para_appe *
                                 new_test_scores[i, [4, 5, 10, 11]] for i in range(len(new_test_scores))])
    full_scores = np.concatenate([new_test_scores, comb_test_scores], axis=1)

    # normalize
    norm_full_scores = normalize_full_scores(sequence_n_frame, full_scores, ver=1, clip_normalize=clip_normalize)

    if save_roc_pr_idx is not None:
        # ROC
        fpr_1, tpr_1, _ = roc_curve(frame_labels, full_scores[:, save_roc_pr_idx])

        return full_scores, roc_auc_score(frame_labels, full_scores[:, save_roc_pr_idx])


###########################################
def get_segments(seq):
    def find_ends(seq):
        tmp = np.insert(seq, 0, -10)
        diff = tmp[1:] - tmp[:-1]
        peaks = np.where(diff != 1)[0]
        #
        ret = np.empty((len(peaks), 2), dtype=int)
        for i in range(len(ret)):
            ret[i] = [peaks[i], (peaks[i + 1] - 1) if i < len(ret) - 1 else (len(seq) - 1)]
        return ret

    #
    ends = find_ends(seq)
    return np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(
        -1) + 1  # +1 for 1-based index (same as UCSD data)


def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i + 1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret


# consider the consecutive future frames to compute the regularity scores
def multi_future_frames_to_scores(old_scores):
    n_frames = len(old_scores)
    new_scores = np.zeros(old_scores.shape)

    # ------------------------based on current, future and the past frames----------------------------------
    flag_preserve_head_tail = False

    if flag_preserve_head_tail:
        # preserve the first frames
        new_scores[0] = old_scores[0]
        new_scores[1] = old_scores[1]
        # preserve the last frames
        new_scores[n_frames - 2] = old_scores[n_frames - 2]
        new_scores[n_frames - 1] = old_scores[n_frames - 1]
    else:
        # process the first frame
        new_scores[0] = (1 / (3 ** 2)) * (old_scores[0] * 3 + old_scores[1] * 2 + old_scores[2] * 1)
        new_scores[1] = (1 / (4 ** 2)) * (old_scores[0] * 2 + old_scores[1] * 3 + old_scores[2] * 2 + old_scores[3] * 1)
        # process the last frames
        new_scores[n_frames - 2] = (1 / (4 ** 2)) * (
                old_scores[n_frames - 4] * 1 + old_scores[n_frames - 3] * 2 + old_scores[n_frames - 2] * 3 + old_scores[
            n_frames - 1] * 2)
        new_scores[n_frames - 1] = (1 / (3 ** 2)) * (
                old_scores[n_frames - 3] * 1 + old_scores[n_frames - 2] * 2 + old_scores[n_frames - 1] * 3)

    for i in range(2, n_frames - 2):
        new_scores[i] = 1 / (5 ** 2) * (
                old_scores[i - 2] * 1 + old_scores[i - 1] * 2 + old_scores[i] * 3 + old_scores[i + 1] * 2 +
                old_scores[i + 2] * 1)

    return new_scores


def generate_mask(h, w):
    mask = np.ones((h, w))
    patch_size = 16
    n_row = random.sample(range(h//patch_size), 2)
    n_col = random.sample(range(w//patch_size), 3)
    for r in range(2):
        for c in range(3):
            mask[n_row[r] * patch_size: (n_row[r] + 1) * patch_size,
            n_col[c] * patch_size: (n_col[c] + 1) * patch_size] = np.zeros((patch_size, patch_size))

    return mask







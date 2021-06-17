import sys
import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# dataset
UCSDped1 = {'name': 'UCSDped1',
            'path': '/home/zlab-4/Chenxia/dataset/UCSDped1',
            'n_clip_train': 34,
            'n_clip_test': 36,
            'ground_truth': [[60, 152], [50, 175], [91, 200], [31, 168], [5, 90, 140, 200], [1, 100, 110, 200],
                             [1, 175], [1, 94], [1, 48], [1, 140], [70, 165], [130, 200], [1, 156], [1, 200], [138, 200],
                             [123, 200], [1, 47], [54, 120], [64, 138], [45, 175], [31, 200], [16, 107], [8, 165],
                             [50, 171], [40, 135], [77, 144], [10, 122], [105, 200], [1, 15, 45, 113], [175, 200],
                             [1, 180], [1, 52, 65, 115], [5, 165], [1, 121], [86, 200], [15, 108]],
            'ground_truth_mask': np.arange(36)+1}
UCSDped2 = {'name': 'UCSDped2',
            'path': '/home/zlab-4/Chenxia/dataset/UCSDped2',
            'n_clip_train': 16,
            'n_clip_test': 12,
            'ground_truth': [[61, 180], [95, 180], [1, 146], [31, 180], [1, 129], [1, 159],
                             [46, 180], [1, 180], [1, 120], [1, 150], [1, 180], [88, 180]],
            'ground_truth_mask': np.arange(12)+1}

Avenue = {'name': 'Avenue',
          'path': '/home/zlab-4/Chenxia/dataset/Avenue',
          'test_mask_path': '/home/zlab-4/Chenxia/dataset/Avenue/testing_label_mask',
          'n_clip_train': 16,
          'n_clip_test': 21,
          'ground_truth': None,
          'ground_truth_mask': np.arange(21)+1}

UMN_scene1 = {'name': 'UMN_scene1',
              'path': '/home/zlab-4/Chenxia/dataset/scene1',
              'n_images_train': 400,
              'n_images_test': 1053,
              'n_clip_train': 1,
              'n_clip_test': 1,
              'ground_truth': [[332, 441, 580, 669]]}

UMN_scene2 = {'name': 'UMN_scene2',
              'path': '/home/zlab-4/Chenxia/dataset/scene2',
              'n_images_train': 350,
              'n_images_test': 3796,
              'n_clip_train': 1,
              'n_clip_test': 1,
              'ground_truth': [[4, 183, 803, 882, 1418, 1627, 2137, 2216, 3006, 3127, 3621, 3794]]}

UMN_scene3 = {'name': 'UMN_scene3',
              'path': '/home/zlab-4/Chenxia/dataset/scene3',
              'n_images_train': 400,
              'n_images_test': 1742,
              'n_clip_train': 1,
              'n_clip_test': 1,
              'ground_truth': [[200, 239, 888, 917, 1705, 1724]]}

dataset_dict = {'UCSDped1': UCSDped1, 'UCSDped2': UCSDped2, 'Avenue': Avenue, 'UMN_scene1': UMN_scene1, 'UMN_scene2': UMN_scene2, 'UMN_scene3': UMN_scene3}


h, w, d = 128, 192, 3        #  128, 192, 3

print(tf.test.is_gpu_available())


def main(argv):
    parser = argparse.ArgumentParser(description='task n umber')
    parser.add_argument('-d', '--dataset', help='the name of dataset', default='UCSDped2')
    parser.add_argument('-t', '--task', help='task to perform', default=2)
    parser.add_argument('-set', '--set', help='test set', default=0)
    parser.add_argument('-e', '--epoch', help='number of epoch', default=100)
    parser.add_argument('-m', '--model', help='start model idx', default=0)
    parser.add_argument('-s', '--step', help='step for max_patch assessment', default=4)
    parser.add_argument('-p_flow', '--para_flow', help='the paramter to force flow magitude', default=1)
    parser.add_argument('-p_angle', '--para_angle', help='the paramter to force flow angle', default=0)
    parser.add_argument('-p_appe', '--para_appe', help='the paramter to force apperance', default=0.1)
    parser.add_argument('-bs', '--batch_size', help='batch size for training', default=8)
    parser.add_argument('-b', '--bool', help='boolean value for if consider multiple future frames to detect anomaly', default=1)
    args = vars(parser.parse_args())

    dataset = dataset_dict[args['dataset']]
    dataset['path_train'] = '%s/Train' % dataset['path']
    dataset['path_test'] = '%s/Test' % dataset['path']

    args = vars(parser.parse_args())
    dataset['path_flow_train'] = '%s/flow_Train' % dataset['path']
    dataset['path_flow_test'] = '%s/flow_Test' % dataset['path']
    dataset['path_flow_ang_train'] = '%s/flow_Train_ang' % dataset['path']
    dataset['path_flow_ang_test'] = '%s/flow_Test_ang' % dataset['path']
    #
    test_set = bool(int(args['set']))
    n_epoch_destination = int(args['epoch'])
    model_idx_to_start = int(args['model'])
    step = int(args['step'])
    task = int(args['task'])
    model_test = model_idx_to_start
    batch_size = int(args['batch_size'])
    bool_future_anomaly = bool(int(args['bool']))
    print('Selected task = %d' % task)
    print('started time: %s' % datetime.datetime.now())

    '''======================'''
    ''' Task 1: Prepare data '''
    '''======================'''
    if task == 1:
        if dataset['name'] == 'UMN_scene1' or dataset['name'] == 'UMN_scene2' or dataset['name'] == 'UMN_scene3':
            load_images_and_flows_umn(dataset, new_size=[h, w], train=not test_set, channel=d)
        else:
            load_images_and_flows(dataset, new_size=[h, w], train=not test_set, channel=d)

    '''========================='''
    ''' Task 2: Train GAN model '''
    '''========================='''
    if task == 2:
        # load data *H*W*C
        if dataset['name'] == 'UMN_scene1' or dataset['name'] == 'UMN_scene2' or dataset['name'] == 'UMN_scene3':
            image_data, flow_data, flow_ang_data = load_images_and_flows_umn(dataset, new_size=[h, w], train=not test_set, channel=d)
        else:
            image_data, flow_data, flow_ang_data = load_images_and_flows(dataset, new_size=[h, w], train=not test_set, channel=d)

        assert image_data.shape[1] == h and flow_data.shape[1] == h
        assert image_data.shape[2] == w and flow_data.shape[2] == w
        assert image_data.shape[-1] == d and flow_data.shape[-1] == d

        # train model
        train_model(image_data, flow_data, flow_ang_data,  n_epoch_destination, dataset_name=dataset['name'],
                                         start_model_idx=model_idx_to_start, batch_size=batch_size, channel=d)

    '''========================'''
    ''' Task 3: Test GAN model'''
    '''========================'''
    if task == 3:
        print('test set: ', test_set)

        sequence_n_frame = count_sequence_n_frame(dataset, test=test_set) - 1  # minus 1 because of the removal of last frame
        print(sequence_n_frame)
        test_model(h, w, dataset, sequence_n_frame, batch_size=batch_size, model_idx=model_test,
                       using_test_data=test_set, channel=d)

    '''======================================'''
    ''' Task 4: Assess max patch flow scores in global and local combination '''
    '''======================================'''
    if task == 4:
        if dataset['name'] == 'UMN_scene1' or dataset['name'] == 'UMN_scene2' or dataset['name'] == 'UMN_scene3':
            sequence_n_frame = [dataset['n_images_test']]
        else:
            sequence_n_frame = count_sequence_n_frame(dataset, test=True)

        if dataset['name'] == 'Avenue':
                dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
        # compute the global scores in test dataset
        score_frame = calc_score_full_clips(dataset, model_test, train=not test_set)
        # whether consider the past and future frame to caculate score
        if bool_future_anomaly:
            score_frame = multi_future_frames_to_scores(score_frame)

        training_errors = get_weights(dataset, model_test, bool_future_anomaly=bool_future_anomaly)
        w_img, w_flow, w_angle = training_errors[0].flatten(), training_errors[1].flatten(), training_errors[2].flatten()
        # print('weights: w_img =', str(w_img), '- vw_flow =', str(w_flow), '- vw_flow_ang =', str(w_angle))
        #
        labels_exclude_last, labels_exclude_first, labels_full = get_test_frame_labels(dataset['ground_truth'], sequence_n_frame)
        print('compute the auc by global detection')
        s_appe, _, _, _ = full_assess_AUC(dataset, score_frame, labels_exclude_last, w_img, w_flow, w_angle,
                                          model_test, sequence_n_frame, True, para_appe=1, para_flow=5,
                                          para_angle=0.1)
        s_global_appe = s_appe[:, 4]      # only need the mean_appe
        print('compute the auc by local detection')
        full_scores, _ = full_assess_AUC_by_max_patch(dataset, model_test, labels_exclude_last, step=4,
                                                            sequence_n_frame=sequence_n_frame,
                                                            clip_normalize=True,
                                                            bool_future_anomaly=bool_future_anomaly,
                                                            para_appe=args['para_appe'], para_flow=args['para_flow'],
                                                            para_angle=args['para_angle'])
        s_local_flow = full_scores[:, [0, 2]]         # only need the max_mean_flow and mean_angle
        s_comb = np.array([args['para_appe'] * s_global_appe[i] + args['para_flow'] * s_local_flow[i, 0]
                           + args['para_angle'] * s_local_flow[i, 1] for i in range(len(s_global_appe))])

        new_s_test = np.concatenate([np.expand_dims(s_comb, axis=1), np.expand_dims(s_global_appe, axis=1),np.expand_dims(s_local_flow[:, 0], axis=1)], axis=1)
        print('AUC:', ['%.3f' % roc_auc_score(labels_exclude_last, new_s_test[:, i]) for i in range(new_s_test.shape[1])])



if __name__ == '__main__':
    main(sys.argv)


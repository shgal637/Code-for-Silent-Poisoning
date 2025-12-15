# -*- coding: utf-8 -*-
"""
Parse configuration file
"""

import argparse
import logging

import yaml

from datasets.base_dataset import get_num_classes
import datetime

def get_args(temp):
    """
    parse configuration yaml file

    :return: configuration
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str)
    # temp = parser.parse_args()
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(temp.config, 'r', encoding='utf-8')
    cfg = f.read()
    args = yaml.load(cfg, Loader=yaml.SafeLoader)
    f.close()
    args['num_classes'] = get_num_classes(args['dataset'])

    if 'train_label_non_iid' not in args.keys():
        args['train_label_non_iid'] = None
    if 'train_label_fix_backdoor' not in args.keys():
        args['train_label_fix_backdoor'] = -1

    args['time'] = False

    now = datetime.datetime.now()
    time = now.strftime("%m-%d-%H-%M-%S")
    args['file_time'] = time
    set_logging(args['log'], time)
    return args


def set_logging(log_file, time):
    """
    configure logging INFO messaged located in tests/result

    :param str log_file: path of log file
    """

    logging.basicConfig(
        level=logging.INFO,
        filename='../tests/result/{}-{}.txt'.format(log_file, time),
        filemode='w',
        format='[%(asctime)s| %(levelname)s| %(processName)s] %(message)s' # 日志格式
    )

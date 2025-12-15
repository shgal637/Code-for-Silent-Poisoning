import ast
import logging
import os
import sys

import torch

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common.parser import get_args
from datasets.base_dataset import get_dataloader
from vfl.vfl import get_vfl
from vfl.vfl_fixture import VFLFixture
import random
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def run_compare_experiments(train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, g_r_train_loader, args,
                            backdoor_indices=None, backdoor_target_indices=None,
                            labeled_loader=None, unlabeled_loader=None, s_r_train_dl=None, sr_ba_backdoor_target_indices=None,
                            labeled_indices_dict=None, backdoor_y_test_true=None):
    if args['backdoor'] == 'poison':
        # data poisoning attack
        logging.info('------- data poisoning attack -------')
        run_experiment(train_loader=backdoor_train_loader, test_loader=test_loader,
                       backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type='poison')
    elif args['backdoor'] == 'g_r':
        # gradient-replacement
        logging.info('------- gradient replacement attack -------')
        run_experiment(train_loader=g_r_train_loader, test_loader=test_loader,
                       backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type='g_r',
                       backdoor_indices=backdoor_indices,
                       backdoor_target_indices=backdoor_target_indices)
    elif args['backdoor'] == 'no':
        # normal training
        logging.info('------- no attack -------')
        run_experiment(train_loader=train_loader, test_loader=test_loader, backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type='no')
    elif args['backdoor'] in ['sr_ba', 'villain', 'splitNN']:
        logging.info('------- SR-BA attack -------')
        run_experiment(train_loader=s_r_train_dl, test_loader=test_loader,
                       backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type=args['backdoor'],
                       backdoor_indices=backdoor_indices,
                       backdoor_target_indices=sr_ba_backdoor_target_indices,
                       backdoor_train_loader=backdoor_train_loader,
                       labeled_indices_dict=labeled_indices_dict, backdoor_y_test_true=backdoor_y_test_true)
    elif args['backdoor'] == 'baseline':
        # baseline attack
        logging.info('------- baseline attack -------')
        # if args['save_model']:
        #     args['load_model'] = 1
        run_experiment(train_loader=train_loader, test_loader=test_loader,
                       backdoor_train_loader=backdoor_train_loader,
                       backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type='baseline',
                       backdoor_indices=backdoor_indices,
                       labeled_loader=labeled_loader,
                       unlabeled_loader=unlabeled_loader)
    elif args['backdoor'] == 'lr_ba':
        # LR-BA
        logging.info('------- LR-BA attack -------')
        # if args['save_model']:
        #     args['load_model'] = 1
        run_experiment(train_loader=train_loader, test_loader=test_loader,
                       backdoor_train_loader=backdoor_train_loader,
                       backdoor_test_loader=backdoor_test_loader,
                       args=args, attack_type='lr_ba',
                       backdoor_indices=backdoor_indices,
                       labeled_loader=labeled_loader,
                       unlabeled_loader=unlabeled_loader, backdoor_y_test_true=backdoor_y_test_true)



def run_experiment(train_loader, test_loader, backdoor_test_loader, args, attack_type,
                   backdoor_indices=None, backdoor_target_indices=None,
                   labeled_loader=None, unlabeled_loader=None,
                   backdoor_train_loader=None,labeled_indices_dict=None, backdoor_y_test_true=None):
    if attack_type == 'no':
        args['backdoor'] = 'no'
    elif attack_type == 'poison':
        args['backdoor'] = 'poison'
    elif attack_type == 'g_r':
        args['backdoor'] = 'g_r'
    elif attack_type == 'baseline':
        args['backdoor'] = 'baseline'
    elif attack_type == 'lr_ba':
        args['backdoor'] = 'lr_ba'
    elif attack_type in ['sr_ba', 'villain', 'splitNN', 'TECB']:
        args['backdoor'] = attack_type

    vfl = get_vfl(args=args,
                  backdoor_indices=backdoor_indices,
                  backdoor_target_indices=backdoor_target_indices,
                  train_loader=train_loader,
                  labeled_indices_dict=labeled_indices_dict, backdoor_y_test_true=backdoor_y_test_true)


    vfl_fixture = VFLFixture(vfl, args=args)
    vfl_fixture.fit(
        train_loader, test_loader,
        backdoor_test_loader=backdoor_test_loader,
        title=attack_type)

    if attack_type == 'lr_ba':
        vfl_fixture.lr_ba_attack(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader)
    elif attack_type == 'baseline':
        vfl_fixture.baseline_attack(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step-gamma', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--backdoor', type=str, default='no', choices=['no', 'lr_ba', 'sr_ba', 'villain', 'splitNN'])
    parser.add_argument('--mal-optim', type=ast.literal_eval, default=False)
    parser.add_argument('--top-model', type=ast.literal_eval, default=True)
    parser.add_argument('--trigger', type=str, default='pixel')
    # possible defenses on/off paras
    parser.add_argument('--gc', help='turn_on_gradient_compression', type=ast.literal_eval, default=False)
    parser.add_argument('--noisy-gradients', help='turn_on_lap_noise', type=ast.literal_eval, default=False)
    parser.add_argument('--max-norm', help='turn_on_max_norm', type=ast.literal_eval, default=False)
    parser.add_argument('--norm-clip', help='turn_on_nrom-clip', type=ast.literal_eval, default=False)
    parser.add_argument('--FP', help='turn_on_fine_pruning', type=float, default=0)
    # paras about possible defenses
    parser.add_argument('--gc-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.75)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=1e-3)
    parser.add_argument('--clip-threshold', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=0.5)
    parser.add_argument('--outlier-detection', type=float, default=0)
    parser.add_argument('--embedding-detection', type=ast.literal_eval, default=False)
    parser.add_argument('--use-random', type=ast.literal_eval, default=False)
    parser.add_argument('--random', type=int, default=1)
    parser.add_argument('--debug', type=ast.literal_eval, default=False)
    parser.add_argument('--generation', type=ast.literal_eval, default=False)
    parser.add_argument('--trigger-add', help='for feature attack', type=ast.literal_eval, default=False)
    parser.add_argument('--pattern-lr', type=float, default=0.001)
    parser.add_argument('--n-passive-party', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.0001, help='for villain, define trigger;')
    parser.add_argument('--model-type', type=str, default='FCN',choices=['FCN', 'Resnet-1', 'Resnet-2-0','Resnet-2-1'])
    parser.add_argument('--load-time', type=str, default='none')
    parser.add_argument('--aggregate', type=str, default='Concate',choices=['Concate', 'Add', 'Mean'])
    parser.add_argument('--poison-rate', type=float, default=0.01)
    parser.add_argument('--m-dimension', type=int, default=10)
    parser.add_argument('--random-test', type=ast.literal_eval, default=False)
    parser.add_argument('--test-norm', help='turn_on_random_norm_in_test_for_mask', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0)
    # for debug or test
    parser.add_argument('--idea', type=int, default=0)
    parser.add_argument('--feature-gamma', type=float, default=1)
    # ABL defense
    parser.add_argument('--ABL', type=ast.literal_eval, default=False)
    parser.add_argument('--isolation-ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--gradient-ascent-type', type=str, default='Flooding', help='type of gradient ascent')
    parser.add_argument('--flooding-gamma', type=float, default=0.5, help='value of gamma or flooding, depends on the type')
    parser.add_argument('--t-epochs', type=int, default=20, help='value of flooding')
    parser.add_argument('--local-feature-gamma', type=float, default=1, help='balance the AS and ASR')
    # Teco defense
    parser.add_argument('--Teco', type=ast.literal_eval, default=False)
    parser.add_argument('--defense-threshold', type=float, default=0, help='defense_threshold for teco')

    temp = parser.parse_args()

    if not temp.use_random:
        args = get_args(temp)
        args['cuda'] = 1 if torch.cuda.is_available() else 0
        args['target_epochs'] = temp.epochs
        args['backdoor'] = temp.backdoor
        args['passive_bottom_lr'] = temp.lr
        args['active_bottom_lr'] = temp.lr
        args['active_top_lr'] = temp.lr
        args['target_batch_size'] = temp.batch_size
        args['passive_bottom_gamma'] = temp.step_gamma
        args['active_bottom_gamma'] = temp.step_gamma
        args['active_top_gamma'] = temp.step_gamma
        args['s_r_amplify_ratio'] = temp.gamma
        args['FP'] = temp.FP
        args['gradient_compression'] = temp.gc
        args['gc_percent'] = temp.gc_percent
        args['noisy_gradients'] = temp.noisy_gradients
        args['noise_scale'] = temp.noise_scale
        args['mal_optim'] = temp.mal_optim
        args['active_top_trainable'] = temp.top_model
        args['trigger'] = temp.trigger
        args['debug'] = temp.debug
        args['generation'] = temp.generation
        args['trigger_add'] = temp.trigger_add
        args['pattern_lr'] = temp.pattern_lr
        args['max_norm'] = temp.max_norm
        args['n_passive_party'] = temp.n_passive_party
        args['epsilon'] = temp.epsilon
        args['norm_clip'] = temp.norm_clip
        args['clip_threshold'] = temp.clip_threshold
        args['model_type'] = temp.model_type
        args['aggregate'] = temp.aggregate
        args['poison_rate'] = temp.poison_rate
        args['m_dimension'] = temp.m_dimension
        args['outlier_detection'] = temp.outlier_detection
        args['embedding_detection'] = temp.embedding_detection
        args['random_test'] = temp.random_test
        args['test_norm'] = temp.test_norm
        args['dropout'] = temp.dropout
        args['sr_feature_amplify_ratio'] = temp.feature_gamma
        # debug
        args['idea'] = temp.idea
        # ABL
        args['ABL'] = temp.ABL
        args['isolation_ratio'] = temp.isolation_ratio
        args['gradient_ascent_type'] = temp.gradient_ascent_type
        args['flooding_gamma'] = temp.flooding_gamma
        args['t_epochs'] = temp.t_epochs
        args['local_feature_gamma'] = temp.local_feature_gamma
        # Teco
        args['Teco'] = temp.Teco
        args['defense_threshold'] = temp.defense_threshold

        length_dict = {'cifar10':50000, 'cifar100':50000, 'cinic':180000}
        epochs_dict = {'cifar10':5, 'cifar100':5, 'cinic':2}

        # args['backdoor_train_size'] = int(args['poison_rate'] * length_dict[args['dataset']] / (args['n_passive_party']+1))
        args['backdoor_train_size'] = int(args['poison_rate'] * length_dict[args['dataset']])
        args['backdoor_epochs'] = epochs_dict[args['dataset']]
        if args['backdoor'] == 'lr_ba':
            args['train_label_size'] = args['backdoor_train_size'] * args['num_classes']

        if args['backdoor'] == 'villain':
            args['trigger'] = 'feature'
            args['trigger_add'] = True
            args['generation'] = False
        elif args['backdoor'] == 'lr_ba':
            args['trigger'] = 'pixel'
            args['trigger_add'] = False
            args['generation'] = False
        elif args['backdoor'] == 'splitNN':
            args['trigger'] = 'feature'
            args['trigger_add'] = False
            args['generation'] = False
        if args['backdoor'] == 'sr_ba':
            if args['trigger'] == 'pixel':
                args['trigger_add'] = True
                args['generation'] = True
            else:
                args['trigger_add'] = False
                args['generation'] = True

        if not args['active_top_trainable']:
            args['model_type'] = 'FCN'
        if temp.load_time != 'none':
            args['load_model'] = 1
            args['load_time'] = temp.load_time
        if args['n_passive_party'] == 1:
            args['adversary'] = 1
        elif args['n_passive_party'] == 3:
            args['adversary'] = 2
        elif args['n_passive_party'] == 5:
            args['adversary'] = 4
        elif args['n_passive_party'] == 7:
            args['adversary'] = 5
        else:
            args['adversary'] = (args['n_passive_party'] - 1) // 2


        logging.info("################################ Prepare Data ############################")
        train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
        backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, \
        s_r_train_dl, sr_ba_backdoor_target_indices, labeled_indices_dict, backdoor_y_test_true = get_dataloader(args)

        if args['dataset'] == 'nus_wide':
            args['backdoor_train_size'] = len(sr_ba_backdoor_target_indices)
            args['backdoor_test_size'] = -1


        run_compare_experiments(train_loader=train_dl,
                                test_loader=test_dl,
                                backdoor_train_loader=backdoor_train_dl,
                                backdoor_test_loader=backdoor_test_dl,
                                g_r_train_loader=g_r_train_dl,
                                args=args,
                                backdoor_indices=backdoor_indices,
                                backdoor_target_indices=backdoor_target_indices,
                                labeled_loader=labeled_dl,
                                unlabeled_loader=unlabeled_dl,
                                s_r_train_dl=s_r_train_dl,
                                sr_ba_backdoor_target_indices=sr_ba_backdoor_target_indices,
                                labeled_indices_dict=labeled_indices_dict,
                                backdoor_y_test_true=backdoor_y_test_true)

    else:
        param_grid = {
            'learning_rate': [0.1, 0.01, 0.001, 0.0005],
            'batch_size': [32, 64, 128, 256],
            'step-gamma': [0.1, 0.01, 0.005, 0.0005],
            'gamma': [1, 5],
            'pattern_lr': [0.1, 0.01, 0.001],
            # 'gc', 'ppdl', 'lap-noise'
            # 'gc-preserved-percent': [0.6, 0.8],
            # 'ppdl-theta-u': [0.5, 0.75, 0.85, 0.9],
            # 'noise-scale': [0.0006]
        }
        # choose different args every time
        choosen_args = []
        MAX_EVALS = 100
        random_num = temp.random

        args = get_args(temp)
        args['cuda'] = 1 if torch.cuda.is_available() else 0
        args['target_epochs'] = temp.epochs
        args['backdoor'] = temp.backdoor
        args['gradient_compression'] = temp.gc
        args['gc_percent'] = temp.gc_percent
        args['noisy_gradients'] = temp.noisy_gradients
        args['noise_scale'] = temp.noise_scale
        args['mal_optim'] = temp.mal_optim
        args['active_top_trainable'] = temp.top_model
        args['trigger'] = temp.trigger
        args['debug'] = temp.debug
        args['generation'] = temp.generation
        args['trigger_add'] = temp.trigger_add
        args['pattern_lr'] = temp.pattern_lr
        args['trigger_size'] = temp.trigger_size
        args['trigger_middle'] = temp.trigger_middle

        for i in range(0, MAX_EVALS):
            if len(choosen_args) == random_num:
                break
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            if hyperparameters in choosen_args:
                continue
            else:
                choosen_args.append(hyperparameters)

            args['passive_bottom_lr'] = hyperparameters['learning_rate']
            args['active_bottom_lr'] = hyperparameters['learning_rate']
            args['active_top_lr'] = hyperparameters['learning_rate']
            args['target_batch_size'] = hyperparameters['batch_size']
            args['passive_bottom_gamma'] = hyperparameters['step-gamma']
            args['active_bottom_gamma'] = hyperparameters['step-gamma']
            args['active_top_gamma'] = hyperparameters['step-gamma']
            args['s_r_amplify_ratio'] = hyperparameters['gamma']
            args['pattern_lr'] = hyperparameters['pattern_lr']

            # args['noise_scale'] = hyperparameters['noise-scale']

            # if args.add_defense:
            #     if hyperparameters['defense'] == 'gc':
            #         args.gc = True
            #         args.gc_preserved_percent = hyperparameters['gc-preserved-percent']
            #     elif hyperparameters['defense'] == 'ppdl':
            #         args.ppdl = True
            #         args.ppdl_theta_u = hyperparameters['ppdl-theta-u']
            #     elif hyperparameters['defense'] == 'lap-noise':
            #         args.lap_noise = True
            #         args.noise_scale = hyperparameters['noise-scale']

            logging.info("################################ Prepare Data ############################")
            train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
            backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, \
            s_r_train_dl, sr_ba_backdoor_target_indices = get_dataloader(args)

            if args['dataset'] == 'nus_wide':
                args['backdoor_train_size'] = len(sr_ba_backdoor_target_indices)
                args['backdoor_test_size'] = -1

            run_compare_experiments(train_loader=train_dl,
                                    test_loader=test_dl,
                                    backdoor_train_loader=backdoor_train_dl,
                                    backdoor_test_loader=backdoor_test_dl,
                                    g_r_train_loader=g_r_train_dl,
                                    args=args,
                                    backdoor_indices=backdoor_indices,
                                    backdoor_target_indices=backdoor_target_indices,
                                    labeled_loader=labeled_dl,
                                    unlabeled_loader=unlabeled_dl,
                                    s_r_train_dl=s_r_train_dl,
                                    sr_ba_backdoor_target_indices=sr_ba_backdoor_target_indices)
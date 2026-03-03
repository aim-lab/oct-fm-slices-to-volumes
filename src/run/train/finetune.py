# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import pandas as pd
import pickle
import os
import time
import sys
import random

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import timm

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import src.utils.lr_decay as lrd
import src.utils.misc as misc
import src.utils.dist_utils as dist_utils
from src.utils.processing import build_transform
from src.utils.datasets import describe_dataset, save_patients_split, get_idx_train_val
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from src.datasets import splitter, build_dataset
from src.models import models_mgmt

from src.models import registry

from src.run.engine_finetune import train_one_epoch, evaluate, retrain_combine


def get_args_parser():
    parser = argparse.ArgumentParser('VIT model finetune for OCT classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--backbone-epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory '
                             'constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--local-model', action='store_true',
                    help='If the model is not hugging face model')

    parser.add_argument('--retrain-combine', action='store_true',
                        help="Retrain model on combination of training and validation set.")

    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze the backbone for training.')

    parser.add_argument('--lora', action='store_true',
                        help='Finetune model with LORA framework.')

    parser.add_argument('--img_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--no-augment', action='store_true',
                        help="Don't perform any data augmentation")
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-n2-m2-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune_vae', default='', type=str,
                        help='vae to load')
    parser.add_argument('--finetune_vit', default='', type=str,
                        help='vit to load')
    parser.add_argument('--contrastive-learning', action='store_true',
                        help="Perform contrastive learning to finetune embeddings model")
    parser.add_argument('--nb-pairs-mult', default=15, type=int,
                        help='sets the number of pairs to be generated where n_pairs = n_iterations * n_sentences * 2 (for pos & neg pairs)')
    parser.add_argument('--task', default='oct_volumes_test', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', type=str)
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')  #TODO Change that to a str. Why ?

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument("--dataset-name", default="oct_rambam", type=str)
    parser.add_argument('--path-patient-split', default='', type=str,
                        help='path containing the different fold and the splits')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_frames', default=1, type=int,
                        help='Number of Frames considered')
    parser.add_argument('--deterministic-sampling', action='store_true',
                        help="Whether we do Random or deterministic sampling for video models.")
    parser.add_argument('--oct-type', type=str,
                        help="Type of oct used in the dataset")
    parser.add_argument('--aggregate', type=str,
                        help="Either scan or patient.")
    parser.add_argument('--suspicious-eye-only', action='store_true',
                        help="Consider only the most suspicious eye based on the map, else consider both eyes.")
    parser.add_argument('--scan-selection', type=str, default="middle_slice",
                        help="middle_slice if we only consider middle slices or multiple slices if we consider multiple slices or volume.")
    # parser.add_argument('--path-patient-split', default="", type=str,
    #                     help="Save path for the train test split")
    # parser.add_argument('--exclude_patients', default=[], type=List,
    #                     help="List of patients we want to exclude from the analysis") # TODO add that

    parser.add_argument('--output_dir', default='trained_models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='logs',
                        help='path where to tensorboard log')
    parser.add_argument('--tensorboard', action='store_true',
                        help='track log in Tensorboard')
    parser.add_argument('--prevent-log', action='store_true',
                        help='Prevent logging')
    parser.add_argument('--prevent-model-save', action='store_true',
                        help='Prevent saving of the model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--kfold', default=5, type=int, metavar='N',
                        help='kfold')
    parser.add_argument('--nested-kfold', default=5, type=int, metavar='N',
                        help='Nested kfold')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    print(torch.get_num_threads())
    torch.set_num_threads(15)
    print(torch.get_num_threads())

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_utils.get_rank()
    print("Seed: ",seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    patient_train_test_splitter = StratifiedKFold(n_splits=args.nested_kfold, shuffle=True, random_state=seed)

    list_patients, nb_eyes = splitter(args)
    
    list_mean_test_auc = []
    list_median_test_auc = []
    for nested_fold, (train_index, test_index) in enumerate(patient_train_test_splitter.split(list_patients, nb_eyes)):

        path_output = os.path.join(args.output_dir, args.model, args.task, f"nested_fold_{nested_fold}/")
        path_logs = os.path.join(args.log_dir, args.model, args.task, f"nested_fold_{nested_fold}/")
        os.makedirs(path_output, exist_ok=True)
        os.makedirs(path_logs, exist_ok=True)

        train_patients = list_patients[train_index]
        test_patients = list_patients[test_index]

        transform_train = build_transform("train", args)
        transform_test = build_transform("test", args)
        
        dataset_train = build_dataset(args, train_patients, transform_train, random_sampling=not args.deterministic_sampling)
        dataset_val = build_dataset(args, train_patients, transform_test, random_sampling=False)
        dataset_test = build_dataset(args, test_patients, transform_test, random_sampling=False)

        train_patients = describe_dataset(dataset_train, "train")
        test_patients = describe_dataset(dataset_test, "test")

        assert set(train_patients).isdisjoint(set(test_patients)), "The train and test patient sets are not disjointed."


        sgkf = StratifiedGroupKFold(n_splits=args.kfold, shuffle=True, random_state=seed)

        labels_train = dataset_train.get_label_list()
        patient_group = dataset_train.get_patient_list()

        fpr_list = []
        tpr_list = []
        roc_auc_list = []
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(dataset_train,labels_train, patient_group)):
            
            if args.path_patient_split:
                "Loading train val patients split from file."
                path_patient_file = os.path.join(args.path_patient_split,f"nested_fold_{nested_fold}",f"patients_split_fold_{fold}.csv")
                train_idx, val_idx = get_idx_train_val(path_patient_file, dataset_train.patients)

            dataset_train_fold = torch.utils.data.Subset(dataset_train,train_idx)
            dataset_val_fold = torch.utils.data.Subset(dataset_val,val_idx)

            describe_dataset(dataset_train_fold, "train_fold")
            describe_dataset(dataset_val_fold, "validation_fold")

            fold_train_patients = set(np.array(patient_group)[train_idx])
            fold_val_patients = set(np.array(patient_group)[val_idx])

            assert fold_train_patients.isdisjoint(fold_val_patients), "The train and val patient sets are not disjointed."
            if not args.path_patient_split:
                assert len(fold_train_patients) + len(fold_val_patients) == len(set(train_patients)), "The union of the fold train and val sets is not the train patient set."

            save_patients_split(fold_train_patients, fold_val_patients, set(test_patients), os.path.join(path_logs,"split_files",f"patients_split_fold_{fold}.csv"))

            if args.distributed:
                num_tasks = dist_utils.get_world_size()
                global_rank = dist_utils.get_rank()
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train_fold, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
                print("Sampler_train = %s" % str(dataset_train_fold))
            else:
                global_rank = 0
                sampler_train = torch.utils.data.SequentialSampler(dataset_train_fold)
            if args.dist_eval:
                if len(dataset_val_fold) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val_fold, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val_fold)

            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)

            if global_rank == 0 and args.log_dir is not None and not args.eval and args.tensorboard:
                log_writer = SummaryWriter(log_dir=path_logs + "_fold_{}".format(fold))
            else:
                log_writer = None

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train_fold, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=False
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val_fold, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=False
            )

            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=False
            )
            mixup_fn = None
            mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
            if mixup_active:
                print("Mixup is activated!")
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                    label_smoothing=args.smoothing, num_classes=args.num_classes)



            model = registry.__dict__[args.model](**vars(args))

            model.to(device)

            model_without_ddp = model
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("Model = %s" % str(model_without_ddp))
            print('number of params (M): %.2f' % (n_parameters / 1.e6))

            eff_batch_size = args.batch_size * args.accum_iter * dist_utils.get_world_size()

            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * eff_batch_size / 256

            print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
            print("actual lr: %.2e" % args.lr)

            print("accumulate grad iterations: %d" % args.accum_iter)
            print("effective batch size: %d" % eff_batch_size)

            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                model_without_ddp = model.module




            param_groups = lrd.param_groups_lrd(model_without_ddp, args.model, args.weight_decay,
                                                layer_decay=args.layer_decay
                                                )
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr)


            loss_scaler = NativeScaler()

            if args.num_classes == 1:
                criterion = nn.BCEWithLogitsLoss()
            elif mixup_fn is not None:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            print("criterion = %s" % str(criterion))

            models_mgmt.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler)


            print(f"Start training for {args.epochs} epochs")
            start_time = time.time()
            max_auc = 0.0
            best_epoch = 0
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, fold, epoch, loss_scaler,
                    args.clip_grad, mixup_fn,
                    log_writer=log_writer,
                    args=args
                )

                val_stats, val_auc_roc, _, _ = evaluate(data_loader_val, model, device, path_logs, fold, epoch, mode='val',
                                                        num_class=args.num_classes, prevent_log=args.prevent_log)
                if max_auc < val_auc_roc:
                    max_auc = val_auc_roc
                    best_epoch = epoch

                    if args.output_dir and not args.prevent_model_save and not args.retrain_combine:
                        models_mgmt.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, path_save=path_output, fold=fold)

                if log_writer is not None:
                    log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
                    log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'fold': fold,
                            'epoch': epoch,
                            'n_parameters': n_parameters}

                if dist_utils.is_main_process() and not args.prevent_log:
                    if log_writer is not None:
                        log_writer.flush()
                    with open(path_logs + "training_logs/fold_{}.txt".format(fold), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            if args.retrain_combine:
                retrain_combine(args, model, dataset_train, best_epoch, criterion, path_output, path_logs, device, fold, log_writer)
                

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            state_dict_best = torch.load(path_output + 'checkpoint-best_fold_{}.pth'.format(fold), map_location='cpu')
            model_without_ddp.load_state_dict(state_dict_best['model'])
            print([param for i,param in enumerate(model.parameters()) if i<10])
            _, _, true_labels, probabilities = evaluate(data_loader_test, model_without_ddp, device,
                                                                    path_logs, fold, epoch=0, mode='test', test_patients=test_patients,
                                                                    num_class=args.num_classes, prevent_log=args.prevent_log)
            fpr, tpr, _ = metrics.roc_curve(true_labels, probabilities[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)

            mean_auc = misc.plot_kfold_roc_curve(path_logs, fpr_list, tpr_list, roc_auc_list, np.mean)
            median_auc = misc.plot_kfold_roc_curve(path_logs, fpr_list, tpr_list, roc_auc_list, np.median)

            if args.retrain_combine:
                break


        list_mean_test_auc.append(mean_auc)
        list_median_test_auc.append(median_auc)

        misc.plot_nested_kfold_ci(list_mean_test_auc, "mean", path_save=os.path.dirname(path_logs[:-1]))
        with open(os.path.join(os.path.dirname(path_logs[:-1]),'list_mean_test_auc.pkl'), 'wb') as file:
            pickle.dump(list_mean_test_auc, file)

        misc.plot_nested_kfold_ci(list_median_test_auc, "median", path_save=os.path.dirname(path_logs[:-1]))
        with open(os.path.join(os.path.dirname(path_logs[:-1]),'list_median_test_auc.pkl'), 'wb') as file:
            pickle.dump(list_median_test_auc, file)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    path_output = os.path.join(args.output_dir, args.model, args.task)
    path_logs = os.path.join(args.log_dir, args.model, args.task)

    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)

    command = " ".join(sys.argv)
    misc.log_command_to_readme(path_logs, command)

    if args.output_dir:
        Path(path_output).mkdir(parents=True, exist_ok=True)
    main(args)

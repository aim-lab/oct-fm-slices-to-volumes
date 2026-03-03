# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import datetime
import time
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, \
    multilabel_confusion_matrix
# from pycm import *
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn import metrics

import src.utils.misc as misc
from src.utils import dist_utils
import src.utils.lr_sched as lr_sched
from src.utils.datasets import ContrastiveDataset
import src.utils.lr_decay as lrd
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from src.models import models_mgmt, registry


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, fold: int, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Split: [{}] Epoch: [{}]'.format(fold, epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        frames = samples['frames']
        frames = frames.to(device, non_blocking=True)

        targets = samples['label']
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            frames, targets = mixup_fn(frames, targets)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            outputs = model(frames)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if args.distributed:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = dist_utils.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, path_save, fold, epoch, mode, num_class, prevent_log, test_patients=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Split: [{}] Epoch: [{}] {}:'.format(fold, epoch, mode)

    if not os.path.exists(os.path.join(path_save,"metrics")):
        os.makedirs(os.path.join(path_save, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(path_save, "confusion_matrix"), exist_ok=True)
        os.makedirs(os.path.join(path_save, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(path_save, "training_logs"), exist_ok=True)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    patient_ids_list = []
    eyes_list = []
    path_frames = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["frames"]
        target = batch["label"]
        patients = batch['patient_id']
        eyes = batch['eye']
        frames_path = list(zip(*batch['frames_path']))

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            output = model(images)
            if hasattr(output, "logits"):
                output = output.logits
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())
            patient_ids_list.extend(patients)
            eyes_list.extend(eyes)
            path_frames.extend(frames_path)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list, best_thresh, _ = misc.comp_threshold_pred(true_label_decode_list, prediction_list, metrics.f1_score)

    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,
                                                   labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)

    try :
        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovr', average='macro')
        auc_pr = average_precision_score(true_label_onehot_list, prediction_list, average='macro')
    except:
        auc_roc, auc_pr = np.nan, np.nan

    metric_logger.synchronize_between_processes()

    print(
        'Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc,
                                                                                                           auc_pr, F1,
                                                                                                           mcc))
    results_path = path_save + 'metrics/{}_fold_{}.csv'.format(mode, fold)
    if not prevent_log:
        with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2 = [[acc, sensitivity, specificity, precision, auc_roc, auc_pr, F1, mcc, metric_logger.loss]]
            for i in data2:
                wf.writerow(i)

    if mode == 'test' and not prevent_log:
        misc.plot_cm(true_label_decode_list, prediction_decode_list, best_thresh)
        plt.savefig(path_save + 'confusion_matrix/test_fold_{}.jpg'.format(fold), dpi=600, bbox_inches='tight')
        plt.close()

        misc.save_predictions(path_save, fold, patient_ids_list, eyes_list, path_frames, true_label_decode_list, prediction_list)  # TODO: add prediction_decode_list

    return ({k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, true_label_decode_list,
            np.array(prediction_list))


def misc_measures(confusion_matrix):
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []

    for i in range(1, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        acc.append(1. * (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1))
        sensitivity_ = 1. * cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        sensitivity.append(sensitivity_)
        specificity_ = 1. * cm1[0, 0] / (cm1[0, 1] + cm1[0, 0])
        specificity.append(specificity_)
        precision_ = 1. * cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_ * specificity_))
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_))
        mcc = (cm1[0, 0] * cm1[1, 1] - cm1[0, 1] * cm1[1, 0]) / np.sqrt(
            (cm1[0, 0] + cm1[0, 1]) * (cm1[0, 0] + cm1[1, 0]) * (cm1[1, 1] + cm1[1, 0]) * (cm1[1, 1] + cm1[0, 1]))
        mcc_.append(mcc)

    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()

    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_



def retrain_combine(args, trained_model, ds_train, nb_epochs, criterion, path_output, path_logs, device, fold, log_writer):
    """
    Retrain model on combined Train+validation set
    """
    print(f"Retraining model on Combination of training and validation set for {nb_epochs+1} epochs.")
    #reload model
    model = registry.__dict__[args.model](**vars(args))
    
    print("LEN_STATE_DICT: ", len(model.state_dict().keys()))
    if args.contrastive_learning:
        model.base_model.load_state_dict(trained_model.base_model.state_dict())
        models_mgmt.freeze_model(model)
    print("LEN_STATE_DICT: ", len(model.state_dict().keys()))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    model.to(device)


    #Combine both datasets
    print(f"We're now training on {len(ds_train)} images.")
    data_loader_training_combined = torch.utils.data.DataLoader(
                ds_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
    

    param_groups = lrd.param_groups_lrd(model, args.model, args.weight_decay,
                                                layer_decay=args.layer_decay
                                                )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    mixup_fn = None

    nb_epochs = 100*(nb_epochs==0) + nb_epochs

    for epoch in range(nb_epochs+1):
        #train model
        train_stats = train_one_epoch(
        model, criterion, data_loader_training_combined,
        optimizer, device, fold, epoch, loss_scaler,
        args.clip_grad, mixup_fn,
        log_writer=log_writer,
        args=args
    )
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                'fold': fold,
                'epoch': epoch,
                'n_parameters': n_parameters}

        if dist_utils.is_main_process() and not args.prevent_log:
            if log_writer is not None:
                log_writer.flush()
            with open(path_logs + "training_logs/retrain_combine_fold_{}.txt".format(fold), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    print("LEN_STATE_DICT: ", len(model.state_dict().keys()))
    if args.output_dir and not args.prevent_model_save:
            models_mgmt.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, path_save=path_output, fold=fold)
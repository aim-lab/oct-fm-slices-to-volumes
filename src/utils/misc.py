import os
import sys
from operator import itemgetter
import glob
import re
import time
import datetime
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn import metrics
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE

import torch
import torch.distributed as dist
from torch import inf
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.utils.dist_utils import is_dist_avail_and_initialized


def extract_patient_name(img_path):
    """
    Extract patient name from image path, e.g. extact VAI01 from .../OCTai/VAI01/...
    :type name: str
    :return int
    """
    return img_path.split(os.sep)[-2]


def extract_patient_id(name):
    """
    Extract patient id from patient name (and NOT path), e.g. extact 1 from VAI01
    :type name: str
    :return int
    """
    match = re.search(r'\d+', name)
    if match:
        return int(match.group())
    return None


def extract_middle_slice(volume):
    """
    Extract middle slice from volume.
    :param list volume: List of slices in the scan.
    :param oct_type: Type of OCT scan.
    :return:
    """

    id_middle_slice = len(volume) // 2
    return volume[id_middle_slice - 1]


def extract_multiple_slices(volume, step, oct_type, nb_slices=None, start=None, stop=None):
    """
    Extract multiple slices from the volume.
    :param list volume: List of slices in the scan.
    :param int step: The step between the selected slices.
    :param int nb_slices: The nb of slices selected on both sides of the middle slice. Final number is 1 + 2*nb_slices.
    :param int start: In case nb_slices not defined, the index of the first slice selected.
    :param int stop: In case nb_slices not defined, the index of the last slice selected.
    """
    assert nb_slices is not None or start is not None and stop is not None

    id_middle_slice = len(volume) // 2 - 1
    if nb_slices:
        start = max(id_middle_slice - step * nb_slices,0)
        stop = min(id_middle_slice + step * nb_slices,len(volume)-1)

    list_idx = range(start, stop+1, step)
    if oct_type != "rnfl":
        return itemgetter(*list_idx)(volume)
    else:
        return [itemgetter(*list_idx)(volume)]



def extract_middle_slice_deprecated(volume, oct_type):
    """
    DEPRECATED
    Extract middle slice from volume.
    :param list volume: List of slices in the scan.
    :param oct_type: Type of OCT scan.
    :return:
    """
    min_nb_scan, max_nb_scan, _ = get_thresh_nb_slices(oct_type)
    if len(volume) >= min_nb_scan and len(volume) <= max_nb_scan:
        id_middle_slice = len(volume) // 2
        return volume[id_middle_slice - 1]
    else:
        return


def expand_dir_by_volume(images_dir, exclude_patients, oct_type='all', dataset_name="oct_rambam", verbose=0, exclude=True):
    """
    This function expands the images directory by filtering and organizing images based on patient and eye type.

    Args:
    images_dir (str): The base directory containing patient folders with images.
    exclude_patients (list): List of patient names to exclude from processing.
    oct_type (str): Type of OCT image to filter (default is 'all' for no filter).
    verbose (int): If 1, show how many eyes and patients we drop at each stage.

    Returns:
    list: A list of lists, where each sublist contains image paths for a specific patient and eye type.
    dict: A dictionary with counts of images per patient.
    """
    dir_expanded = []
    img_counter = defaultdict(int)

    ext = get_file_extension(dataset_name)
    if oct_type == 'all':
        crawl_path = os.path.join(images_dir, "*", f"*.{ext}")
    else:
        crawl_path = os.path.join(images_dir, "*", f"*{oct_type.upper()}*.{ext}")

    thresh_nb_slices, _, _ = get_thresh_nb_slices(oct_type)

    images_dict = defaultdict(lambda: defaultdict(list))

    for img_path in glob.glob(crawl_path):
        patient_name = img_path.split(os.sep)[-2]
        patient_id = extract_patient_id(patient_name)

        if patient_name not in exclude_patients:
            parts = os.path.basename(img_path).replace("--","-").replace("_","-").split('-')
            eye_type = parts[2][:2]  # Extract OS or OD

            images_dict[patient_name][eye_type].append(img_path)

    if verbose:
        verbose_expand_dir(images_dict, thresh_nb_slices, exclude_patients)

    # Convert dictionary to required list format
    for patient in sorted(images_dict.keys()):
        patient_scans = [] #new
        nb_scan_patient = sum([len(v) for v in images_dict[patient].values()])
        nb_eyes = len(images_dict[patient].keys())
        if nb_scan_patient == thresh_nb_slices*2 or not thresh_nb_slices and nb_eyes == 2 or oct_type == "all" or not exclude:
            for eye in sorted(images_dict[patient].keys(), reverse=True):
                # dir_expanded.append(sorted(images_dict[patient][eye]))
                patient_scans.append(sorted(images_dict[patient][eye]))
                img_counter[patient] += 1
            dir_expanded.append([list(pair) for pair in zip(*patient_scans)])

    return dir_expanded, img_counter

def safe_extract(patient):
    try:
        return patient[1]
    except :  # Replace with the specific exception you're handling
        return None


def convert_patients_to_scans(expanded_dir, use_middle_slice):
    """
    Convert the expanded dir on patient basis to a dir on scan basis.
    Args:
        expanded_dir (List): Expanded dir by patient basis.
        use_middle_slice (bool): Wether we have extracted only the middle slice.
    """
    if not use_middle_slice:
        left = [[scans[0] for scans in patient] for patient in expanded_dir]
        right = [[scans[1] for scans in patient] for patient in expanded_dir]

    else:
        left = [patient[0] for patient in expanded_dir]
        right = [safe_extract(patient) for patient in expanded_dir] #In case there's only one eye

    return left + right


def comp_threshold_pred(true_labels, probabilities, metric_fn):
    """
    """
    best_metric = 0
    best_threshold = 0
    scores = np.array(probabilities)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores, pos_label=1)
    for thresh in thresholds:
        pred = (scores > thresh).astype(int)
        metric = metric_fn(true_labels, pred)

        if metric > best_metric:
            best_threshold = thresh
            best_metric = metric
    return (scores > best_threshold).astype(int), best_threshold, best_metric


def plot_cm(true_labels, probabilities, best_thresh):
    cm = metrics.confusion_matrix(true_labels, probabilities, labels=[0, 1])
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in (cm / np.sum(cm, axis=1, keepdims=True)).flatten()]
    labels = [f"{v2}\n{v3}" for v2, v3 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')


def save_predictions(path_save, fold, test_patients, eyes, path_frames, labels, predictions,):

    data_save = {"patient_id": test_patients,
                 "eyes": eyes,
                 "path_frames": path_frames,
                 "labels": labels,
                 "predictions": predictions}

    # with open(os.path.join(path_save, "predictions.pkl"), 'wb') as f:
    #     pickle.dump(data_save, f)

    pd.DataFrame(data_save).to_csv(os.path.join(path_save, "predictions", "fold_{}.csv".format(fold)), index=False)


def plot_kfold_roc_curve(path_save, fpr_list, tpr_list, roc_auc_list, fnct):

    all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_list]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(tpr_list)

    tpr_se = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        tpr_se += (np.interp(all_fpr, fpr, tpr) - mean_tpr) ** 2
    tpr_se = np.sqrt(tpr_se / len(tpr_list))

    plt.figure()

    plt.plot(all_fpr, mean_tpr, color='b', label=f'{fnct.__name__} ROC (AUC = {fnct(roc_auc_list):.2f})')
    plt.fill_between(all_fpr, mean_tpr - tpr_se, mean_tpr + tpr_se, color='grey', alpha=0.2, label='Standard Error')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC Curve with {fnct.__name__} and Standard Error')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path_save, f'{fnct.__name__}_roc_curve_kfold.jpg'), dpi=600, bbox_inches='tight')
    plt.close()

    return fnct(roc_auc_list)


def get_confidence_interval(values):
    #TODO: Add docstring

    mean = np.mean(values)

    # Step 2: Calculate the standard error of the mean (SEM)
    sem = stats.sem(values)

    # Step 3: Determine the t-value for a 95% confidence interval with 4 degrees of freedom
    confidence_level = 0.95
    degrees_freedom = len(values) - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    # Step 4: Calculate the confidence interval
    margin_of_error = t_value * sem
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return mean, margin_of_error, confidence_interval


def plot_nested_kfold_ci(list_test_auc, fcnt_name, path_save=None):
    #TODO: add docstring

    mean, margin_of_error, ci = get_confidence_interval(list_test_auc)

    # Step 5: Plot the data with error bars representing the confidence interval
    plt.figure(figsize=(8, 6))
    plt.errorbar(1, mean, yerr=margin_of_error, fmt='o', color='blue', capsize=5)
    plt.xlim(0.5, 1.5)
    plt.ylim(0, 1)
    plt.title('Mean AUC ROC over the 5 train/test splits with 95% Confidence Interval')
    plt.xticks([])
    plt.ylabel('Value')
    plt.grid(True)

    # Annotate the confidence interval on the plot
    plt.text(1, ci[1]+0.1, f'[{ci[0]:.2f} , {ci[1]:.2f}]', horizontalalignment ='center')
    plt.text(1, ci[1]+0.05, f'{mean:.2f} ± {margin_of_error:.2f}', horizontalalignment ='center')

    if path_save:
        plt.savefig(os.path.join(path_save, f'{fcnt_name}_nested_kfold_CI.jpg'), dpi=600, bbox_inches='tight')
    else:
        plt.show()
    plt.close()




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def log_command_to_readme(path_readme, command):
    with open(os.path.join(path_readme, "README.md"), "a") as f:
        f.write("\n## Last Run Command\n")
        f.write("```\n")
        f.write(f"python {command}\n")
        f.write("```\n")


def get_thresh_nb_slices(oct_type):
    if oct_type.lower() == "gcl":
        min_nb = 0
        max_nb = 61
        exact_nb = 61
    elif oct_type.lower() == "onh":
        min_nb = 37
        max_nb = 73
        exact_nb = 73
    else:
        min_nb = 0
        max_nb = 100
        exact_nb = 0
    return exact_nb, min_nb, max_nb


def get_file_extension(dataset_name):
    if dataset_name == "oct_rambam":
        return "tif"
    
    elif dataset_name == "gamma" or dataset_name=="HY":
        return "jpg"


def verbose_expand_dir(images_dict, thresh_nb_slices, exclude_patients):
    remove_patient_nb_slices = set()
    remove_patient_missing_eye = set()
    count_all_scan = 0
    count_rm_one_eye_scan = 0
    count_non_std_scan = 0
    for patient in sorted(images_dict.keys()):
        if patient in exclude_patients:
            images_dict.pop(patient)
        else:
            nb_eyes = len(images_dict[patient].keys())
            count_all_scan+= nb_eyes
            if nb_eyes ==1 :
                if len(list(images_dict[patient].values())[0]) != thresh_nb_slices and  thresh_nb_slices:
                    remove_patient_nb_slices.add(patient)
                    count_non_std_scan+=1
                else:
                    remove_patient_missing_eye.add(patient)
                    count_rm_one_eye_scan+=1
            if nb_eyes == 2:
                nb_slices_eye_1, nb_slices_eye_2 = [len(v) for v in images_dict[patient].values()]
                if not nb_slices_eye_1 == nb_slices_eye_2 == thresh_nb_slices and thresh_nb_slices:
                    count_non_std_scan += (nb_slices_eye_1!=thresh_nb_slices) + (nb_slices_eye_2!=thresh_nb_slices)
                    if (nb_slices_eye_1!=thresh_nb_slices) + (nb_slices_eye_2!=thresh_nb_slices) == 2:
                        remove_patient_nb_slices.add(patient)
                    else:
                        count_rm_one_eye_scan+=1
                        remove_patient_missing_eye.add(patient)
    print(f"We have started from {len(images_dict.keys())} patients with {count_all_scan} scans.")
    print(f"We have removed {len(remove_patient_nb_slices)} patients because of {count_non_std_scan} scans with more or less than {thresh_nb_slices} slices.")
    print(f"We have now {len(images_dict.keys()) - len(remove_patient_nb_slices)} patients with {count_all_scan - count_non_std_scan} scans.")
    print(f"We have then removed {len(remove_patient_missing_eye)} patients, i.e. {count_rm_one_eye_scan} scans because patients missing one eye.")
    print(f"We end up with {len(images_dict.keys()) - len(remove_patient_nb_slices) - len(remove_patient_missing_eye)} with {count_all_scan - count_non_std_scan - count_rm_one_eye_scan} scans.")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def concat_pil_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with a width that can fit both images side by side
    new_image = Image.new('RGB', (width1 + width2, max(height1, height2)))

    # Paste the first image at (0, 0)
    new_image.paste(image1, (0, 0))

    # Paste the second image at (width1, 0)
    new_image.paste(image2, (width1, 0))

    return new_image


def plot_cosine_similarity_distribution(cosine_sim_list, labels, path_logs, fold, epoch):
    if isinstance(cosine_sim_list, list):
        cosine_sim_list = np.array(cosine_sim_list)
    for label in np.unique(labels):
        sns.kdeplot(cosine_sim_list[labels == label], label=str(int(label)), fill=True, common_norm=False)
    plt.legend()
    plt.savefig(os.path.join(path_logs, f'cosine_similarity_distribution_{fold}_{epoch}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_tnse_embeddings(ds, model, path_logs, fold, epoch, device):

    list_embeddings = []
    list_labels = []

    for i, (samples,targets) in enumerate(ds):
        samples = samples.to(device, non_blocking=True)
        emb = model(samples.unsqueeze(0)).last_hidden_state[:,0]
        list_embeddings.append(emb.squeeze().cpu().detach())
        list_labels.append(targets)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(torch.stack(list_embeddings))

    plt.figure(figsize=(8, 6))
    for label in np.unique(list_labels):
        plt.scatter(embeddings_2d[list_labels == label, 0], embeddings_2d[list_labels == label, 1],
                    label=f'Label {label}', alpha=0.7)

    plt.legend()
    plt.title('t-SNE Visualization of Image Embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig(os.path.join(path_logs, f'embeddings_tsne_{fold}_{epoch}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def extract_slices_volume(expanded_dir):
    left_eye_lists = []
    right_eye_lists = []

    # Process each patient's data
    for patient_data in expanded_dir:
        # Total number of images
        num_images = len(patient_data)
        
        # Calculate middle and offsets
        middle_index = num_images // 2 - 1
        indices = [middle_index - 20, middle_index - 10, middle_index, middle_index + 10, middle_index + 20]
        
        # Ensure indices are within bounds
        indices = [i for i in indices if 0 <= i < num_images]
        
        # Extract images based on indices
        left_eye = [patient_data[i][0] for i in indices]
        left_eye_lists.append(left_eye)

        try:
            right_eye = [patient_data[i][1] for i in indices]
            right_eye_lists.append(right_eye)
        except:
            pass

    # Combine results
    result = left_eye_lists + right_eye_lists
    return result


def normalize_tensor(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """
    Normalize a tensor based on mean and std.
    
    Parameters:
    - tensor (torch.Tensor): The input tensor. Shape should start with (batch_size, channels, ...).
    - mean (tuple or list): Mean values for each channel.
    - std (tuple or list): Standard deviation values for each channel.
    
    Returns:
    - torch.Tensor: The normalized tensor.
    """
    # Convert mean and std to tensors
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, *[1] * (tensor.ndim - 2))
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, *[1] * (tensor.ndim - 2))
    
    # Normalize the tensor
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def denormalize_tensor(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """
    Denormalize a tensor based on mean and std.
    
    Parameters:
    - tensor (torch.Tensor): The normalized tensor. Shape should start with (batch_size, channels, ...).
    - mean (tuple or list): Mean values for each channel.
    - std (tuple or list): Standard deviation values for each channel.
    
    Returns:
    - torch.Tensor: The denormalized tensor.
    """
    # Convert mean and std to tensors
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, *[1] * (tensor.ndim - 2))
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, *[1] * (tensor.ndim - 2))
    
    # Denormalize the tensor
    denormalized_tensor = tensor * std + mean
    return denormalized_tensor

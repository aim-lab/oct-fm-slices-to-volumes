import os
from collections import defaultdict,Counter
import numpy as np
import random

import torch
from torchvision.transforms import functional as F

from .build import DATASET_REGISTRY, SPLITTER_REGISTRY
from ..utils.processing import consistent_transform


@SPLITTER_REGISTRY.register()
def CirrusoctSplitter(args):

    patient_scan_count = defaultdict(int)
    for oct in os.listdir(args.data_path):
        status, patient_id, _, _, _, eye = oct.split('-')
        patient_scan_count[patient_id] += 1

    return np.array(list(patient_scan_count.keys())), np.array(list(patient_scan_count.values()))


@DATASET_REGISTRY.register()
def CirrusoctDataset(args, patient_list, transforms, random_sampling):
    
    if args.aggregate.lower()=="scan":
        return CirrusoctScan(args, patient_list, transforms=transforms)
    
    elif args.aggregate.lower()=="volume":
        return CirrusoctVolume(args, patient_list, transforms, random_sampling)
    
    else:
        raise NotImplementedError(f"Invalid aggregation type '{args.aggregate}'. Use 'scan' or 'volume'.")


class CirrusoctScan(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms):
        
        self.root_folder = args.data_path
        self.patients = patients
        self.num_frames = args.num_frames
        self.transforms = transforms
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()


    def _build_index(self):
        volumes = []
        scan_groups = defaultdict(dict)
        for oct in os.listdir(self.root_folder):
            status, patient_id, _, _, _, eye = oct.split('.')[0].split('-')
            label = int(status.lower() == "poag")

            if patient_id in self.patients:
                scan_groups[patient_id][eye] = (os.path.join(self.root_folder,oct),label)
            
        for patient_id, scans in scan_groups.items():
            if "OD" in scans or "OS" in scans:   #Replace or by and if we want two eyes patients only
                for eye, (volume_path,label) in scans.items():
                    volumes.append((patient_id, eye, volume_path, label))
                    self.label_counts[label] += 1
        
        return volumes
    

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        patient_id, eye, volume_path, label = self.samples[idx]

        oct_scan = np.load(volume_path)
        mid_index = len(oct_scan) // 2

        oct_slice = F.to_pil_image(oct_scan[mid_index]).convert("RGB")

        if self.transforms:
            oct_slice = self.transforms(oct_slice)
        
        return {"patient_id": patient_id, "eye": eye, "frames_path": [volume_path], "frames": oct_slice, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[3] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
    

class CirrusoctVolume(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms, random_sampling):
        
        self.root_folder = args.data_path
        self.patients = patients
        self.num_frames = args.num_frames
        self.random_sampling = random_sampling
        self.transforms = transforms
        self.model = args.model
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()


    def _build_index(self):
        volumes = []
        scan_groups = defaultdict(dict)
        for oct in os.listdir(self.root_folder):
            status, patient_id, _, _, _, eye = oct.split('.')[0].split('-')
            label = int(status.lower() == "poag")

            if patient_id in self.patients:
                scan_groups[patient_id][eye] = (os.path.join(self.root_folder,oct),label)
            
        for patient_id, scans in scan_groups.items():
            if "OD" in scans or "OS" in scans:   #Replace or by and if we want two eyes patients only
                for eye, (volume_path,label) in scans.items():
                    volumes.append((patient_id, eye, volume_path, label))
                    self.label_counts[label] += 1
        
        return volumes
    

    def __len__(self):
        return len(self.samples)
    
    def _sample_index(self, nb_frames):
        """
        Sample slices based on `self.random_sampling`.
        """

        if nb_frames < self.num_frames:
            raise ValueError(f"Not enough scans to sample {self.num_frames} frames.")

        if self.random_sampling:
            # Random Sampling Logic
            middle_slice_idx = nb_frames // 2 - 1
            remaining_frames = self.num_frames - 1
            quartile_size = nb_frames // 4

            # Split into quartiles and randomly sample
            quartiles = [
                        range(quartile_size),  # First quartile
                range(quartile_size,2 * quartile_size),  # Second quartile
                range(2 * quartile_size,3 * quartile_size),  # Third quartile
                range(3 * quartile_size,4 * quartile_size)  # Fourth quartile
            ]
            random_samples = [random.choice(q) for q in quartiles if q and remaining_frames > 0]
            while len(random_samples) < remaining_frames:
                random_samples.append(random.choice(range(nb_frames)))

            indices = [middle_slice_idx] + random_samples[:remaining_frames]
        else:
            # Deterministic Uniform Sampling Logic
            indices = [i * (nb_frames // self.num_frames -1) for i in range(1, self.num_frames + 1)]
            

        return sorted(indices)
    
    def __getitem__(self, idx):

        patient_id, eye, volume_path, label = self.samples[idx]

        oct_scan = np.load(volume_path)
        #Change to index sampled
        indices_selected = self._sample_index(len(oct_scan))

        frames = [F.to_pil_image(oct_scan[index]).convert("RGB") for index in indices_selected]

        if self.transforms:
            # frames = [self.transforms(frame) for frame in frames]
            frames = consistent_transform(frames, self.transforms)

        frames = torch.stack(frames) #Shape (F,C,H,W)
        if self.model != "video_mae":
            frames = frames.permute(1,0,2,3) #Shape (C,F,H,W)
        
        return {"patient_id": patient_id, "eye": eye, "frames_path": [volume_path], "indices_selected":indices_selected, "frames": frames, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[3] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
    

def trim_dict(patient_scans, divide_dataset):
    total_scans = sum(patient_scans.values())
    target_scans = total_scans // divide_dataset  # Half of total scans

    sorted_patients = sorted(patient_scans.items(), key=lambda x: -x[1])  # Sort by most scans
    trimmed_dict = {}
    current_count = 0

    for patient, scans in sorted_patients:
        if current_count + scans > target_scans:
            break  # Stop once we reach half
        trimmed_dict[patient] = scans
        current_count += scans

    return trimmed_dict
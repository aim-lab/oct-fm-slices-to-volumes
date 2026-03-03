import os
import glob
import torch
import random

from PIL import Image
import numpy as np
from scipy.io import loadmat
from collections import defaultdict,Counter
from torchvision.transforms import functional as F

from .build import DATASET_REGISTRY, SPLITTER_REGISTRY
from ..utils.processing import consistent_transform


@SPLITTER_REGISTRY.register()
def A2aSplitter(args):
    patient_scan_count = defaultdict(int)
    
    for health_status in ["Control", "AMD"]:
        health_path = os.path.join(args.data_path, health_status)
        if not os.path.exists(health_path):
            continue
        
        for oct in os.listdir(health_path):
            
            patient_id = oct.split("_2013_")[-1].replace(".mat", "")
            patient_scan_count[patient_id] += 1
                    
    return np.array(list(patient_scan_count.keys())), np.array(list(patient_scan_count.values()))


@DATASET_REGISTRY.register()
def A2aDataset(args, patient_list, transforms, random_sampling):
    
    if args.aggregate.lower()=="scan":
        return A2aScan(args, patient_list, transforms=transforms)
    
    elif args.aggregate.lower()=="volume":
        return A2aVolume(args, patient_list, transforms, random_sampling)
    
    else:
        raise NotImplementedError(f"Invalid aggregation type '{args.aggregate}'. Use 'scan' or 'volume'.")
    

class A2aScan(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms):
        
        self.root_folder = args.data_path
        self.patients = set(patients)
        self.num_frames = args.num_frames
        self.transforms = transforms
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        for label, condition in enumerate(["Control", "AMD"]):
            condition_path = os.path.join(self.root_folder, condition)

            for oct in os.listdir(condition_path):

                oct_path = os.path.join(condition_path,oct)
                patient_id = oct.split("_2013_")[-1].replace(".mat", "")

                if patient_id in self.patients:

                    samples.append((patient_id, oct_path, label))
                    self.label_counts[label] += 1
                            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        patient_id, volume_path, label = self.samples[idx]

        oct_scan = loadmat(volume_path)
        oct_scan = np.transpose(oct_scan['images'],(2,0,1))
        mid_index = len(oct_scan) // 2

        oct_slice = F.to_pil_image(oct_scan[mid_index]).convert("RGB")

        if self.transforms:
            oct_slice = self.transforms(oct_slice)
        
        return {"patient_id": patient_id, "eye": "UNK", "frames_path": [volume_path], "frames": oct_slice, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[2] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
    

class A2aVolume(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms, random_sampling):
        
        self.root_folder = args.data_path
        self.patients = set(patients)
        self.num_frames = args.num_frames
        self.random_sampling = random_sampling
        self.transforms = transforms
        self.model = args.model
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        for label, condition in enumerate(["Control", "AMD"]):
            condition_path = os.path.join(self.root_folder, condition)

            for oct in os.listdir(condition_path):

                oct_path = os.path.join(condition_path,oct)
                patient_id = oct.split("_2013_")[-1].replace(".mat", "")

                if patient_id in self.patients:

                    samples.append((patient_id, oct_path, label))
                    self.label_counts[label] += 1
                            
        return samples
    
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

        patient_id, volume_path, label = self.samples[idx]

        oct_scan = loadmat(volume_path)
        oct_scan = np.transpose(oct_scan['images'],(2,0,1))
        indices_selected = self._sample_index(len(oct_scan))

        frames = [F.to_pil_image(oct_scan[index]).convert("RGB") for index in indices_selected]

        if self.transforms:
            frames = consistent_transform(frames, self.transforms)

        frames = torch.stack(frames) #Shape (F,C,H,W)
        if self.model != "video_mae":
            frames = frames.permute(1,0,2,3) #Shape (C,F,H,W)
        
        return {"patient_id": patient_id, "eye": "UNK", "frames_path": [volume_path], "frames": frames, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[2] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
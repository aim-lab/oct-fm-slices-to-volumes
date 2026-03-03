import os
import re
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
def NehutSplitter(args):
    patient_scan_count = defaultdict(int)
    
    for health_status in ["NORMAL", "DRUSEN", "CNV"]:
        health_path = os.path.join(args.data_path, health_status)
        if not os.path.exists(health_path):
            continue
        
        for oct in os.listdir(health_path):
            
            patient_id = "_".join([health_status,oct])
            patient_path = os.path.join(health_path,oct)
            patient_scan_count[patient_id] += 1 + len([folder for folder in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path,folder))])/2
                    
    return np.array(list(patient_scan_count.keys())), np.array(list(patient_scan_count.values()))


@DATASET_REGISTRY.register()
def NehutDataset(args, patient_list, transforms, random_sampling):
    
    if args.aggregate.lower()=="scan":
        return NehutScan(args, patient_list, transforms=transforms)
    
    elif args.aggregate.lower()=="volume":
        return NehutVolume(args, patient_list, transforms, random_sampling)
    
    else:
        raise NotImplementedError(f"Invalid aggregation type '{args.aggregate}'. Use 'scan' or 'volume'.")
    

class NehutScan(torch.utils.data.Dataset):
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
        for label, condition in enumerate(["NORMAL", "DRUSEN", "CNV"]):
            condition_path = os.path.join(self.root_folder, condition)

            for oct in os.listdir(condition_path):

                oct_path = os.path.join(condition_path,oct)
                patient_id = "_".join([condition,oct])

                if patient_id in self.patients:
                    
                    label = int(label > 0)

                    subdirs = [d for d in os.listdir(oct_path) if os.path.isdir(os.path.join(oct_path, d))]

                    if subdirs:  # If OS/OD folders exist
                        for eye in subdirs:
                            eye_path = os.path.join(oct_path, eye)
                            image_files = sorted(
                                os.listdir(eye_path)
                            )

                            if len(image_files) < self.num_frames:
                                continue  # Skip if not enough slices

                            if self.num_frames == 1:
                                selected_scans = [image_files[len(image_files) // 2]]
                            else:
                                indices = torch.linspace(9, len(image_files)-10, steps=self.num_frames).long().tolist()
                                selected_scans = [image_files[i] for i in indices]

                            samples.extend([(patient_id, eye, os.path.join(eye_path, scan), label) for scan in selected_scans])
                            self.label_counts[label] += len(selected_scans)
                    
                    else:  # If images are directly in the patient folder
                        image_files = sorted(
                            os.listdir(oct_path)
                        )

                        if len(image_files) < self.num_frames:
                            continue  # Skip if not enough slices

                        if self.num_frames == 1:
                            selected_scans = [image_files[len(image_files) // 2]]
                        else:
                            indices = torch.linspace(9, len(image_files)-10, steps=self.num_frames).long().tolist()
                            selected_scans = [image_files[i] for i in indices]

                        samples.extend([(patient_id, "UNK", os.path.join(oct_path, scan), label) for scan in selected_scans])
                        self.label_counts[label] += len(selected_scans)
                            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        patient_id, eye, slice_file, label = self.samples[idx]

        slice = Image.open(slice_file)
        slice = slice.convert("RGB")
        
        if self.transforms:
            slice = self.transforms(slice)

        return {"patient_id": patient_id, "eye": eye, "frames_path": [slice_file], "frames": slice, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[3] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
    

class NehutVolume(torch.utils.data.Dataset):
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
        for label, condition in enumerate(["NORMAL", "DRUSEN", "CNV"]):
            condition_path = os.path.join(self.root_folder, condition)

            for oct in os.listdir(condition_path):

                oct_path = os.path.join(condition_path,oct)
                patient_id = "_".join([condition,oct])

                if patient_id in self.patients:
                    
                    label = int(label > 0)

                    subdirs = [d for d in os.listdir(oct_path) if os.path.isdir(os.path.join(oct_path, d))]

                    if subdirs:  # If OS/OD folders exist
                        for eye in subdirs:
                            eye_path = os.path.join(oct_path, eye)
                            image_files = sorted(
                                os.listdir(eye_path)
                            )
                            
                            if len(image_files) >= self.num_frames:
                                samples.append((patient_id, eye, [os.path.join(eye_path,scan) for scan in image_files], label))
                                self.label_counts[label] += 1

                    else:

                        image_files = sorted(
                            os.listdir(oct_path)
                        )

                        if len(image_files) >= self.num_frames:
                            samples.append((patient_id, "UNK", [os.path.join(oct_path,scan) for scan in image_files], label))
                            self.label_counts[label] += 1
                            
        return samples
    
    def __len__(self):
        return len(self.samples)
    

    def _sample_scans(self, scan_files, patient_id):
        """
        Sample slices based on `self.random_sampling`.
        """

        nb_frames = len(scan_files)
        if nb_frames < self.num_frames:
            raise ValueError(f"Not enough scans to sample {self.num_frames} frames for {patient_id}.")

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
            indices = np.linspace(0, nb_frames-1, self.num_frames, dtype=int)

        selected_slices = [scan_files[i] for i in sorted(indices)]
        return selected_slices
    
    def __getitem__(self, idx):

        patient_id, eye, scan_files, label = self.samples[idx]

        frame_paths = self._sample_scans(scan_files, patient_id)  # Resample scans dynamically

        frames = [Image.open(path).convert("RGB") for path in frame_paths]  # Open all frames
        
        if self.transforms:
            frames = consistent_transform(frames, self.transforms)

        frames = torch.stack(frames) #Shape (F,C,H,W)
        if self.model != "video_mae":
            frames = frames.permute(1,0,2,3) #Shape (C,F,H,W)
        
        return {"patient_id": patient_id, "eye": eye, "frames_path": frame_paths, "frames": frames, "label": label}

    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[3] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
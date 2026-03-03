import os
import glob
import torch
import random
import pandas as pd
import re

from PIL import Image
import numpy as np
from collections import defaultdict,Counter

from .build import DATASET_REGISTRY, SPLITTER_REGISTRY
from ..utils.processing import consistent_transform


LABEL_FILES = {"Train": "glaucoma_grading_training_GT.xlsx",
               "Validation": "glaucoma_grading.xlsx",
               "Test": "glaucoma_grading_testing_GT.xlsx",
}



@SPLITTER_REGISTRY.register()
def GammaSplitter(args):

    patient_scan_counts = {}

    for split in ["Train", "Validation", "Test"]:
        modality_path = os.path.join(args.data_path, split, "multi-modality_images")

        if not os.path.exists(modality_path):
            continue

        for patient_id in os.listdir(modality_path):
            patient_path = os.path.join(modality_path, patient_id)
            if not os.path.isdir(patient_path):
                continue
            
            # Each patient should have a folder inside with the same patient_id
            scan_folder = os.path.join(patient_path, patient_id)
            if not os.path.isdir(scan_folder):
                continue
            
            patient_scan_counts[patient_id] = patient_scan_counts.get(patient_id, 0) + 1
    

    return np.array(list(patient_scan_counts.keys())), np.array(list(patient_scan_counts.values()))


@DATASET_REGISTRY.register()
def GammaDataset(args, patient_list, transforms, random_sampling):
    
    if args.aggregate.lower()=="scan":
        return GammaScan(args, patient_list, transforms=transforms)
    
    elif args.aggregate.lower()=="volume":
        return GammaVolume(args, patient_list, transforms, random_sampling)
    
    else:
        raise NotImplementedError(f"Invalid aggregation type '{args.aggregate}'. Use 'scan' or 'volume'.")
    

class GammaScan(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms):
        
        self.root_folder = args.data_path
        self.patients = set(patients)

        self.label_file = self._get_label_file()

        self.num_frames = args.num_frames
        self.transforms = transforms
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()


    def _get_label_file(self):

        label_file = pd.DataFrame()
        for split in ["Train","Validation", "Test"]:
            temp = pd.read_excel(os.path.join(self.root_folder, split, LABEL_FILES[split]))
            label_file = pd.concat([label_file,temp])

        return label_file


    def _build_index(self):
        samples = []
        for split in ["Train", "Validation", "Test"]:
            modality_path = os.path.join(self.root_folder, split, "multi-modality_images")

            if not os.path.exists(modality_path):
                continue

            for patient_id in os.listdir(modality_path):
                patient_path = os.path.join(modality_path, patient_id)
                
                if patient_id not in self.patients:
                    continue
                
                label = self.label_file[self.label_file['data']==int(patient_id)]['non'].values[0]

                scan_folder = os.path.join(patient_path, patient_id)
                image_files = sorted(os.listdir(scan_folder), key=lambda x: int(x.split("_")[0]))

                if self.num_frames == 1:
                    # Select the middle slice
                    mid_index = len(image_files) // 2
                    selected_scans = [os.path.join(scan_folder,image_files[mid_index])]

                else:
                    # Select num_frames uniformly
                    indices = torch.linspace(9, len(image_files)-10, steps=self.num_frames).long().tolist()
                    selected_scans = [os.path.join(scan_folder,image_files[i]) for i in indices]


                samples.extend([(patient_id, scan, label) for scan in selected_scans])
                self.label_counts[label] += self.num_frames
            
        return samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient, scan_file, label = self.samples[idx]

        image = Image.open(scan_file)
        image = image.convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
        
        return {"patient_id": patient, "eye":"UNK", "frames_path": [scan_file], "frames": image, "label": label}
    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[2] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)
    



class GammaVolume(torch.utils.data.Dataset):
    def __init__(self, args, patients, transforms, random_sampling):
        
        self.root_folder = args.data_path
        self.patients = set(patients)

        self.model = args.model

        self.label_file = self._get_label_file()

        self.num_frames = args.num_frames
        self.random_sampling = random_sampling

        self.transforms = transforms
        self.samples = []
        self.label_counts = Counter()
        self.samples = self._build_index()


    def _get_label_file(self):

        label_file = pd.DataFrame()
        for split in ["Train","Validation", "Test"]:
            temp = pd.read_excel(os.path.join(self.root_folder, split, LABEL_FILES[split]))
            label_file = pd.concat([label_file,temp])

        return label_file


    def _build_index(self):
        samples = []
        for split in ["Train", "Validation", "Test"]:
            modality_path = os.path.join(self.root_folder, split, "multi-modality_images")

            if not os.path.exists(modality_path):
                continue

            for patient_id in os.listdir(modality_path):
                patient_path = os.path.join(modality_path, patient_id)
                
                if patient_id not in self.patients:
                    continue
                
                label = self.label_file[self.label_file['data']==int(patient_id)]['non'].values[0]

                scan_folder = os.path.join(patient_path, patient_id)
                image_files = sorted(os.listdir(scan_folder), key=lambda x: int(x.split("_")[0]))

                samples.append((patient_id, [os.path.join(scan_folder, img_path) for img_path in image_files], label))
                self.label_counts[label] += 1
            
        return samples



    def _sample_scans(self, scan_files):
        """
        Sample slices based on `self.random_sampling`.
        """

        num_images = len(scan_files)
        if num_images < self.num_frames:
            raise ValueError(f"Not enough scans to sample {self.num_frames} frames.")

        if self.random_sampling:
            # Random Sampling Logic
            middle_slice_idx = num_images // 2 - 1
            middle_slice = scan_files[middle_slice_idx]
            remaining_frames = self.num_frames - 1
            quartile_size = num_images // 4

            # Split into quartiles and randomly sample
            quartiles = [
                scan_files[:quartile_size],  # First quartile
                scan_files[quartile_size:2 * quartile_size],  # Second quartile
                scan_files[2 * quartile_size:3 * quartile_size],  # Third quartile
                scan_files[3 * quartile_size:]  # Fourth quartile
            ]
            random_samples = [random.choice(q) for q in quartiles if q and remaining_frames > 0]
            while len(random_samples) < remaining_frames:
                random_samples.append(random.choice(scan_files))

            selected_slices = [middle_slice] + random_samples[:remaining_frames]
            selected_slices = sorted(selected_slices, key=lambda x: int(re.search(r'(\d+)_', x).group(1)))

        else:
            # Deterministic Uniform Sampling Logic
            # indices = [i * (num_images // (self.num_frames + 1) -1) for i in range(1, self.num_frames + 1)]
            indices = np.linspace(1,num_images,self.num_frames,dtype=int,endpoint=False)
            selected_slices = [scan_files[i] for i in indices]

        return selected_slices


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient, scan_files, label = self.samples[idx]

        frame_paths = self._sample_scans(scan_files)  # Resample scans dynamically

        success = False
        while not success:
            try:
                frames = [Image.open(path).convert("RGB") for path in frame_paths]  # Open all frames
                success = True
            except:
                print(frame_paths)
                frame_paths = self._sample_scans(scan_files)

        if self.transforms:
            # frames = [self.transforms(frame) for frame in frames]
            frames = consistent_transform(frames, self.transforms)
        
        frames = torch.stack(frames) #Shape (F,C,H,W)
        if self.model != "video_mae":
            frames = frames.permute(1,0,2,3) #Shape (C,F,H,W)

        return {"patient_id": patient, "eye":"UNK", "frames_path": frame_paths, "frames": frames, "label": label}
    
    def get_patient_list(self):
        return [sample[0] for sample in self.samples]
    
    def get_label_list(self):
        return [sample[2] for sample in self.samples]
    
    def get_label_counts(self):
        return dict(self.label_counts)

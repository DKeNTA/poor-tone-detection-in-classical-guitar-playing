import os
import glob

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.data_transforms import get_transforms

class MyDataset(Dataset):
    def __init__(self, filepaths, labels=None, transform=None):
        self.filepaths = filepaths
        self.labels = labels if labels is not None else []
        self.transform = transform if transform is not None else lambda x: x

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio_file = self.filepaths[idx]

        y, sr = librosa.load(audio_file, sr=None) 
        data = self.transform(y, sr)

        if len(self.labels) >= 1:
            label = torch.tensor(self.labels[idx])
            return data, label
        else:
            return data, torch.tensor(-1)  # Return a dummy label
    
class MyDataLoader:
    def __init__(self, dataset_path, batch_size, trace_func=print, sr=44100, duration=0.5, frame_length=1380, hop_length=345, n_mels=128, pitch_shift=5, random_seed=42):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.trace_func = trace_func 
        self.sr = sr
        self.duration = duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.pitch_shift = pitch_shift
        self.random_seed = random_seed

        self.train_transform, self.test_transform = get_transforms(sr, duration, frame_length, hop_length, n_mels, pitch_shift, mode='train')

    def get_dataloader_for_train(self):
        filepaths = glob.glob(os.path.join(self.dataset_path, 'train/**/*.wav'), recursive=True)
        label_dict = {
            'unlabeled': 0,
            'good': 1,
            'buzzing_on_plucking': -1,
            'buzzing_during_fretting': -2,
            'muffled': -3,
            'muted': -4,
            'finger_noise': -5,
            'premature_string_release': -6,
            'others': -7
        }

        unlabeled_filepaths = []
        labeled_filepaths = []
        labels = []
        for file in filepaths:
            for key, value in label_dict.items():
                if key in file.split('/'):
                    if value == 0:
                        unlabeled_filepaths.append(file)
                    else:
                        labeled_filepaths.append(file)
                        labels.append(value)
                    break
            else:
                raise ValueError(f"Error: Unexpected file name: {file}")
            
        train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(labeled_filepaths, labels, test_size=0.2, random_state=self.random_seed, stratify=labels)
        train_filepaths += unlabeled_filepaths
        train_labels = [label if label >= 0 else -1 for label in train_labels] +  [label_dict['unlabeled']] * len(unlabeled_filepaths)
        val_labels = [label if label >= 0 else -1 for label in val_labels]

        train_dataset = MyDataset(train_filepaths, labels=train_labels, transform=self.train_transform)
        val_dataset = MyDataset(val_filepaths, labels=val_labels, transform=self.test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.trace_func("Loaded data for train.")

        return train_dataloader, val_dataloader

    def get_dataloader_for_pretrain(self):
        filepaths = glob.glob(os.path.join(self.dataset_path, 'train/**/*.wav'), recursive=True)
        label_dict = {
            'unlabeled': 0,
            'good': 1,
            'poor': -1
        }

        if self.cross_validation:
            unlabeled_filepaths = []
            for file in filepaths:
                for key, value in label_dict.items():
                    if key in file.split('/'):
                        if value == 0:
                            unlabeled_filepaths.append(file)
                        break
                else:
                    raise ValueError(f"Error: Unexpected file name: {file}")
                
            train_filepaths, val_filepaths = train_test_split(unlabeled_filepaths, test_size=0.1)
            setting_c_filepaths = train_filepaths
        
        else:
            unlabeled_filepaths = []
            labeled_filepaths = []
            for file in filepaths:
                for key, value in label_dict.items():
                    if key in file.split('/'):
                        if value == 0:
                            unlabeled_filepaths.append(file)
                        elif value == 1:
                            labeled_filepaths.append(file)
                        break
                else:
                    raise ValueError(f"Error: Unexpected file name: {file}")
        
            train_labeled_filepaths, val_filepaths = train_test_split(labeled_filepaths, test_size=0.2, random_state=self.random_seed)
            setting_c_filepaths = labeled_filepaths
            train_filepaths = unlabeled_filepaths + train_labeled_filepaths
            
        train_dataset = MyDataset(train_filepaths, transform=self.train_transform)
        val_dataset = MyDataset(val_filepaths, transform=self.test_transform)
        setting_c_dataset = MyDataset(setting_c_filepaths, transform=self.test_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        setting_c_dataloader = DataLoader(setting_c_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.trace_func(f"Loaded data for pretrain.")
        return train_dataloader, val_dataloader, setting_c_dataloader
    
    def get_dataloader_for_test(self, labeled=False):
        filepaths = glob.glob(os.path.join(self.dataset_path, 'test/**/*.wav'), recursive=True)
        label_dict = {
            'good': 0,
            'buzzing_on_plucking': 1,
            'buzzing_during_fretting': 2,
            'muffled': 3,
            'mute': 4,
            'finger_noise': 5,   
            'premature_string_release': 6,
            'others': 7
        }

        labels = []
        for file in filepaths:
            for key, value in label_dict.items():
                if key in file.split('/'):
                    labels.append(value)
                    break
            else:
                raise ValueError(f"Error: Unexpected file name: {file}")
            
        if labeled == False:
            labels = [label if label == 0 else 1 for label in labels]

        test_dataset = MyDataset(filepaths, labels=labels, transform=self.test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.trace_func(f"Loaded data for test.")
        return test_dataloader
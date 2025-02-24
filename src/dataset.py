import torch
import librosa
import numpy as np
import copy
import os

""" 
    Dataset Class 
"""
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, dataset_root, feature_extractor, max_duration):
        self.examples = examples['path']
        self.labels = examples['label']
        self.dataset_root = dataset_root
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.sr = 16_000

    def __getitem__(self, idx):
        try:
            audio_file = os.path.join(self.dataset_root, self.examples[idx])
            inputs = self.feature_extractor(
                librosa.load(audio_file, sr=self.sr)[0].squeeze(),
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length')
            item = {'input_values': inputs['input_values'].squeeze()}
            item["labels"] = torch.tensor(self.labels[idx])
        except:
            print("Audio not available", self.examples[idx])
            # Return a random audio sample from the dataset
            random_idx = np.random.randint(len(self.examples))
            audio_file = os.path.join(self.dataset_root, self.examples[random_idx])
            inputs = self.feature_extractor(
                librosa.load(audio_file, sr=self.sr)[0].squeeze(),
                sampling_rate=self.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length')
            item = {'input_values': inputs['input_values'].squeeze()}
            item["labels"] = torch.tensor(self.labels[random_idx])
        return item

    def __len__(self):
        return len(self.examples)
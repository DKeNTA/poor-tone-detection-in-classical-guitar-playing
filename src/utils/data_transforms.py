import math

import librosa
import numpy as np
import torch

from .signal_preprocessing import SignalPreprocessor

class RandomPitchShift:
    def __init__(self, pitch_shift_range):
        self.pitch_shift_range = pitch_shift_range

    def __call__(self, signal, sr):
        n_steps = np.random.randint(self.pitch_shift_range[0], self.pitch_shift_range[1] + 1)
        shifted_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
        return shifted_signal, sr

class WaveToFeature:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def __call__(self, signal, sr, trimming=True, normalize=True):
        feature = self.preprocessor.preprocess(signal, sr, needs_trimming=trimming, needs_normalize=normalize)
        return feature, sr
    
class OverlapFeature:
    def __init__(self, window_length, overlap_length):
        self.window_length = window_length
        self.overlap_length = overlap_length

    def __call__(self, feature):
        features_list = [feature]
        for start in range(0, feature.shape[1] - self.window_length + 1, self.overlap_length):
            end = start + self.window_length
            splitted_feature = feature[:, start:end]
            features_list.append(splitted_feature)

        return features_list

class ListToTensor:
    def __call__(self, features_list):
        data_list = [torch.tensor(feature).unsqueeze(0) for feature in features_list]
        return data_list
    
class MyTransforms:
    def __init__(self, transform, pass_sr_to=None):
        self.transform = transform
        self.pass_sr_to = pass_sr_to

    def __call__(self, data, sr):
        for i, t in enumerate(self.transform):
            if self.pass_sr_to is not None and i in self.pass_sr_to:
                data, sr = t(data, sr)
            else:
                data = t(data)
        return data
    

def get_transforms(sr, duration, frame_length, hop_length, n_mels, pitch_shift=5, mode='train'):
    preprocessor = SignalPreprocessor(sr, duration, frame_length, hop_length, n_mels)

    wave_to_feature = WaveToFeature(preprocessor)

    if mode == 'train':
        random_pitch_shift = RandomPitchShift(pitch_shift_range=(-pitch_shift, pitch_shift))

        window_length = math.ceil((duration / 2) * sr / hop_length)
        overlap_length = math.ceil(window_length / 2)
        overlap_feature = OverlapFeature(window_length=window_length, overlap_length=overlap_length)
        train_transform = MyTransforms([
            random_pitch_shift,
            wave_to_feature,
            overlap_feature,
            ListToTensor(),
            ], pass_sr_to=[0,1])

        test_transform = MyTransforms([
            wave_to_feature,
            overlap_feature,
            ListToTensor(),
            ], pass_sr_to=[0])
        
        return train_transform, test_transform

    elif mode == 'test':
        window_length = math.ceil((duration / 2) * sr / hop_length)
        overlap_length = math.ceil(window_length / 2)
        overlap_feature = OverlapFeature(window_length=window_length, overlap_length=overlap_length)

        transform = MyTransforms([
            wave_to_feature,
            overlap_feature,
            ListToTensor(),
            ], pass_sr_to=[0])
        
        return transform
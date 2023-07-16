import librosa
import numpy as np

class SignalPreprocessor:
    def __init__(self, sr=44100, duration=0.5, frame_length=1380, hop_length=345, n_mels=128):
        self.sr = sr
        self.duration = duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.num_samples = int(sr * duration)

    def shaping_data(self, signal, sr, needs_padding=True, needs_trimming=True):
        if sr != self.sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sr)

        if needs_trimming:
            signal = self._trim(signal)

        if needs_padding:
            signal = self._padding(signal)

        return signal
        
    def _trim(self, signal):
        signal = signal[:self.num_samples+1]
        return signal
    
    def _padding(self, signal, mode='constant'):
        if self._is_padding_necessary(signal):
            num_missing_samples = self.num_samples - len(signal)
            signal = np.pad(signal,
                            (0, num_missing_samples),
                            mode=mode)
        return signal

    def _is_padding_necessary(self, signal):
        return len(signal) < self.num_samples

    def _normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())

        return norm_array

    def extract_feature(self, signal, needs_normalize=True):
        spec = librosa.feature.melspectrogram(y=signal,
                                              sr=self.sr,
                                              n_fft=self.frame_length,
                                              hop_length=self.hop_length,
                                              n_mels=self.n_mels)
        feature = librosa.power_to_db(spec, ref=np.max)
        if needs_normalize:
            feature = self._normalize(feature)
        
        return feature

    def preprocess(self, signal, sr, needs_padding=True, needs_trimming=True, needs_normalize=True):
        signal = self.shaping_data(signal, sr, needs_padding=needs_padding, needs_trimming=needs_trimming)

        feature = self.extract_feature(signal, needs_normalize=needs_normalize)

        return feature
    

def split_in_onsets(signal, sr, frame_length=1024, hop_length=512, delta=0.12):
        
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length, center=True)

    onset_envelope = rms[0, 1:] - rms[0, :-1] 
    onset_envelope = np.maximum(0.0, onset_envelope)
    onset_envelope = onset_envelope / onset_envelope.max() 

    w = 175 / 1000 * sr // hop_length

    onset_frames = librosa.util.peak_pick(onset_envelope, pre_max=w, post_max=w, pre_avg=w, post_avg=w, delta=delta, wait=w)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    onset_samples = onset_times * sr
    split_signals = []
    for i in range(len(onset_samples)):

        if i == len(onset_samples)-1:
            split_signal = signal[int(onset_samples[i]):]
        else:
            split_signal = signal[int(onset_samples[i]):int(onset_samples[i+1])]

        split_signals.append(split_signal)

    return split_signals, onset_times
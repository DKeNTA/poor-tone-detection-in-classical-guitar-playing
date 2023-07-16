import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from detector.score_output import Scorer
from src.utils.data_transforms import WaveToFeature
from src.utils.signal_preprocessing import SignalPreprocessor, split_in_onsets

class PoorToneDetector:
    def __init__(self):

        self.sr = 44100
        self.duration = 0.5
        self.frame_length = 1380
        self.hop_length = 345
        self.n_mels = 128

        self.weight_path = 'src/weights/parameters_for_demo.pth'
        self.model_path = 'src/model.py'
        self.latent_dim = 128

        preprocessor = SignalPreprocessor(self.sr, self.duration, self.frame_length, self.hop_length, self.n_mels)
        self.wave_to_feature = WaveToFeature(preprocessor)
        self.scorer = Scorer(self.weight_path, self.model_path, latent_dim=self.latent_dim, sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length, n_mels=self.n_mels)

    def plot(self, signal, feature, sr, scores, onsets=[], mistakes=[], th=0.5):
        ax1 = self.fig.add_subplot(3, 1, 1)
        signal /= np.abs(signal).max()
        librosa.display.waveshow(signal, sr=sr, ax=ax1)
        ax1.set_title('Wave')

        if len(onsets) >= 1:
            if len(mistakes) >= 1:
                poor_tones = np.zeros(len(signal))
                for i in mistakes:
                    poor_tones_onset = int(onsets[i] * sr)
                    if len(onsets) == i+1:
                        poor_tones[poor_tones_onset:] += signal[poor_tones_onset:]
                    else:
                        poor_tones_offset = int(onsets[i+1] * sr) - 1
                        poor_tones[poor_tones_onset:poor_tones_offset] += signal[poor_tones_onset:poor_tones_offset]

                librosa.display.waveshow(poor_tones, sr=sr, color='r')
            plt.vlines(onsets, -1, 1, color='y', linestyle='--')

        ax1.label_outer()

        ax2 = self.fig.add_subplot(3, 1, 2, sharex=ax1)
        librosa.display.specshow(feature, sr=sr, x_axis='time', y_axis='mel', hop_length=self.hop_length, ax=ax2)
        ax2.set_title('Mel Spectrogram')
        ax2.label_outer()

        ax3 = self.fig.add_subplot(3, 1, 3, sharex=ax1)
        plt.bar(onsets, scores, width=0.15, align='edge', linewidth=5, color='r')    
        if len(onsets) > 1:
            plt.xticks(onsets, np.round(onsets, 1).tolist(), rotation=45)

        for onset, score in zip(onsets, scores):
            if score > th:
                s = round(float(score), 2)
                ax3.text(onset, th+0.1, s, ha='center', va='bottom')

        plt.ylim(0,3)
        plt.hlines(th, 0, len(signal)/sr, color='black', linestyle='--')
        plt.xlabel('Times')
        plt.ylabel('Score')

        plt.tight_layout()

    def detect_poor_tone(self, y, sr, delta, th=0.5):

        if delta == 0:
            score = self.scorer.output_score(y, sr)
            onset = np.array([0])
            # self.plot(y, feature, sr, [score], onsets=onset, th=th)

            signals = [y]
            onsets = onset.tolist()

        else:
            scores = []
            # poor_tones = []
            signals, onsets = split_in_onsets(y, sr, delta=delta)

            for i, signal in enumerate(signals):
                score = self.scorer.output_score(signal, sr)
                scores.append(score)
                # if self.scorer.is_detect(score, th):
                #     poor_tones.append(i)

            # self.plot(y, feature, sr, scores, onsets=onsets, mistakes=poor_tones, th=th)
            onsets = np.round(onsets, decimals=1).tolist()

        return signals, onsets, scores



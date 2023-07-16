import importlib
import os

import torch

from src.utils.data_transforms import get_transforms

class Scorer:
    def __init__(self, weight_path, model_path, latent_dim=64, sr=44100, duration=0.5, frame_length=1380, hop_length=345, n_mels=128):
        self.transform = get_transforms(sr, duration, frame_length, hop_length, n_mels, mode='test')

        model_path = os.path.splitext(model_path)[0].replace('/', '.')

        self.device = torch.device('cpu')

        model = importlib.import_module(model_path)
        Encoder = getattr(model, "Encoder")
        self.net = Encoder(latent_dim).to(self.device)

        state_dict = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state_dict['net_dict'])
        self.c = torch.Tensor(state_dict['center']).to(self.device)
        
        self.net.eval()

    def test_data(self, data):
        with torch.no_grad():
            x = [x_i.float().to(self.device) for x_i in data]

            z = self.net(*x[1:])
            
            score = torch.sum((z - self.c) ** 2, dim=1)

        return score[0].detach().cpu().numpy()

    def output_score(self, signal, sr):      
        data = self.transform(signal, sr)    
        data = [d.unsqueeze(0) for d in data]

        score = self.test_data(data)

        return score
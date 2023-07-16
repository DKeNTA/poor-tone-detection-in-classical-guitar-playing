import argparse
import csv
import importlib
import os
import torch

import numpy as np
from barbar import Bar

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.manifold import TSNE

from data_loader import MyDataLoader
from utils.utils import set_seed

class ModelTester:
    def __init__(self, weight_path, latent_dim, device, model_path='model', trace_func=print, save_csvfile=None):
        self.device = device
        self.trace_func = trace_func
        self.csv_file = save_csvfile

        model = importlib.import_module(model_path)
        Encoder = getattr(model, "Encoder")
        self.net = Encoder(latent_dim).to(device)

        state_dict = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(state_dict['net_dict'])
        self.c = torch.Tensor(state_dict['center']).to(device)
        
        self.net.eval()

        if save_csvfile != None:
            self.log = {'Weight Path':os.path.basename(weight_path)}

    def test_data(self, data):
        with torch.no_grad():
            x = [x_i.float().to(self.device) for x_i in data]

            z = self.net(*x[1:])
            score = torch.sum((z - self.c) ** 2, dim=1)

        return score[0].detach().cpu().numpy()
    
    def test_data_with_mapping(self, data_list, th):
        score_z = []
        with torch.no_grad():
            for x in data_list:
                x = [x_i.float().to(self.device) for x_i in x]

                z = self.net(*x[1:])
                score = torch.sum((z - self.c) ** 2, dim=1)

                score_z += list(zip(score.cpu().data.numpy().tolist(),
                                    z.cpu().data.numpy().tolist()))
        
        score, z = zip(*score_z)
        labels = [1 if s > th else 0 for s in score]
        # concat center
        z += (self.c.cpu().data.numpy(),)
        labels += [-1]
        score = np.array(score)
        z = np.array(z)
        labels = np.array(labels)
            
        self._mapping(z, labels, '2d')

    def test_dataset(self, data_loader, test_mode=None, th=None, mapping=None):
        label_score_z = []
        self.trace_func('Testing...')
        with torch.no_grad():
            for x, labels in Bar(data_loader):
                labels.to(self.device)
                x = [x_i.float().to(self.device) for x_i in x]

                z = self.net(*x[1:])
                scores = torch.sum((z - self.c) ** 2, dim=1)
                
                label_score_z += list(zip(labels.cpu().data.numpy().tolist(),
                                          scores.cpu().data.numpy().tolist(),
                                          z.cpu().data.numpy().tolist()))

        c_label = [-1]
        c_score = [0]
        label_score_z += list(zip(c_label,
                                  c_score,
                                  [self.c.cpu().data.numpy().tolist()]))
        labels, scores, z = zip(*label_score_z)
        labels = np.array(tuple(labels))
        scores = np.array(scores)
        z = np.array(z)

        if test_mode == 'labeled':
            labels_tmp = np.array(tuple(label if label==0 else 1 for label in list(labels)))
            self._evaluation(labels_tmp[:-1], scores[:-1])
            if mapping != None:
                self._mapping_labeled(z, labels, mapping)
        elif test_mode == 'pn':
            self._test_by_specifying_a_threshold(z, labels, scores, th, mapping)
        else:
            labels_tmp = np.array(tuple(label if label==0 else 1 for label in list(labels)))
            self._evaluation(labels_tmp[:-1], scores[:-1])
            if mapping != None:
                self._mapping(z, labels, mapping)

    def _evaluation(self, labels, scores):
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

        f1_max_idx = np.argmax(f1)

        f1_max = np.max(f1)
        precision_F1_max = precision[f1_max_idx]
        recall_F1_max = recall[f1_max_idx]
        threshold_F1_max = thresholds[f1_max_idx]

        #for p,r,F,t in zip(precision, recall, F_measure, thresholds):
        #    self.trace_func(f"Precision : {p:.4f}  Recall : {r:.4f}  F_measure : {F:.4f} Thresholds : {t:.4f}")

        self.trace_func(f"PR-AUC    : {100.*auc:.2f}%")
        self.trace_func(f"F measure : {f1_max:.4f}")
        self.trace_func(f"Precision : {precision_F1_max:.4f}")
        self.trace_func(f"Recall    : {recall_F1_max:.4f}")
        self.trace_func(f"Threshold : {threshold_F1_max:.4f}")
        
        if self.csv_file != None:
            result = {'PR-AUC':auc, 'F1':f1_max, 'Precision':precision_F1_max, 'Recall':recall_F1_max, 'Threshold':threshold_F1_max}
            self.log.update(result)
            fieldnames = ['Weight Path', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Threshold']
            if os.path.isfile(self.csv_file):
                with open(self.csv_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(self.log)
            else:
                with open(self.csv_file, 'w') as f:            
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(self.log)

        """
        plt.plot(recall, precision, label='PR curve (area = %.4f)'%auc)
        plt.legend()
        plt.title('PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.grid()
        plt.show()
        """

    def _mapping(self, z, labels, mapping='2d'):
        point_dict = {
            -1: {'label': 'center', 'color': 'black', 'marker': '*'},
            0:  {'label': 'good', 'color': 'blue', 'marker': 'o'},
            1:  {'label': 'poor', 'color': 'red', 'marker': 'x'},
        }
        label_num = len(point_dict) - 1

        if mapping == '2d':
            self._plot_2d(z, labels, label_num, point_dict)
        elif mapping == '3d':
            self._plot_3d(z, labels, label_num, point_dict)

    def _mapping_labeled(self, z, labels, mapping='2d'):
        point_dict = {
            -1: {'label': 'center', 'color': 'black', 'marker': '*'},
            0:  {'label': 'good', 'color': 'blue', 'marker': 'o'},
            1:  {'label': 'buzzing_on_plucking', 'color': 'red', 'marker': 'v'},
            2:  {'label': 'buzzing_during_fretting', 'color': 'pink', 'marker': '^'},
            3:  {'label': 'muffled', 'color': 'orange', 'marker': 's'},
            4:  {'label': 'muted', 'color': 'gray', 'marker': 'x'},
            5:  {'label': 'finger_noise', 'color': 'green', 'marker': '1'},
            6:  {'label': 'premature_string_release', 'color': 'olive', 'marker': 'D'},
            7:  {'label': 'others', 'color': 'magenta', 'marker': '2'},
        }
        label_num = len(point_dict) - 1

        if mapping == '2d':
            self._plot_2d(z, labels, label_num, point_dict)
        elif mapping == '3d':
            self._plot_3d(z, labels, label_num, point_dict)

    def _test_by_specifying_a_threshold(self, z, labels, scores, th, mapping='2d'):
        labels_pn = []
        for label, score in zip(labels, scores):
            if label == -1:
                label_pn = -1
            elif score < th:
                if label == 0:
                    # TN
                    label_pn = 0  
                else:
                    # FN
                    label_pn = 2
            else:
                if label == 0:
                    # FP
                    label_pn = 3
                else:
                    # TP
                    label_pn = 1

            labels_pn.append(label_pn)

        tp_num = labels_pn.count(1)
        fp_num = labels_pn.count(3) 
        fn_num = labels_pn.count(2) 
        precision = tp_num / (tp_num + fp_num)
        recall = tp_num / (tp_num + fn_num)

        F1 = 2 * precision * recall / (precision + recall)

        self.trace_func(f"F measure : {F1:.4f}\nPrecision : {precision:.4f}\nRecall    : {recall:.4f}")

        if mapping != None:
            labels_pn = np.array(tuple(labels_pn))

            point_dict = {
                    -1: {'label': 'center', 'color': 'black', 'marker': '*'},
                    0:  {'label': 'TN', 'color': 'blue', 'marker': 'o'},
                    1:  {'label': 'TP', 'color': 'green', 'marker': 'v'},
                    2:  {'label': 'FN', 'color': 'red', 'marker': '^'},
                    3:  {'label': 'FP', 'color': 'orange', 'marker': 's'}
            }
            label_num = len(point_dict) - 1

            if mapping == '2d':
                self._plot_2d(z, labels_pn, label_num, point_dict)
            elif mapping == '3d':
                self._plot_3d(z, labels_pn, label_num, point_dict)

    def _plot_2d(self, z, labels, label_num, point_dict):
        tsne_2d = TSNE(n_components=2, random_state=42)
        z_2d = tsne_2d.fit_transform(z)

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle('2D Feature Space Plot')

        ax = fig.add_subplot()
        for i in range(-1, label_num):
            target = z_2d[labels == i]
            if len(target) > 0:
                ax.scatter(target[:,0], target[:,1],
                           label=point_dict[i]['label'],
                           color=point_dict[i]['color'],
                           marker=point_dict[i]['marker'],
                           alpha=0.75 if point_dict[i]['color'] != 'black' else 1.0)
        #plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
        plt.legend(loc='best', fontsize=14)
        plt.subplots_adjust(right=0.8) 
        plt.savefig('mapping.png', bbox_inches='tight')

        plt.show()

    def _plot_3d(self, z, labels, label_num, point_dict):
        tsne_3d = TSNE(n_components=3, random_state=5)
        z_3d = tsne_3d.fit_transform(z)

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle('3D Feature Space Plot')

        ax = fig.add_subplot(projection='3d')
        for i in range(-1, label_num):
            target = z_3d[labels == i]
            if len(target) > 0:
                ax.scatter(target[:,0], target[:,1], target[:,2],
                           label=point_dict[i]['label'],
                           color=point_dict[i]['color'],
                           marker=point_dict[i]['marker'],
                           alpha=0.75 if point_dict[i]['color'] != 'black' else 1.0)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='../dataset/test/',
                        help="Test dataset path to load.")
    parser.add_argument('--weight_path', type=str, default='weights/parameters.pth',
                        help="Path to the weights of the trained model.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for processing the test dataset.")
    parser.add_argument('--latent_dim', type=int, default=128,
                        help="Latent dimensions of the model.")
    parser.add_argument('--model_path', type=str, default="model.py", 
                        help="Name of the model-module to import Encoder from")
    parser.add_argument('--threshold', type=float, default=1.0,
                        help="Threshold to use for the test.")
    parser.add_argument('--sr', type=int, default=44100,
                        help='Sampling rate.')
    parser.add_argument('--frame_length', type=int, default=1024,
                        help='Window size for the STFT used in extracting the Mel spectrogram.')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Number of samples between successive frames in the STFT.')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of Mel frequency bands to use when extracting the Mel spectrogram.')
    parser.add_argument('--multi_input', action='store_true',
                        help="Whether to use multiple inputs for the model.")
    parser.add_argument('--mapping', type=str, choices=['2d', '3d'], default=None,
                        help="Mapping method to use for visualizing the results ('2d' or '3d').")
    parser.add_argument('--mode', type=str, choices=['labeled', 'pn'], default=None,
                        help="Type of test to perform ('labeled' or 'pn').")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_dataloader = MyDataLoader(args.test_data_path, args.batch_size, sr=args.sr, frame_length=args.frame_length, hop_length=args.hop_length, n_mels=args.n_mels)
    test_data = my_dataloader.get_dataloader_for_test(labeled=(args.mode == 'labeled'))

    model_path = os.path.splitext(args.model_path)[0].replace('/', '.')
    tester = ModelTester(args.weight_path, args.latent_dim, device, model_path=model_path)

    tester.test_dataset(test_data, test_mode=args.mode, th=args.threshold, mapping=args.mapping)

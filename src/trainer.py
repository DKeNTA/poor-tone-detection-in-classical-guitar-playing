import importlib

import numpy as np
import torch
from barbar import Bar
from sklearn import metrics

from utils.early_stopping import EarlyStopping

class ModelTrainer:
    def __init__(self, args, device, model_path, logger):
        self.args = args
        self.device = device  
        self.logger = logger

        model = importlib.import_module(model_path)
        Encoder = getattr(model, "Encoder")
        self.encoder = Encoder

        self.eps = 1e-06 

    def train(self, train_dataloader, val_dataloader):
        net = self.encoder(self.args.latent_dim).to(self.device)

        early_stopping = EarlyStopping(monitor_metrics=['f1_max'], patience=self.args.patience, verbose=True, delta=self.args.es_delta, path=self.args.save_path, trace_func=self.logger.info)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                               milestones=self.args.lr_milestones, gamma=0.1)
        
        state_dict = torch.load(self.args.ae_save_path)
        net.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
        self.logger.info(f"Loaded weights of {self.args.ae_save_path}.")

        log = {
            'train_loss' : [],
            'validation_loss' : [],
            'PR-AUC' : [],
            'F1' : []
            }      

        for epoch in range(self.args.num_epochs):
            net.train()
            total_loss = 0
            self.logger.info(f"Training Encoder... Epoch: {epoch}")
            for x, semi_targets in Bar(train_dataloader):
                semi_targets = semi_targets.to(self.device)
                x = [x_i.float().to(self.device) for x_i in x]

                optimizer.zero_grad()

                z = net(*x[1:])
                
                dist = torch.sum((z - c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.args.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            train_loss = total_loss/len(train_dataloader)
            self.logger.info(f"Train Loss: {train_loss:.6f}")

            val_loss, auc, f1 = self.validation(net, c, val_dataloader)

            log['train_loss'].append(train_loss)
            log['validation_loss'].append(val_loss)
            log['PR-AUC'].append(auc)
            log['F1'].append(f1)

            early_stopping({'f1_max': f1}, net, log, epoch, center=c)

            if early_stopping.early_stop: 
                self.logger.info("Early stopping.")
                break

        self.logger.info("##### Training completed. #####")

    def validation(self, net, center, val_dataloader):
        net.eval()
        label_score = []
        val_loss = 0
        with torch.no_grad():
            for x, labels in Bar(val_dataloader):
                labels = labels.to(self.device)
                x = [x_i.float().to(self.device) for x_i in x]
                    
                z = net(*x[1:])
                
                dist = torch.sum((z - center) ** 2, dim=1)
                losses = torch.where(labels == 0, dist, self.args.eta * ((dist + self.eps) ** -labels.float()))
                loss = torch.mean(losses)

                scores = dist

                val_loss += loss.item()
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))
         
        loss = val_loss / len(val_dataloader)

        labels, scores = zip(*label_score)
        labels = [1 if label == -1 else 0 for label in labels] # norma: 1 → 0, anomaly: -1 → 1
        labels = np.array(labels)
        scores = np.array(scores)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
        auc = metrics.auc(recall, precision)
        F_measure = np.nan_to_num(2 * precision * recall / (precision + recall))

        F_max_id = np.argmax(F_measure)
        F_measure_max = np.max(F_measure)
        precision_F_max = precision[F_max_id]
        recall_F_max = recall[F_max_id]
        threshold_F_max = thresholds[F_max_id]

        self.logger.info(f"Validation Loss: {loss:.3f}")
        self.logger.info(f"F-measure : {F_measure_max:.4f}  Precision : {precision_F_max:.4f}  Recall : {recall_F_max:.4f}  Threshold : {threshold_F_max:.4f}")
        self.logger.info(f"PR-AUC : {100.*auc:.2f}%")

        return loss, auc, F_measure_max
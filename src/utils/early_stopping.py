import numpy as np
import torch

class EarlyStopping:
    def __init__(self, monitor_metrics=['val_loss'], patience=10, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        self.monitor_metrics = monitor_metrics
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = path
        self.trace_func = trace_func
        
        self.counter = 0
        self.early_stop = False
        self.best_scores = {monitor: np.Inf if monitor == 'val_loss' else -np.Inf for monitor in self.monitor_metrics}
        self.best_epoch = 0

    def __call__(self, scores, model, log, epoch, center=None):
        improved = False
        for monitor, score in scores.items():
            if self.monitor_criteria(monitor, score):
                if self.verbose:
                    self.trace_func(f"{monitor} improved ({self.best_scores[monitor]:.6f} --> {score:.6f}).")
                self.best_scores[monitor] = score
                improved = True

        if improved:
            self.save_checkpoint(model, log, center)
            self.best_epoch = epoch
            self.counter = 0
        else:      
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")     
            if self.counter >= self.patience:
                self.early_stop = True       

    def monitor_criteria(self, monitor, score):
        if monitor == 'val_loss':
            return score < self.best_scores[monitor] - self.delta
        else:
            return score > self.best_scores[monitor] + self.delta

    def save_checkpoint(self, model, log, center=None):
        if self.verbose:
            self.trace_func("Saving model ...")

        if center == None:
            torch.save({'net_dict': model.state_dict(),
                        'log': log}, self.save_path)
        else:
            torch.save({'center': center.cpu().data.numpy().tolist(),
                        'net_dict': model.state_dict(),
                        'log': log}, self.save_path)
        
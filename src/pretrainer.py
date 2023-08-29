import importlib

import torch
from barbar import Bar

from utils.early_stopping import EarlyStopping
from utils.utils import weights_init_he

class ModelPretrainer:
    def __init__(self, args, device, model_path, logger):
        self.args = args
        self.device = device
        self.logger = logger

        model = importlib.import_module(model_path)
        Autoencoder = getattr(model, "Autoencoder")
        Encoder = getattr(model, "Encoder")
        self.autoencoder = Autoencoder
        self.encoder = Encoder
    
    def train(self, train_dataloader, val_dataloader, setting_c_dataloader):
        ae = self.autoencoder(self.args.latent_dim).to(self.device)

        early_stopping = EarlyStopping(monitor_metrics=['val_loss'], patience=self.args.ae_patience, verbose=True, delta=self.args.ae_es_delta, path=self.args.ae_save_path, trace_func=self.logger.info)

        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.ae_lr,
                               weight_decay=self.args.ae_weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=self.args.ae_lr_milestones, gamma=0.1)

        ae.apply(lambda m: weights_init_he(m))
        log = {'pretrain_loss' : [],
               'validation_loss' : []}

        for epoch in range(self.args.ae_num_epochs):
            ae.train()
            total_loss = 0
            self.logger.info(f"Pretraining Autoencoder... Epoch: {epoch}")
            for x, _ in Bar(train_dataloader):
                x = [x_i.float().to(self.device) for x_i in x]

                optimizer.zero_grad()

                x_hat = ae(*x[1:])
                loss = torch.mean(torch.sum((x_hat - x[0]) ** 2, dim=tuple(range(1, x_hat.dim()))))

                loss.backward()           
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            pretrain_loss = total_loss / len(train_dataloader)
            
            self.logger.info(f"Pretrain Loss: {pretrain_loss:.6f}")

            val_loss = self.validation(ae, val_dataloader)

            log['pretrain_loss'].append(pretrain_loss)
            log['validation_loss'].append(val_loss)

            early_stopping({'val_loss': val_loss}, ae, log, epoch)

            if early_stopping.early_stop: 
                self.logger.info("Early stopping.")
                break
        
        self.logger.info("##### Pretraining completed. #####")
        state_dict = torch.load(self.args.ae_save_path)
        ae.load_state_dict(state_dict['net_dict'])
        self.save_weights(ae, setting_c_dataloader, log)

    def validation(self, ae, val_dataloader):
        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in Bar(val_dataloader):
                x = [x_i.float().to(self.device) for x_i in x]

                x_hat = ae(*x[1:])
                loss = torch.mean(torch.sum((x_hat - x[0]) ** 2, dim=tuple(range(1, x_hat.dim()))))

                val_loss += loss.item()
         
        loss = val_loss / len(val_dataloader)

        self.logger.info(f"Validation Loss: {loss:.6f}")

        return loss
    
    def save_weights(self, model, dataloader, log):
        net = self.encoder(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)

        c = self.set_c(net, dataloader)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict(),
                    'log': log}, self.args.ae_save_path)
        self.logger.info(f"pretrained_parameters saved to {self.args.ae_save_path}.")


    def set_c(self, model, dataloader, eps=0.1):
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in Bar(dataloader):
                x = [x_i.float().to(self.device) for x_i in x]

                z = model(*x[1:])

                z_.append(z.detach())
                
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.logger.info('##### Setting center done. #####')

        return c

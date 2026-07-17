import comet_ml
import gc
import torch
try:
    from pytorch_lightning.core.module import LightningModule
except:
    from lightning.pytorch.core.module import LightningModule
from torch.utils.data import DataLoader

from dataset import DDPDataset, DdpSampler, collate_fn
from models.Znet3 import Znet


class DDPLightning(LightningModule):

    def __init__(self, config, comet_exp=None):
        super().__init__()
        
        self.config = config
        self.comet_exp = comet_exp

        self.z_loss_fn = torch.nn.MSELoss(reduction='none')
        self.b_loss_fn = torch.nn.MSELoss(reduction='none')

        self.train_dataset = self.prep_train_dataset_and_create_nn()

        self.epsilon = 1e-8
        self.m = -9
        self.c = 1
        self.validation_step_outputs=[]

    def set_comet_exp(self, comet_exp):
        self.comet_exp = comet_exp


    def prep_train_dataset_and_create_nn(self):
        dataset = DDPDataset(self.config, self.config['train_dir'], mode='train')
        self.net = Znet(scale=dataset.scale)

        self.min_z = dataset.scale['true_z']['min']
        self.max_z = dataset.scale['true_z']['max']

        self.min_b = dataset.scale['background']['min']
        self.max_b = dataset.scale['background']['max']

        return dataset


    def train_dataloader(self):
        batch_sampler = DdpSampler(self.config, self.train_dataset.n_bins, batch_size=self.config['batch_size'], shuffle=True)
        loader = DataLoader(self.train_dataset, num_workers=self.config['num_workers'], 
            collate_fn=collate_fn, batch_sampler=batch_sampler, pin_memory=False)

        return loader

    
    def val_dataloader(self):
        dataset = DDPDataset(self.config, self.config['val_dir'], mode='val')
        batch_sampler = DdpSampler(self.config, dataset.n_bins, batch_size=self.config['batch_size'], shuffle=False)
        loader = DataLoader(dataset, num_workers=self.config['num_workers'], 
            collate_fn=collate_fn, batch_sampler=batch_sampler, pin_memory=False)

        return loader


    def forward(self, x):
        z_pred, b_pred = self.net(x)
        return z_pred, b_pred


    def training_step(self, batch, batch_idx):
        x, z, b = batch
        z_pred, b_pred = self.net(x)

        z_rescaled = (z - self.min_z) / (self.max_z - self.min_z)
        b_rescaled = (b - self.min_b) / (self.max_b - self.min_b)

        if self.config['b_loss_type'] == 'mse':
            z_pred_rescaled = (z_pred - self.min_z) / (self.max_z - self.min_z)
            b_pred_rescaled = (b_pred - self.min_b) / (self.max_b - self.min_b)

            z_loss = self.z_loss_fn(z_pred_rescaled, z_rescaled)
            b_loss = self.b_loss_fn(b_pred_rescaled, b_rescaled)
            
        elif self.config['b_loss_type'] == 'mse_err':
            z_loss = self.z_loss_fn(z_pred, z)
            b_err  = torch.sqrt(torch.maximum(b,torch.ones_like(b)))
            b_loss = self.b_loss_fn(b_pred/b_err, b/b_err)

        if self.config['rel_mse'] == True:
            z_loss = z_loss * (self.m * torch.abs(z_rescaled) + self.c)
            b_loss = b_loss * (self.m * torch.abs(b_rescaled) + self.c)

        z_loss = z_loss.mean()
        b_loss = b_loss.mean()

        loss = self.config['z_loss_wt'] * z_loss + self.config['b_loss_wt'] * b_loss

        self.log('train/loss', loss.item())
        self.log('train/loss_z', z_loss.item())
        self.log('train/loss_b', b_loss.item())

        return loss
        

    def validation_step(self, batch, batch_idx):
        x, z, b = batch
        z_pred, b_pred = self.net(x)

        z_rescaled = (z - self.min_z) / (self.max_z - self.min_z)
        b_rescaled = (b - self.min_b) / (self.max_b - self.min_b)

        z_pred_rescaled = (z_pred - self.min_z) / (self.max_z - self.min_z)
        b_pred_rescaled = (b_pred - self.min_b) / (self.max_b - self.min_b)

        z_loss = self.z_loss_fn(z_pred_rescaled, z_rescaled)

        if self.config['b_loss_type'] == 'mse':
            b_loss = self.b_loss_fn(b_pred_rescaled, b_rescaled)
        elif self.config['b_loss_type'] == 'mse_err':
            z_loss = self.z_loss_fn(z_pred, z)
            b_err  = torch.sqrt(torch.maximum(b,torch.ones_like(b)))
            b_loss = self.b_loss_fn(b_pred/b_err, b/b_err)

        if self.config['rel_mse'] == True:
            z_loss = z_loss * (self.m * torch.abs(z_rescaled) + self.c)
            b_loss = b_loss * (self.m * torch.abs(b_rescaled) + self.c)

        z_loss = z_loss.mean()
        b_loss = b_loss.mean()

        loss = self.config['z_loss_wt'] * z_loss + self.config['b_loss_wt'] * b_loss

        self.log('val/loss', loss.item())
        self.log('val/loss_z', z_loss.item())
        self.log('val/loss_b', b_loss.item())
        
        self.validation_step_outputs.append((loss, z_loss, b_loss))

        return loss, z_loss, b_loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['learning_rate'])

        if self.config['lr_scheduler'] == None:
            return optimizer
        
        elif list(self.config['lr_scheduler'].keys())[0] == 'CosineAnnealingLR':
            Tmax = self.config['lr_scheduler']['CosineAnnealingLR']['T_max']
            Tmax = self.config['num_epochs'] if Tmax == -1 else Tmax
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = Tmax,
                eta_min = self.config['lr_scheduler']['CosineAnnealingLR']['eta_min'],
                last_epoch = self.config['lr_scheduler']['CosineAnnealingLR']['last_epoch'],
                verbose = self.config['lr_scheduler']['CosineAnnealingLR']['verbose'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def on_validation_epoch_end(self):
        outputs=self.validation_step_outputs
        
        avg_loss = torch.FloatTensor([x[0] for x in outputs]).mean()
        self.log('val/avg_loss', avg_loss.item())

        avg_loss_z = torch.FloatTensor([x[1] for x in outputs]).mean()
        self.log('val/avg_loss_z', avg_loss_z.item())

        avg_loss_b = torch.FloatTensor([x[2] for x in outputs]).mean()
        self.log('val/avg_loss_b', avg_loss_b.item())

        self.log('lr', self.lr_schedulers().get_last_lr()[0])
        
        #Free memory 
        self.validation_step_outputs.clear()
        gc.collect()

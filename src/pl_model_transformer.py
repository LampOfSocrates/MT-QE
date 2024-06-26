import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor , DeviceStatsMonitor

from callback_gpu import GPUMonitorCallback
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from common import ROOT_FOLDER

class TranslationQualityModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TranslationQualityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        '''
        # Define the example input array. Carefully shaped
        self.example_input_array = (
            torch.zeros(1, input_dim),  # src_embedding
            torch.zeros(1, input_dim),  # mt_embedding
            torch.zeros(1, input_dim)   # ref_embedding
        )
        '''
        

    def forward(self, src_embedding, mt_embedding, ref_embedding):
        x = torch.cat((src_embedding, mt_embedding, ref_embedding), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(dim=1)  # Ensure output shape is [batch_size]

    def training_step(self, batch, batch_idx):

        wandb_logger = self.logger.experiment
        #print(type(wandb_logger)) # TODO use this logger to pump in some data to wanbd
        src_embedding = batch['src_embedding']
        mt_embedding = batch['mt_embedding']
        ref_embedding = batch['ref_embedding']
        score = batch['score']

        output = self(src_embedding, mt_embedding, ref_embedding)
        loss = self.criterion(output, score)
        self.log('train_loss', loss, batch_size=len(batch['score']),  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_embedding = batch['src_embedding']
        mt_embedding = batch['mt_embedding']
        ref_embedding = batch['ref_embedding']
        score = batch['score']

        output = self(src_embedding, mt_embedding, ref_embedding)
        loss = self.criterion(output, score)
        mae = self.mae(output, score)
        mse = self.mse(output, score)
        r2 = self.r2(output, score)

        dct = {"val_loss": loss, "val_mae": mae, "val_mse": mse, "val_r2": r2 }
        self.log_dict(dct, batch_size=len(batch['score']), on_step=False, on_epoch=True, prog_bar=True, logger=True)


        return {"loss": loss, "mae": mae, "mse": mse, "r2": r2}

    def test_step(self, batch, batch_idx):
        src_embedding = batch['src_embedding']
        mt_embedding = batch['mt_embedding']
        ref_embedding = batch['ref_embedding']
        score = batch['score']

        output = self(src_embedding, mt_embedding, ref_embedding)
        loss = self.criterion(output, score)
        mae = self.mae(output, score)
        mse = self.mse(output, score)
        r2 = self.r2(output, score)
        
        
        dct = {"test_loss": loss, "test_mae": mae, "test_mse": mse, "test_r2": r2 }
        self.log_dict(dct, batch_size=len(batch['score']), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss, "mae": mae, "mse": mse, "r2": r2}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=True,
            mode='min'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f"{ROOT_FOLDER}/model1/checkpoints", # directory where checkpoints are saved
            filename="model1-{epoch}-{val_loss:.2f}", # filename pattern
            save_top_k=3,
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        device_stats = DeviceStatsMonitor()

        callbacks = [early_stopping, checkpoint_callback,  lr_monitor]
        if torch.cuda.is_available():
            callbacks.append(GPUMonitorCallback())

        return callbacks
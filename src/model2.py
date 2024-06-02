import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor

class TranslationQualityModel(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128, output_dim=1, learning_rate=0.001):
        super(TranslationQualityModel, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.backbone = BertModel.from_pretrained(model_name)

        # Regression head
        self.fc1 = nn.Linear(self.backbone.config.hidden_size * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Loss and metrics
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()

    def forward(self, src, mt, ref):
        # Tokenize input sequences
        src_tokens = self.tokenizer(src, return_tensors='pt', padding=True, truncation=True).to(self.device)
        mt_tokens = self.tokenizer(mt, return_tensors='pt', padding=True, truncation=True).to(self.device)
        ref_tokens = self.tokenizer(ref, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Get embeddings from BERT
        src_embedding = self.backbone(**src_tokens).last_hidden_state[:, 0, :]
        mt_embedding = self.backbone(**mt_tokens).last_hidden_state[:, 0, :]
        ref_embedding = self.backbone(**ref_tokens).last_hidden_state[:, 0, :]

        # Concatenate embeddings and pass through regression head
        x = torch.cat((src_embedding, mt_embedding, ref_embedding), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(dim=1)

    def training_step(self, batch, batch_idx):
        
        src, mt, ref, score = batch['src'], batch['mt'], batch['ref'], batch['score']
        try:
            output = self(src, mt, ref)
        except Exception as e:
            print("Model encountered exception, lets print what it faced")
            print(batch)
            print(batch_idx)
            print("mt", mt)
            return None

        
        loss = self.criterion(output, score)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, mt, ref, score = batch['src'], batch['mt'], batch['ref'], batch['score']
        output = self(src, mt, ref)
        loss = self.criterion(output, score)
        mae = self.mae(output, score)
        mse = self.mse(output, score)
        r2 = self.r2(output, score)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2', r2, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_mae": mae, "val_mse": mse, "val_r2": r2}

    def test_step(self, batch, batch_idx):
        src, mt, ref, score = batch['src'], batch['mt'], batch['ref'], batch['score']
        output = self(src, mt, ref)
        loss = self.criterion(output, score)
        mae = self.mae(output, score)
        mse = self.mse(output, score)
        r2 = self.r2(output, score)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae', mae, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_r2', r2, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_mae": mae, "test_mse": mse, "test_r2": r2}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=True,
            mode='min'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        device_stats = DeviceStatsMonitor()

        callbacks = [early_stopping, checkpoint_callback, lr_monitor]
        return callbacks

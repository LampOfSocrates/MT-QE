import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv

class GCN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, lr=0.01):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lr = lr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.compute_loss(out, batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.compute_loss(out, batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_loss(self, out, data):
        # Assuming self-supervised learning; define your loss function here
        # This is a placeholder and needs to be adapted for your specific task
        loss = F.mse_loss(out, torch.zeros_like(out))  # Example: MSE loss with dummy target
        return loss

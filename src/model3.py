import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Union
from lru_cache import tensor_lru_cache
from pooling_utils import average_pooling, max_pooling
from feedforward import FeedForward
CACHE_SIZE = 1024
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data import random_split
import logging
from common import ROOT_FOLDER , save_errdata_to_file
logger = logging.getLogger(__name__)
from dat_wmt import TranslationDataset
BATCH_SIZE=8

class TranslationQualityModel(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128, output_dim=1, learning_rate=0.001):
        super(TranslationQualityModel, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Regression head. We are passing src/mt/ref and so input id 768 x 3 .where 768 represents the sentence embedding that is getting learned.
        #self.fc1 = nn.Linear(self.backbone.config.hidden_size * 3, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, output_dim)

        print("self.backbone.config.hidden_size * 3,", self.backbone.config.hidden_size * 3,)
        self.estimator = FeedForward(
            in_dim=self.backbone.config.hidden_size * 2,
            #hidden_sizes=self.hidden_dim,
            activations="Tanh",
            dropout=0.1,
            final_activation=None,
            out_dim=1,
        )
        # Loss and metrics
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.r2 = R2Score()
        self.learning_rate = 1e-5
        self.load_data()
        self.caching=False
    
    def load_data(self):
        file_path = 'data/2022-da.csv'
        dataset = TranslationDataset(file_path, max_words_in_sentence=50, lp="cs-en")

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        print("train size: ", train_size)
        print("val size: ", val_size)
        print("test size: ", test_size)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])



    def forward(self, 
                src_input_ids: torch.tensor,
                src_attention_mask: torch.tensor,
                mt_input_ids: torch.tensor,
                mt_attention_mask: torch.tensor,
                ref_input_ids: torch.tensor,
                ref_attention_mask: torch.tensor,):
        ''' Here we just concatenate the embeddings'''

        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        ref_sentemb = self.get_sentence_embedding(ref_input_ids, ref_attention_mask)
        mt_sentemb = self.get_sentence_embedding(mt_input_ids, mt_attention_mask)
        return self.estimate(src_sentemb, mt_sentemb, ref_sentemb)

    def estimate(
        self,
        src_sentemb: torch.Tensor,
        mt_sentemb: torch.Tensor,
        ref_sentemb: torch.Tensor,
    ) :
        """Method that takes the sentence embeddings from the Encoder and runs the
        Estimator Feed-Forward on top.

        Args:
            src_sentemb [torch.Tensor]: Source sentence embedding
            mt_sentemb [torch.Tensor]: Translation sentence embedding
            ref_sentemb [torch.Tensor]: Reference sentence embedding

        Return:
            Prediction object with sentence scores.
        """
        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        #diff_src = torch.abs(mt_sentemb - src_sentemb)

        #prod_ref = mt_sentemb * ref_sentemb
        #prod_src = mt_sentemb * src_sentemb

        #embedded_sequences = torch.cat(
        #    (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
        #    dim=1,
        #)
        embedded_sequences = torch.cat(
            ( ref_sentemb, diff_ref),
            dim=1,
        )
        
        return self.estimator(embedded_sequences).view(-1)

    def set_embedding_cache(self):
        """Function that when called turns embedding caching on."""
        self.caching = True

    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Function that extracts sentence embeddings for
        a single sentence and allows for caching embeddings.

        Args:
            tokens (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            token_type_ids (torch.Tensor): Model token_type_ids [batch_size x seq_len].
                Optional

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        if self.caching:
            return self.retrieve_sentence_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            return self.compute_sentence_embedding(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )

    @tensor_lru_cache(maxsize=CACHE_SIZE)
    def retrieve_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper for `get_sentence_embedding` function that caches results."""
        return self.compute_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def compute_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Function that extracts sentence embeddings for
        a single sentence.

        Args:
            tokens (torch.Tensor): sequences [batch_size x seq_len].
            attention_mask (torch.Tensor): attention_mask [batch_size x seq_len].
            token_type_ids (torch.Tensor): Model token_type_ids [batch_size x seq_len].
                Optional

        Returns:
            torch.Tensor [batch_size x hidden_size] with sentence embeddings.
        """
        last_hidden_states, _, all_layers = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )

        encoder_out = {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
        
        sentemb = encoder_out["sentemb"]

        return sentemb

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.

        Returns:
            [torch.Tensor] Loss value
        """
        
        batch_input, batch_target = batch

        
        
        try:
            batch_prediction = self.forward(**batch_input)

        except Exception as e:
            print("Model encountered exception while training, lets print what it faced")
            save_errdata_to_file(batch, './data')
            print(batch_idx)
            return None
        
        loss = self.criterion(batch_prediction, batch_target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        
        loss = self.criterion(batch_prediction, batch_target)
        mae = self.mae(batch_prediction, batch_target)
        mse = self.mse(batch_prediction, batch_target)
        r2 = self.r2(batch_prediction, batch_target)
        
        self.log_dict({'val_loss': loss , 'val_mae' : mae , 'val_mse': mse, 'val_r2': r2}, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_mae": mae, "val_mse": mse, "val_r2": r2}

    def test_step(self, batch, batch_idx):
        batch_input, batch_target = batch
        batch_prediction = self.forward(**batch_input)
        print(batch_prediction.shape)
        print(batch_target.shape)
        
        loss = self.criterion(batch_prediction, batch_target)
        mae = self.mae(batch_prediction, batch_target)
        mse = self.mse(batch_prediction, batch_target)
        #r2 = self.r2(batch_prediction, batch_target)
        self.log_dict({'test_loss': loss , 'test_mae' : mae , 'test_mse': mse}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": loss, "test_mae": mae, "test_mse": mse}

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
            dirpath=f"{ROOT_FOLDER}/model2/checkpoints", # directory where checkpoints are saved
            filename="model2-{epoch}-{val_loss:.2f}", # filename pattern
            save_top_k=3,
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        device_stats = DeviceStatsMonitor()

        callbacks = [early_stopping,  checkpoint_callback, lr_monitor]
        return callbacks

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and depending on the 'stage' training labels/targets.
        """
        inputs = {k: [str(dic[k]) for dic in sample] for k in sample[0] if k != "score"}
        src_inputs = self.tokenize_sample(inputs["src"])
        mt_inputs = self.tokenize_sample(inputs["mt"])
        ref_inputs = self.tokenize_sample(inputs["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}
        model_inputs = {**src_inputs, **mt_inputs, **ref_inputs}

        if stage == "predict":
            return model_inputs

        scores = [float(s["score"]) for s in sample]
        targets = torch.tensor(scores, dtype=torch.float)

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets

    def tokenize_sample(self, sample: List[str]):
        tokenizer_output = self.tokenizer(
                sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50 - 2,
            )
        return tokenizer_output

    def train_dataloader(self) -> DataLoader:
        """Method that loads the train dataloader. Can be called every epoch to load a
        different trainset if `reload_dataloaders_every_n_epochs=1` in Lightning
        Trainer.
        """
        
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=BATCH_SIZE,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

    def val_dataloader(self) -> DataLoader:
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

    def test_dataloader(self) -> DataLoader:
        
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=lambda s: self.prepare_sample(s, stage="fit"),
            num_workers=2 * self.trainer.num_devices,
        )

from typing import Optional, Dict, Literal
import numpy as np

import torch
import torch.nn as nn  
from dataclasses import dataclass, fields

from data_utils import NERDataset_lstm_cnn 
from helpers import lstm_cnn_config_data 

feature_extractor_types_ = Literal["lstm", "cnn"]

# NOTE: should be called before the model initialization
def prepare_model_config(ner_dataset: Optional[NERDataset_lstm_cnn]=None, **kwargs):
    lstm_cnn_model_config = {
        "vocab_size": None,
        "num_labels": 1,
        "input_dim": None,
        "hidden_dim": None,
        "n_layers": 1,
        "dropout": 0.0,
        "feature_extractor": None,
        "pretrained_embeds": None
    }

    # update feature_extractor
    lstm_cnn_model_config["feature_extractor"] = kwargs["feature_extractor"] if "feature_extractor" in kwargs else lstm_cnn_config_data["FEATURE_EXTRACTOR"]
    # update dropout
    lstm_cnn_model_config["dropout"] = kwargs["dropout"] if "dropout" in kwargs else lstm_cnn_config_data["DROPOUT"]
    # update input_dim
    lstm_cnn_model_config["input_dim"] = kwargs["input_dim"] if "input_dim" in kwargs else lstm_cnn_config_data["INPUT_DIM"]
    # update hidden_dim
    lstm_cnn_model_config["hidden_dim"] = kwargs["hidden_dim"] if "hidden_dim" in kwargs else lstm_cnn_config_data["HIDDEN_DIM"]
    # update n_layers
    if lstm_cnn_config_data["FEATURE_EXTRACTOR"] == "lstm":
        lstm_cnn_model_config["n_layers"] = kwargs["n_layers"] if "n_layers" in kwargs else lstm_cnn_config_data["N_LAYERS"]

    if ner_dataset is not None:
        # update vocab_size
        lstm_cnn_model_config["vocab_size"] = len(ner_dataset.vocab)
        # update num_labels
        lstm_cnn_model_config["num_labels"] = len(ner_dataset.dataset.labels)
        # update pretrain_embeds
        if lstm_cnn_config_data["USE_GLOVE"]:
            lstm_cnn_model_config["pretrained_embeds"] = ner_dataset.glove_embeds

    return lstm_cnn_model_config

@dataclass
class LSTMCNNTokenClassifierOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None

    def __getitem__(self, key):
        for field in fields(self):
            if field.name == key:
                return getattr(self, key)
        raise KeyError(f"{key} is not a valid attribute")

class LSTMCNNForTokenClassification(nn.Module):
    def __init__(self, config: Dict=None,
                 num_labels: int=None,
                 feature_extractor: Optional[feature_extractor_types_]="lstm",
                 vocab_size: Optional[int]=None,
                 input_dim: Optional[int]=None,
                 hidden_dim: Optional[int]=None,
                 n_layers: Optional[int]=None,
                 dropout: Optional[float]=None,
                 pretrained_embeds: Optional[np.array]=None, **kwargs):
        super(LSTMCNNForTokenClassification, self).__init__()

        if num_labels is not None and config.get("num_labels") is not None:
            raise ValueError("Required to mention the num_labels parameter")

        self.num_labels = config["num_labels"] if not num_labels else num_labels
        self.vocab_size = config["vocab_size"] if not vocab_size else vocab_size
        self.input_dim = config["input_dim"] if not input_dim else input_dim
        self.hidden_dim = config["hidden_dim"] if not hidden_dim else hidden_dim
        dropout = config["dropout"] if not dropout else dropout
        self.dropout = nn.Dropout(dropout)

        if self.hidden_dim == None:
            raise ValueError("Required to mention the `hidden_dim` parameter OR `HIDDEN_DIM` config-value")

        self.embeds = nn.Embedding(self.vocab_size, self.input_dim, padding_idx=0)
        pretrained_embeds =  config["pretrained_embeds"] if not pretrained_embeds else pretrained_embeds
        if pretrained_embeds is not None:
            nn.init.normal_(self.embeds.weight.data[0], mean=0, std=0.1)  # Random initialization for padding token
            nn.init.normal_(self.embeds.weight.data[1], mean=0, std=0.1)  # Random initialization for unknown token
            self.embeds.weight.data[2:].copy_(torch.from_numpy(pretrained_embeds))
        else:
            self.embeds.weight.data.copy_(torch.from_numpy(self.random_embedding(self.vocab_size, self.input_dim)))

        self.feature_extractor = feature_extractor
        if self.feature_extractor == "lstm":
            self.n_layers = config["n_layers"] if not n_layers else n_layers 
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        elif self.feature_extractor == "cnn":
            self.word2cnn = nn.Linear(self.input_dim, self.hidden_dim*2)
            self.cnn_list = list()
            for _ in range(4):
                self.cnn_list.append(nn.Conv1d(self.hidden_dim*2, self.hidden_dim*2, kernel_size=3, padding=1))
                self.cnn_list.append(nn.ReLU())
                self.cnn_list.append(nn.Dropout(dropout))
                self.cnn_list.append(nn.BatchNorm1d(self.hidden_dim*2))
            self.cnn = nn.Sequential(*self.cnn_list)
        else:
            raise ValueError(f"Invalid feature-extractor provided, expected feature-extractor-types are {feature_extractor_types_}")

        self.classifier = nn.Linear(self.hidden_dim*2, self.num_labels)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_embeds = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(1, vocab_size):
            pretrain_embeds[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_embeds

    def forward(self,
                word_ids: Optional[torch.tensor]=None,
                sequence_length: Optional[torch.tensor]=None,
                labels: Optional[torch.Tensor]=None,
                mask: Optional[torch.Tensor]=None):
        batch_size, seq_len = word_ids.size()
        sequence_embedding = torch.cat([self.embeds(word_ids)], 2)
        sequence_embedding = self.dropout(sequence_embedding)
        if self.feature_extractor == "lstm":
            hidden=None
            lstm_out, hidden = self.lstm(sequence_embedding, hidden)
            logits = self.dropout(lstm_out)
        else:
            word_in = torch.tanh(self.word2cnn(sequence_embedding)).transpose(2, 1).contiguous()
            logits = self.cnn(word_in).transpose(1, 2).contiguous()

        logits = self.classifier(logits)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.feature_extractor == "lstm":
            return LSTMCNNTokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=lstm_out
            )
        else:
            return LSTMCNNTokenClassifierOutput(
                loss=loss,
                logits=logits
            )
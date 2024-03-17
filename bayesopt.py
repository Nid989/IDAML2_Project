import gc
import time
import logging
from typing import Optional

import torch.utils.checkpoint
from transformers import (
    AutoTokenizer
)

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper

from data_utils import NERDataset_transformers, NERDataset_lstm_cnn
from helpers import general_config_data, transformers_config_data, lstm_cnn_config_data
from helpers import save_skopt_results
from models.modeling_xlm_roberta import XLMRobertaForTokenClassification
from models.modeling_lstm_cnn import prepare_model_config, LSTMCNNForTokenClassification
from model_utils import Transformer_Trainer, LSTM_CNN_Trainer

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('logfile_bayesopt.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# hyperparameter space (Λ)
# ---------------------- # Hyperparamter space for transformers # ---------------------- # 
thp_space = [Integer(10, 100, name='max_sequence_len'),
             Integer(1, 16, name='batch_size'),
             Integer(1, 10, name='num_epochs'),
             Real(1e-6, 1e-2, name='learning_rate', prior='log-uniform'),
             Real(1e-6, 1e-2, name='weight_decay', prior='log-uniform')]

# ---------------------- # Hyperparameter space for LSTM (`no CNN`) # ---------------------- # 
lchp_space = [Integer(10, 100, name='max_sequence_len'),
              Integer(1, 16, name='batch_size'),
              Integer(1, 10, name='num_epochs'),   
              Categorical(categories=[50, 100, 200, 300], name='input_dim'),
              Categorical(categories=[64, 128, 256, 512, 768], name='hidden_dim'),
              Integer(1, 5, name="n_layers"),
              Real(0.0, 0.5, name='dropout'),
              Real(1e-6, 1e-2, name='learning_rate', prior='log-uniform'),
              Real(1e-6, 1e-2, name='weight_decay', prior='log-uniform')]

@use_named_args(thp_space)
def t_objective_fxn(max_sequence_len: int, batch_size: int, num_epochs: int,
                    learning_rate: float, weight_decay: float) -> Optional[float]:
    start_time = time.time()

    logging.info("Starting (transformers) objective function execution.")
    logging.info("Received attributes: batch_size={}, num_epochs={}, max_sequence_len={}, learning_rate={}, weight_decay={}".format(batch_size, num_epochs, max_sequence_len, learning_rate, weight_decay))

    logging.info("Initializing Tokenizer: AutoTokenizer.from_pretrained('xlm-roberta-base')")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    logging.info("Tokenizer initialized successfully.")

    logging.info("Preparing NER Dataset: NERDataset_transformers(tokenizer=tokenizer, config=config_data, max_sequence_len=max_sequence_len)")
    ner_dataset = NERDataset_transformers(tokenizer=tokenizer, config=transformers_config_data, max_sequence_len=max_sequence_len)
    ner_dataset.on_setup()
    logging.info("NER Dataset prepared successfully.")

    logging.info("Initializing Model: XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(ner_dataset.dataset.labels))")
    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(ner_dataset.dataset.labels))
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("Model initialized successfully.")

    logging.info("Training...")
    trainer = Transformer_Trainer(model=model,
                                  dataset=ner_dataset,
                                  batch_size=batch_size,
                                  num_epochs=num_epochs,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay)
    trainer.train(save=False)
    logging.info("Training completed.")

    logging.info("Evaluation...")
    results = trainer.get_valid_scores(dataloader=trainer.valid_dataloader,
                                       desc="Empirical Risk Calculation on Validation Data")
    logging.info("Measured `f1` score: {}".format(results["f1"]))
    logging.info("Evaluation completed.")

    del tokenizer
    del ner_dataset
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Objective function execution completed in {:.2f} seconds.".format(execution_time))

    return -results["f1"]

@use_named_args(lchp_space)
def lc_objective_fxn(max_sequence_len: int, batch_size: int, num_epochs: int,
                     input_dim: int, hidden_dim: int, n_layers: int, dropout: float,
                     learning_rate: float, weight_decay: float) -> Optional[float]:
    start_time = time.time()

    logging.info("Starting (lstm-cnn) objective function execution.")
    logging.info("Received attributes: batch_size={}, num_epochs={}, max_sequence_len={}, input_dim: {}, hidden_dim: {}, n_layer: {}, dropout: {}, learning_rate={}, weight_decay={}".format(batch_size, num_epochs, max_sequence_len, input_dim, hidden_dim, n_layers, dropout, learning_rate, weight_decay))

    logging.info("Preparing NER Dataset: NERDataset_lstm_cnn(config=config_data, glove_dim=input_dim)")
    ner_dataset = NERDataset_lstm_cnn(config=lstm_cnn_config_data, glove_dim=input_dim)
    ner_dataset.on_setup()
    logging.info("NER Dataset prepared successfully.")

    logging.info("Preparing model configuration dict: prepare_model_config(**kwargs)")
    lstm_cnn_model_config = prepare_model_config(ner_dataset=ner_dataset,
                                                 feature_extractor="lstm",
                                                 dropout=dropout,
                                                 input_dim=input_dim,
                                                 hidden_dim=hidden_dim,
                                                 n_layers=n_layers)
    logging.info("Model configuration dict prepared successfully.")

    logging.info("Initializing Model: LSTMCNNForTokenClassification(config=lstm_cnn_model_config)")
    model = LSTMCNNForTokenClassification(config=lstm_cnn_model_config)
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("Model initialized successfully.")

    logging.info("Training...")
    trainer = LSTM_CNN_Trainer(model=model,
                               dataset=ner_dataset,
                               batch_size=batch_size,
                               num_epochs=num_epochs,
                               learning_rate=learning_rate,
                               weight_decay=weight_decay)
    trainer.train(save=False)
    logging.info("Training completed.")

    logging.info("Evaluation...")
    results = trainer.get_valid_scores(dataloader=trainer.valid_dataloader,
                                       desc="Empirical Risk Calculation on Validation Data")
    logging.info("Measured `f1` score: {}".format(results["f1"]))
    logging.info("Evaluation completed.")

    del ner_dataset
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Objective function execution completed in {:.2f} seconds.".format(execution_time))

    return -results["f1"]


if __name__ == "__main__":
    modeling_type = general_config_data["MODELING_TYPE"]
    if modeling_type == "transformers":
        logging.info("Start: scikit-optimize `Gaussian-Process` optimization procedure.")
        res_gp = gp_minimize(t_objective_fxn, thp_space, n_calls=15, random_state=42, callback=[DeltaXStopper(0.001)])
        logging.info("End: scikit-optimize `Gaussian-Process` optimization procedure.")

        best_hyperparams = res_gp.x
        # logging.info("Best Hyperparameters: ", best_hyperparams)
        print(best_hyperparams)

        save_skopt_results(res_gp, logging=logging)

    elif modeling_type == "lstm_cnn":
        logging.info("Start: scikit-optimize `Gaussian-Process` optimization procedure.")
        res_gp = gp_minimize(lc_objective_fxn, lchp_space, n_calls=15, random_state=42, callback=[DeltaXStopper(0.001)])
        logging.info("End: scikit-optimize `Gaussian-Process` optimization procedure.")

        best_hyperparams = res_gp.x
        # logging.info("Best Hyperparameters: ", best_hyperparams)
        print(best_hyperparams)

        save_skopt_results(res_gp, logging=logging)
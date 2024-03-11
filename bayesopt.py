import gc
import time
import logging
from typing import Optional

import torch.utils.checkpoint
from transformers import (
    AutoTokenizer
)

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper

from data_utils import NERDataset_transformers
from helpers import config_data
from helpers import save_skopt_results
from models.modeling_xlm_roberta import XLMRobertaForTokenClassification
from model_utils import Trainer

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('logfile_bayesopt.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# hyperparameter space (Λ)
space = [Integer(150, 250, name='max_sequence_len'),
         Integer(1, 4, name='batch_size'),
         Integer(1, 3, name='num_epochs'),
         Real(1e-6, 1e-2, name='learning_rate', prior='log-uniform'),
         Real(1e-6, 1e-2, name='weight_decay', prior='log-uniform')]

@use_named_args(space)
def objective_function(max_sequence_len: int, batch_size: int, num_epochs: int,
                       learning_rate: float, weight_decay: float) -> Optional[float]:
    start_time = time.time()

    logging.info("Starting objective function execution.")
    logging.info("Received attributes: learning_rate={}, weight_decay={}, batch_size={}, num_epochs={}, max_sequence_len={}".format(learning_rate, weight_decay, batch_size, num_epochs, max_sequence_len))

    logging.info("Initializing Tokenizer: AutoTokenizer.from_pretrained('xlm-roberta-base')")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    logging.info("Tokenizer initialized successfully.")

    logging.info("Preparing NER Dataset: NERDataset_transformers(tokenizer=tokenizer, config=config_data, max_sequence_len=max_sequence_len)")
    ner_dataset = NERDataset_transformers(tokenizer=tokenizer, config=config_data, max_sequence_len=max_sequence_len)
    ner_dataset.on_setup()
    logging.info("NER Dataset prepared successfully.")

    logging.info("Initializing Model: XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=len(ner_dataset.dataset.labels))")
    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(ner_dataset.dataset.labels))
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logging.info("Model initialized successfully.")

    logging.info("Training...")
    trainer = Trainer(model=model,
                      dataset=ner_dataset,
                      batch_size=batch_size,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate,
                      weight_decay=weight_decay)
    trainer.train()
    logging.info("Training completed.")

    logging.info("Evaluation...")
    results = trainer.get_valid_scores(dataloader=trainer.valid_dataloader,
                                       desc="Empirical Risk Calculation on Validation Data")
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

if __name__ == "__main__":
    logging.info("Start: scikit-optimize `Gaussian-Process` optimization procedure.")
    res_gp = gp_minimize(objective_function, space, n_calls=20, random_state=42, callback=[DeltaXStopper(0.001)])
    logging.info("End: scikit-optimize `Gaussian-Process` optimization procedure.")

    best_hyperparams = res_gp.x
    print(best_hyperparams)

    save_skopt_results(res_gp, logging=logging)
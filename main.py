import logging

import torch
from transformers import AutoTokenizer

from data_utils import NERDataset_LSTM_CNN, NERDataset_transformers
from models.modeling_xlm_roberta import XLMRobertaForTokenClassification, XLMRobertaCRFForTokenClassification
from models.modeling_lstm_cnn import LSTMCNNForTokenClassification, prepare_model_config
from helpers import general_config_data, lstm_cnn_config_data, transformers_config_data, print_model_details
from model_utils import Transformer_Trainer, LSTM_CNN_Trainer

# Set up logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('logfile_main.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    modeling_type = general_config_data["MODELING_TYPE"]
    logger.info(f"Modeling type: {modeling_type}")

    if modeling_type == "transformers":

        model_checkpoint = transformers_config_data["MODEL_CHECKPOINT"]
        logger.info(f"Model checkpoint: {model_checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                  add_prefix_space=True)  # specific to XLM-RoBERTa
        logger.info("Tokenizer loaded")

        ner_dataset = NERDataset_transformers(tokenizer=tokenizer, config=transformers_config_data)
        logger.info("NER Dataset created")

        if transformers_config_data["MODEL_IMPLEMENTATION"] == "tokenclassification":
            ner_dataset.on_setup(use_crf=False)
            logger.info("Using TokenClassification model")
            model = XLMRobertaForTokenClassification.from_pretrained(model_checkpoint,
                                                                     num_labels=len(ner_dataset.dataset.labels))
            model.to(device)
            print_model_details(model)

        elif transformers_config_data["MODEL_IMPLEMENTATION"] == "tokenclassification w/ LC-CRF":
            ner_dataset.on_setup(use_crf=True)
            logger.info("Using TokenClassification w/ LC-CRF model")
            model = XLMRobertaCRFForTokenClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=len(ner_dataset.dataset.labels))
            model.to(device)
            print_model_details(model)

        else:
            raise ValueError("Invalid MODEL_IMPLEMENTATION value. Please check the transformers_config.yaml file.")

        logger.info("Training...")
        trainer = Transformer_Trainer(model=model,
                                      dataset=ner_dataset,
                                      use_crf=ner_dataset.use_crf)
        logger.info("Trainer initialized")
        trainer.train(use_patience=False)
        logger.info("Training completed")

        logging.info("Evaluation...")
        results = trainer.get_valid_scores(dataloader=trainer.valid_dataloader,
                                           desc="Empirical Risk Calculation on Validation Data")
        logging.info("Measured `f1` score: {}, `accuracy` score: {}".format(results["f1"], results["accuracy"]))
        logging.info("Evaluation completed.")

    elif modeling_type == "lstm_cnn":
        feature_extractor = lstm_cnn_config_data["FEATURE_EXTRACTOR"]
        logger.info(f"Feature extractor utilized: {feature_extractor}")

        ner_dataset = NERDataset_LSTM_CNN(config=lstm_cnn_config_data,
                                          glove_dim=lstm_cnn_config_data["INPUT_DIM"])
        ner_dataset.on_setup()
        logger.info("NER Dataset created")

        lstm_cnn_model_config = prepare_model_config(ner_dataset=ner_dataset)

        logger.info("Using TokenClassification model")
        model = LSTMCNNForTokenClassification(config=lstm_cnn_model_config)
        model.to(device)
        print_model_details(model)
        print(model)

        logger.info("Training...")
        trainer = LSTM_CNN_Trainer(model=model,
                                   dataset=ner_dataset)
        
        logger.info("Trainer initialized")
        trainer.train(use_patience=False)
        logger.info("Training completed")

        logging.info("Evaluation...")
        results = trainer.get_valid_scores(dataloader=trainer.valid_dataloader,
                                           desc="Empirical Risk Calculation on Validation Data")
        logging.info("Measured `f1` score: {}, `accuracy` score: {}".format(results["f1"], results["accuracy"]))
        logging.info("Evaluation completed.")
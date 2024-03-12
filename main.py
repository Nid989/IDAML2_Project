import torch
from transformers import AutoTokenizer

from data_utils import NERDataset_transformers
from models.modeling_xlm_roberta import XLMRobertaForTokenClassification, XLMRobertaCRFForTokenClassification
from helpers import general_config_data, transformers_config_data, print_model_details
from model_utils import Trainer

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    modeling_type = general_config_data["MODELING_TYPE"]
    if modeling_type == "transformers":

        model_checkpoint = transformers_config_data["MODEL_CHECKPOINT"]

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                  add_prefix_space=True) # specific to XLM-RoBERTa
        ner_dataset = NERDataset_transformers(tokenizer=tokenizer, config=transformers_config_data)

        # ----------------- # INFO # ----------------- #
        # under transformers we have 3 different model implementation
        # this includes, i) TokenClassification (NER), ii) TokenClassification w/ Linear Chain CRF
        # -------------------------------------------- #

        if transformers_config_data["MODEL_IMPLEMENTATION"] == "tokenclassification":
            ner_dataset.on_setup(use_crf=False)
            model = XLMRobertaForTokenClassification.from_pretrained(model_checkpoint,
                                                                     num_labels=len(ner_dataset.dataset.labels))
            model.to(device)  
            print_model_details(model)

        elif transformers_config_data["MODEL_IMPLEMENTATION"] == "tokenclassification w/ LC-CRF":
            ner_dataset.on_setup(use_crf=True)
            model = XLMRobertaCRFForTokenClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=len(ner_dataset.dataset.labels))
            model.to(device)  
            print_model_details(model)

        else:
            raise ValueError("Invalid MODEL_IMPLEMENTATION value. Please check the transformers_config.yaml file.")

        trainer = Trainer(model=model,
                          dataset=ner_dataset,
                          use_crf=ner_dataset.use_crf)
        trainer.train(use_patience=False)

    elif modeling_type == "lstm":
        pass
    elif modeling_type == "cnn":
        pass
import torch
from helpers import config_data as config
from transformers import AutoTokenizer
from data_utils import NERDataset_transformers
from models.modeling_xlm_roberta import XLMRobertaforTokenClassification, XLMRobertaCRFforTokenClassification
from model_utils import train, generate_results

def print_model_details(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {pytorch_total_trainable_params}")

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(config)

    modeling_type = config["MODELING_TYPE"]
    if modeling_type == "transformers":
        transformers_config = config["TRANSFORMERS"]
        model_checkpoint = transformers_config["MODEL_CHECKPOINT"]
        # load tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                  add_prefix_space=True) # specific to XLM-RoBERTa
        ner_dataset = NERDataset_transformers(tokenizer=tokenizer)
        # under transformers we have 3 different model implementation
        # this includes, i) TokenClassification (NER), ii) TokenClassification w/ Linear Chain CRF, 
        # iii) Ensemble of TokenClassification w/ and w/o Linear Chain CRF
        if transformers_config["MODEL_IMPLEMENTATION"] == "tokenclassification":
            ner_dataset.on_setup()
            print("Loading model...")
            model = XLMRobertaforTokenClassification.from_pretrained(model_checkpoint,
                                                                     num_labels=len(ner_dataset.dataset.labels))
            model.to(device)    
            # train model 
            train(model=model,
                  dataset=ner_dataset)
            # generate results
            results = generate_results(model=model,
                                       dataset=ner_dataset)
            print(results)

        elif transformers_config["MODEL_IMPLEMENTATION"] == "tokenclassification w/ LC-CRF":
            ner_dataset.on_setup(use_crf=True)
            print("Loading model...")
            model = XLMRobertaCRFforTokenClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=len(ner_dataset.dataset.labels))
            model.to(device)    
            # train model 
            train(model=model,
                  dataset=ner_dataset)
            # generate results
            results = generate_results(model=model,
                                       dataset=ner_dataset,
                                       use_crf=True)
            print(results)

        elif transformers_config["MODEL_IMPLEMENTATION"] == "ensemble tokenclassification w/ and w/o LC-CRF":
            pass
        else:
            raise ValueError("Invalid MODEL_IMPLEMENTATION value. Please check the config.yaml file.")
        pass

    elif modeling_type == "lstm":
        pass
    elif modeling_type == "cnn":
        pass
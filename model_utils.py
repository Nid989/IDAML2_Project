import os
import gc
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.utils.checkpoint
from evaluate import load
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from models.modeling_xlm_roberta import XLMRobertaForTokenClassification
from data_utils import NERDataset_transformers
from helpers import transformers_config_data as config_data

seqeval = load("seqeval")

def get_scores(p, ner_labels_list, full_rep: bool=False, use_crf: bool=False):
    predictions, labels = p

    if use_crf:
        ignore_idx_list = [0, 2, 1] # <s>, </s>, <pad>
    else:
        ignore_idx_list = [-100]

    true_predictions = [
        [ner_labels_list[p] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ner_labels_list[l] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=False)

    if full_rep:
        return results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
class Trainer:
    def __init__(self,
                 model: XLMRobertaForTokenClassification,
                 dataset: NERDataset_transformers,
                 use_crf: bool=False,
                 **hparam_kwargs):
        self.model = model
        self.dataset = dataset
        self.use_crf = use_crf
        self.batch_size = int(hparam_kwargs["batch_size"]) if "batch_size" in hparam_kwargs else int(config_data["BATCH_SIZE"])
        self.num_epochs = int(hparam_kwargs["num_epochs"]) if "num_epochs" in hparam_kwargs else int(config_data["NUM_EPOCHS"])
        self.learning_rate = float(hparam_kwargs["learning_rate"]) if "learning_rate" in hparam_kwargs else float(config_data["LEARNING_RATE"])
        self.weight_decay = float(hparam_kwargs["weight_decay"]) if "weight_decay" in hparam_kwargs else float(config_data["WEIGHT_DECAY"])

        self.optimizer = AdamW(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        self.train_dataloader = dataset.set_up_dataloader("train", batch_size=self.batch_size)
        self.valid_dataloader= dataset.set_up_dataloader("valid", batch_size=self.batch_size)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train(self, use_patience: bool=False, **gen_kwargs):
        train_losses = []
        valid_losses = []
        valid_f1 = []

        patience = 1
        early_stopping_threshold = gen_kwargs["early_stopping_threshold"] if "early_stopping_threshold" in gen_kwargs else config_data["EARLY_STOPPING_THRESHOLD"]

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            valid_loss = self.valid_epoch()
            valid_losses.append(valid_loss)

            valid_results = self.get_valid_scores(dataloader=self.valid_dataloader,
                                                  desc="Validation Generation Iteration",
                                                  **gen_kwargs)
            valid_f1.append(valid_results["f1"])

            print("Epoch: {:0.2f}\ttrain_loss: {:0.2f}\tval_loss: {:0.2f}\tmin_validation_loss: {:0.2f}".format(
                epoch+1, train_loss, valid_loss, min(valid_losses)))

            print("val_precision: {:0.2f}\tval_recall: {:0.2f}\tval_f1: {:0.2f}\tval_accuracy: {:0.2f}".format(
                valid_results["precision"], valid_results["recall"], valid_results["f1"], valid_results["accuracy"]))

            if use_patience:
                if valid_results["f1"] < max(valid_f1):
                    patience = patience + 1
                    if patience == early_stopping_threshold:
                        break
                else:
                    patience = 1

            del train_loss
            del valid_loss
            gc.collect()
            torch.cuda.empty_cache()

        self.save_model()

    def train_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(self.train_dataloader, desc="Training Iteration")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels, mask = batch
            self.optimizer.zero_grad()
            if self.use_crf:
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    mask=mask,
                                    reduction='mean')
            else:
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
            loss = outputs.loss
            epoch_train_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            pbar.set_description("train_loss={0:.3f}".format(loss.item()))

        del batch
        del input_ids
        del attention_mask
        del labels
        del mask
        del outputs
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_train_loss / step

    def valid_epoch(self):
        self.model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(self.valid_dataloader, desc="Validation Loss Iteration")
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels, mask = batch
                if self.use_crf:
                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        mask=mask,
                                        reduction="mean")
                else:
                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                loss = outputs.loss
                epoch_val_loss += loss.item()

                pbar.set_description("val_loss={0:.3f}".format(loss.item()))

        del batch
        del input_ids
        del attention_mask
        del labels
        del mask
        del outputs
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        return epoch_val_loss / step

    def test_epoch(self, dataloader, desc, **gen_kwargs):
        self.model.eval()
        out_predictions = []
        gold = []
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=desc)
            for step, batch in enumerate(pbar):
                batch = (t.to(self.device) for t in batch)
                input_ids, attention_mask, labels, mask = batch
                if self.use_crf:
                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **gen_kwargs)
                    predictions = outputs.predictions

                    out_predictions.extend(predictions.cpu().tolist())
                    gold.extend(labels.cpu().tolist())
                else:
                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        **gen_kwargs)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    predictions = probabilities.argmax(dim=-1).cpu().tolist()

                    out_predictions.extend(predictions)
                    gold.extend(labels.cpu().tolist())

        del batch
        del input_ids
        del attention_mask
        del labels
        del mask
        del outputs
        del probabilities
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        return out_predictions, gold

    def get_valid_scores(self,
                         dataloader,
                         desc,
                         log_results: bool=False,
                         **gen_kwargs):
        predictions, gold = self.test_epoch(dataloader=dataloader,
                                            desc=desc,
                                            **gen_kwargs)
        result = get_scores(p=(predictions, gold),
                            ner_labels_list=self.dataset.dataset.labels,
                            use_crf=self.use_crf)
        if log_results:
            test_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
            test_df.to_csv()
            print(test_df)

        del predictions
        del gold
        gc.collect()
        torch.cuda.empty_cache()

        return result
    
    def _save(self, state_dict=None):
        
        output_dir = config_data["PATH_TO_MODEL_OUTPUT_DIR"] + f"{config_data["MODEL_NAME"]}_{config_data["VERSION"]}_" + datetime.now().strftime("%Y%m%d%H%M%S")

        os.makedirs(output_dir, exist_ok=True)
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, "WEIGHTS_NAME"))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def save_model(self, state_dict=None):
        self._save(state_dict=state_dict)
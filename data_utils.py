import gc
import itertools
import functools
from collections import OrderedDict
from typing import List, Literal, Dict

import torch
import spacy
from torch.utils.data import DataLoader, TensorDataset
import torchtext
from transformers import (
    AutoTokenizer
)
from dataset import BC5CDR, MultiCoNER2
from helpers import load_glove_embeddings

# nlp = spacy.load("en_core_web_sm")
dataset_types_ = Literal['BC5CDR', 'MultiCoNER2']
data_types_ = Literal['train', 'valid', 'test']
    
# ---------------------- # NERDataset for LSTM-CNN # ---------------------- #
class NERDataset_lstm_cnn:
    def __init__(self, config: Dict, **kwargs):
        self.modeling_type = "lstm_cnn"
        self.dataset_name = config["DATASET_NAME"]
        self.use_glove = kwargs["use_glove"] if "use_glove" in kwargs else config["USE_GLOVE"]
        self.rmv_stopwords = kwargs["rmv_stopwords"] if "rmv_stopwords" in kwargs else config["RMV_STOPWORDS"]
        self.lemmatize = kwargs["lemmatize"] if "lemmatize" in kwargs else config["LEMMATIZE"]
        # self.on_setup(**kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.use_glove:
            if "glove_dim" not in kwargs:
                raise AttributeError("Glove initialization requires the specification of embedding dimension!")
            else:
                glove_filename = "./glove.6B.{}d.txt".format(int(kwargs["glove_dim"]))
                self.glove_embeds, glove_vocab = load_glove_embeddings(glove_filename, dimension=int(kwargs["glove_dim"]))
                # There appears to be an issue with torchtext.vocab.vocab where it deletes the initial token in the vocabulary (OrderedDict);
                # therefore, we need to append an ignore token at the beginning.
                glove_vocab = ["<del>"] + glove_vocab
                glove_vocab_stoi = OrderedDict(dict(zip(glove_vocab, range(len(glove_vocab)))))
                self.vocab = torchtext.vocab.vocab(glove_vocab_stoi)
                self.vocab.insert_token("<pad>", 0)
                self.vocab.insert_token("<unk>", 1)
                self.vocab.set_default_index(0)
                self.vocab.set_default_index(1)
        else:
            self.vocab = self.prepare_vocab()

    def on_setup(self, **kwargs):
        if self.dataset_name == "BC5CDR":
            self.dataset = BC5CDR("./data/bc5cdr_data/")
        elif self.dataset_name == "MultiCoNER2":
            if "lang" not in kwargs:
                lang = "en"
                print("Selecting `en` as default value for attr. `lang` under MultiCoNER2")
            else:
                lang = kwargs["lang"]
            self.dataset = MultiCoNER2("./data/multiconer2_data/", lang=lang)
        else:
            raise ValueError(f"Invalid dataset-name provided, expected dataset-types are {dataset_types_}")

    def prepare_vocab(self):
        all_data = list(itertools.chain.from_iterable([getattr(self.dataset, f"{data_type}_data")["tokens"].values.tolist() \
                    for data_type in ["train", "valid"]]))
        if self.lemmatize:
            pass
            # all_data = [[token.lemma_ for token in nlp(" ".join(instance))] for instance in all_data]
        _vocab = torchtext.vocab.build_vocab_from_iterator(all_data,
                                           specials=["<pad>", "<unk>"])
        return _vocab

    def preprocess_data(self, data_type: data_types_="train"):

        def _map_tokens_to_ids(instance: List[str]):
            return [self.vocab[token] if token in self.vocab else self.vocab["<unk>"] for token in instance]

        def _map_labels_to_ids(instance: List[str]):
            return [self.dataset.labels2idx[label] for label in instance]

        def _pad_sequence(sequence: List[int], max_length: int=150, padding_token_id: int=0):
            return sequence[:max_length] + [padding_token_id] * max(0, max_length - len(sequence))

        dataset_split = getattr(self.dataset, f"{data_type}_data")

        if self.rmv_stopwords:
            pass
            # dataset_split["tokens"] = dataset_split.apply(
            #     lambda row: [token.text for token in nlp(" ".join(row["tokens"])) if not token.is_stop],
            #     axis=1
            # )
        if self.lemmatize:
            pass
            # dataset_split["tokens"] = dataset_split.apply(
            #     lambda row: [token.lemma_ for token in nlp(" ".join(row["tokens"]))],
            #     axis=1
            # )
        setattr(self.dataset, f"{data_type}_data", dataset_split)

        model_inputs = dict()
        source = dataset_split["tokens"].values.tolist()
        target = dataset_split["labels"].values.tolist()
        sequence_length = dataset_split.apply(lambda row: len(row["tokens"]), axis=1).values.tolist()
        _pad_sequence_partial = functools.partial(_pad_sequence, max_length=150, padding_token_id=0)
        model_inputs["word_ids"] = list(map(_pad_sequence_partial, list(map(_map_tokens_to_ids, source))))
        _pad_sequence_partial = functools.partial(_pad_sequence, max_length=150, padding_token_id=-100)
        model_inputs["labels"] = list(map(_pad_sequence_partial, list(map(_map_labels_to_ids, target))))
        model_inputs["sequence_length"] = [torch.tensor(i) for i in sequence_length]

        model_inputs["word_ids"] = torch.tensor([i for i in model_inputs["word_ids"]], dtype=torch.long, device=self.device)
        model_inputs["labels"] = torch.tensor([i for i in model_inputs["labels"]], dtype=torch.long, device=self.device)
        model_inputs["sequence_length"] = torch.tensor([i for i in model_inputs["sequence_length"]], dtype=torch.long, device=self.device)
        model_inputs["mask"] = model_inputs["labels"].ne(0).int()
        return model_inputs

    def set_up_dataloader(self, data_type: data_types_="train", batch_size: int=2):
        dataset_split = self.preprocess_data(data_type=data_type)
        dataset_split = TensorDataset(dataset_split["word_ids"],
                                      dataset_split["labels"],
                                      dataset_split["sequence_length"],
                                      dataset_split["mask"])
        gc.collect()
        return DataLoader(dataset_split,
                          batch_size=batch_size,
                          shuffle=True)
    
# ---------------------- # NERDataset for Transformers # ---------------------- #
class NERDataset_transformers:
    def __init__(self, tokenizer: AutoTokenizer, config: Dict, **kwargs):
        self.modeling_type = "transformers"
        self.dataset_name = config["DATASET_NAME"]
        self.tokenizer = tokenizer
        self.max_seq_len = kwargs["max_sequence_len"] if "max_sequence_len" in kwargs else config["MAX_SEQUENCE_LEN"]
        # NOTE: (callable under "__main__") on-set depends upon the whether we're implementing TokenClassification or TokenClassification w/ Linear Chain CRF.
        # self.on_setup(**kwargs)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def on_setup(self, **kwargs):
        if self.dataset_name == "BC5CDR":
            self.dataset = BC5CDR("./data/bc5cdr_data/")
        elif self.dataset_name == "MultiCoNER2":
            if "lang" not in kwargs:
                lang = "en"
                print("Selecting `en` as default value for attr. `lang` under MultiCoNER2")
            else:
                lang = kwargs["lang"]
            self.dataset = MultiCoNER2("./data/multiconer2_data/", lang=lang)
        else:
            raise ValueError(f"Invalid dataset-name provided, expected dataset-types are {dataset_types_}")

    def preprocess_data(self, data_type: data_types_="train"):
        dataset_split = getattr(self.dataset, f"{data_type}_data")

        source = [s for s in dataset_split["tokens"].values.tolist()]
        model_inputs = self.tokenizer(source,
                                      max_length=self.max_seq_len,
                                      padding="max_length",
                                      truncation=True,
                                      is_split_into_words=True)

        def synchronize_labels(dataset_split, model_inputs):
            """
            synchronize labels w.r.t tokenized model inputs
            """
            label_all_tokens = True
            NER_labels = []
            NER_labels_mask = []
            for index, label in enumerate(dataset_split["labels"].values.tolist()):
                word_ids = model_inputs.word_ids(batch_index=index)
                previous_word_idx = None
                label_ids = []
                label_masks = []
                for index2, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        label_id = -100
                        label_ids.append(label_id)
                        label_masks.append(0)
                    elif label[word_idx] == 0:
                        label_ids.append(0)
                        label_masks.append(1)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.dataset.labels2idx[label[word_idx]])
                        label_masks.append(1)
                    else:
                        label_ids.append(self.dataset.labels2idx[label[word_idx]] if label_all_tokens else -100)
                        label_masks.append(1 if label_all_tokens else 0)
                    previous_word_idx = word_idx
                NER_labels.append(label_ids)
                NER_labels_mask.append(label_masks)
            return NER_labels, NER_labels_mask

        labels, labels_mask = synchronize_labels(dataset_split, model_inputs)
        model_inputs["labels"] = labels
        model_inputs["mask"] = labels_mask
        model_inputs["input_ids"] = torch.tensor([i for i in model_inputs["input_ids"]], dtype=torch.long, device=self.device)
        model_inputs["attention_mask"] = torch.tensor([i for i in model_inputs["attention_mask"]], dtype=torch.long, device=self.device)
        model_inputs["labels"] = torch.tensor([i for i in model_inputs["labels"]], dtype=torch.long, device=self.device)
        model_inputs["mask"] = torch.tensor([i for i in model_inputs["mask"]], dtype=torch.long, device=self.device)

        del dataset_split
        del source
        del labels
        del labels_mask
        gc.collect()
        return model_inputs

    # NOTE: remove arg `exp_run` once finalized with everything!
    def set_up_dataloader(self, data_type: data_types_="train", batch_size: int=2):
        dataset_split = self.preprocess_data(data_type=data_type)
        dataset_split = TensorDataset(dataset_split["input_ids"],
                                      dataset_split["attention_mask"],
                                      dataset_split["labels"],
                                      dataset_split["mask"])
        gc.collect()
        return DataLoader(dataset_split,
                          batch_size=batch_size,
                          shuffle=True)
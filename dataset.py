import os
import pandas as pd 
import itertools
import typing
from collections import defaultdict
from typing import Dict, List, Literal, Optional

# ---------------------- # BC5CDR dataset # ---------------------- # 
class BC5CDR:
    def __init__(self, path_to_data_dir: str, remove_IOB: bool=False):
        super().__init__()
        self.path_to_data_dir = path_to_data_dir
        self.remove_IOB = remove_IOB
        self._load_data()
        self._configure_labels()

    def _configure_labels(self):
        # BC5CDR coprus; define 2 distinct named entity types (i.e. `Diesease`, `Chemical`).
        self.types = self._get_entity_types()
        self.labels = list(itertools.chain.from_iterable([[f"{tag}-{label}" for tag in ['B', 'I']] for label in self.types] + [['O']]))
        self.labels2idx = dict(zip(self.labels, list(range(len(self.labels)))))
        self.idx2labels = dict((idx, label) for label, idx in self.labels2idx.items())

    def __str__(self):
        specifics = """BC5CDR dataset
Entity types: { $ENTITY_TYPES$ }
Train data: $TRAIN_DATA_SIZE$
Validation data: $VALID_DATA_SIZE$
Test data: $TEST_DATA_SIZE$"""
        specifics = specifics.replace("$ENTITY_TYPES$", ", ".join(self.types))
        specifics = specifics.replace("$TRAIN_DATA_SIZE$", str(self.train_data.shape[0]))
        specifics = specifics.replace("$VALID_DATA_SIZE$", str(self.valid_data.shape[0]))
        specifics = specifics.replace("$TEST_DATA_SIZE$", str(self.test_data.shape[0]))
        return specifics

    def _load_data(self):
        self.train_docs = self._read_data(filename="cdr_train.conll")
        self.valid_docs = self._read_data(filename="cdr_valid.conll")
        self.test_docs = self._read_data(filename="cdr_test.conll")

        self.train_data = pd.DataFrame(self.train_docs)
        self.valid_data = pd.DataFrame(self.valid_docs)
        self.test_data = pd.DataFrame(self.test_docs)

    def _read_data(self, filename: str) -> Dict[str, List[str]]:
        file_path = os.path.join(self.path_to_data_dir, filename)
        documents = defaultdict(list)
        with open(file_path, 'r') as file:
            current_doc_id = None
            tokens = []
            labels = []
            counter = 0
            for line in file:
                if line.startswith('# doc_id'):
                    if current_doc_id is not None:
                        documents['doc_id'].append(current_doc_id)
                        documents['tokens'].append(tokens)
                        documents['labels'].append(labels)
                    current_doc_id = int(line.split('=')[1].strip())
                    tokens = []
                    labels = []
                else:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        tokens.append(parts[0])
                        labels.append(parts[1].split('-')[-1] if self.remove_IOB else parts[1])
            # append the last documents
            if current_doc_id is not None:
                documents['doc_id'].append(current_doc_id)
                documents['tokens'].append(tokens)
                documents['labels'].append(labels)
        return documents

    def _get_entity_types(self) -> List[str]:
        types = ['Disease', 'Chemical']
        return types
        
# ---------------------- # MultiCoNER2 dataset # ---------------------- #
class MultiCoNER2:
    def __init__(self, path_to_data_dir: str, lang: Optional[str]=None, remove_IOB: bool=False):
        super().__init__()
        self.path_to_data_dir = path_to_data_dir
        self.lang = None
        self.remove_IOB = remove_IOB
        self._validate_lang_attr(lang)
        self._load_data()
        self._configure_labels()

    def _validate_lang_attr(self, lang):
        languages_ = Literal["bn", "de", "en", "es", "fa", "fr", "hi", "it", "multi", "pt", "sv", "uk", "zh"]
        lang = "en" if lang is None else lang
        if lang not in typing.get_args(languages_):
            raise ValueError("The language specified as 'lang' is not considered valid.")
        else:
            super().__setattr__("lang", lang)

    def _configure_labels(self):
        # MultiCoNER2 dataset; define 33 distinct entity types
        self.types = self._get_entity_types()
        self.labels = list(itertools.chain.from_iterable([[f"{tag}-{label}" for tag in ["B", "I"]] for label in self.types] + [["O"]]))
        self.labels2idx = dict(zip(self.labels, list(range(len(self.labels)))))
        self.idx2labels = dict((idx, label) for label, idx in self.labels2idx.items())

    def __str__(self):
        specifics = """MultiCoNER dataset
Entity types: { $ENTITY_TYPES$ }
Train data: $TRAIN_DATA_SIZE$
Validation data: $VALID_DATA_SIZE$
Test data: $TEST_DATA_SIZE$"""
        specifics = specifics.replace("$ENTITY_TYPES$", ", ".join(self.types))
        specifics = specifics.replace("$TRAIN_DATA_SIZE$", str(self.train_data.shape[0]))
        specifics = specifics.replace("$VALID_DATA_SIZE$", str(self.valid_data.shape[0]))
        specifics = specifics.replace("$TEST_DATA_SIZE$", str(self.test_data.shape[0]))
        return specifics

    def _load_data(self):
        self.train_docs = self._read_data(f"{self.lang}-train.conll")
        self.valid_docs = self._read_data(f"{self.lang}-dev.conll")
        self.test_docs = self._read_data(f"{self.lang}_test_withtags.conll")

        self.train_data = pd.DataFrame(self.train_docs)
        self.valid_data = pd.DataFrame(self.valid_docs)
        self.test_data = pd.DataFrame(self.test_docs)

    def _read_data(self, filename: str) -> Dict[str, List[str]]:
            file_path = os.path.join(self.path_to_data_dir, filename)
            documents = defaultdict(list)
            with open(file_path, "r") as file:
                current_doc_id = None
                tokens = []
                labels = []
                counter = 0
                for line in file:
                    if line.startswith("# id"):
                        if current_doc_id is not None:
                            documents["doc_id"].append(current_doc_id)
                            documents["tokens"].append(tokens)
                            documents["labels"].append(labels)
                        current_doc_id = line.split(" ")[2].strip()
                        tokens = []
                        labels = []
                    else:
                        parts = line.strip().split()
                        if len(parts) == 4:
                            tokens.append(parts[0])
                            labels.append(parts[-1].split("-")[-1] if self.remove_IOB else parts[-1])
                # append the last documents
                if current_doc_id is not None:
                    documents["doc_id"].append(current_doc_id)
                    documents["tokens"].append(tokens)
                    documents["labels"].append(labels)
            return documents

    def _get_entity_types(self) -> List[str]:
        # extract entity types from derived train data
        types = sorted(set([word.split("-")[-1] for word in itertools.chain.from_iterable(self.train_data["labels"]) if word != "O"])) + ["O"]
        # save_to_pickle(types, f"{self.lang}_entity_types.pkl")
        return types
import os
import gc
import yaml
import pandas as pd
import numpy as np
import itertools
import typing
import logging
import functools
from tqdm.auto import tqdm
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Literal, Optional, Union

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

from model_utils import get_scores

data_types_ = Literal['train', 'valid', 'test']
metric_types_ = Literal["precision", "recall", "f1", "accuracy"]

class NER_Evaluation:
    def __init__(self, model, dataset, data_type: data_types_="valid", modeling_type="transformers"):
        super(NER_Evaluation, self).__init__()

        self.model = model
        self.dataset = dataset
        self.data_type = data_type
        self.modeling_type = modeling_type 

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._on_setup()

    def _on_setup(self):
        results, (true_predictions, true_labels) = self._generate_results()
        self.results = results
        self.true_predictions = true_predictions
        self.true_labels = true_labels
        self.results_df = self._tabularize_results()

    def _generate_results(self):
        predictions = []
        ground_truth = []
        data_loader = self.dataset.set_up_dataloader(data_type=self.data_type)
        for batch in tqdm(data_loader):
            if self.modeling_type == "transformers":
                input_ids, attention_mask, labels, _ = batch
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                probabilities = F.softmax(outputs.logits, dim=-1)
                batch_predictions = probabilities.argmax(dim=-1).to("cpu").tolist()
                predictions.extend(batch_predictions)
                ground_truth.extend(labels.to("cpu").tolist())
            elif self.modeling_type == "lstm_cnn":
                word_ids, labels, _, mask = batch
                outputs = self.model(word_ids=word_ids)
                probabilities = F.softmax(outputs.logits, dim=-1)
                batch_predictions = probabilities.argmax(dim=-1).to("cpu").tolist()
                predictions.extend(batch_predictions)
                ground_truth.extend(labels.to("cpu").tolist())

        del probabilities
        del batch_predictions
        gc.collect()

        torch.cuda.empty_cache()
        results, (true_predictions, true_labels) = get_scores((predictions, ground_truth),
                                                              ner_labels_list=self.dataset.dataset.labels,
                                                              full_rep=True)
        return results, (true_predictions, true_labels)

    def _tabularize_results(self):
        results_dict = defaultdict(list)
        for key in self.results.keys():
            if key in ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"]:
                continue
            results_dict["entity"].append(key)
            results_dict["precision"].append(self.results[key]["precision"])
            results_dict["recall"].append(self.results[key]["recall"])
            results_dict["f1"].append(self.results[key]["f1"])
            results_dict["number"].append(self.results[key]["number"])
        return pd.DataFrame(results_dict)

    def plot_evaluation_metric(self, metric: metric_types_, figsize=(8, 4), rotation=90):
        self.results_df[metric].plot(kind='line', figsize=figsize, title=metric.capitalize(), marker='', linestyle='-')
        self.results_df[metric].plot(kind='line', marker='o', color='red', linestyle='None')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xticks(range(len(self.results_df)), self.results_df['entity'], rotation=rotation)
        plt.xlabel('Entity')
        plt.ylabel(metric.capitalize())
        plt.show()

    def get_aggregated_entities(self):
        return {
            "Medical": ["Disease", "Symptom", "AnatomicalStructure", "MedicalProcedure", "Medication/Vaccine"],
            "Product": ["OtherPROD", "Drink", "Food", "Vehicle", "Clothing"],
            "Person": ["OtherPER", "SportsManager", "Cleric", "Politician", "Athlete", "Artist", "Scientist"],
            "Group": ["MusicalGRP", "PublicCorp", "PrivateCorp", "AerospaceManufacturer", "SportsGRP", "CarManufacturer", "ORG"],
            "Creative Works": ["VisualWork", "MusicalWork", "WrittenWork", "ArtWork", "Software"],
            "Location": ["Facility", "OtherLOC", "HumanSettlement", "Station"]
        }

    # visualization using t-SNE only limited to transformers based models
    def visualize_entities(self, aggregate_categories: bool=False):
        def process_batch(batch):
            input_ids, attention_mask, labels, mask = batch
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels,
                                 output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            return last_hidden_state, labels

        def find_aggregate_category(label: str):
            aggregated_entities_dict = self.get_aggregated_entities()
            for category, entity_list in aggregated_entities_dict.items():
                if label in entity_list:
                    return category
            return label

        dataloader = self.dataset.set_up_dataloader(data_type=self.data_type, batch_size=1)
        all_last_hidden_states = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation Data Iteration")
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                last_hidden_state, labels = process_batch(batch)
                all_last_hidden_states.append(last_hidden_state)
                all_labels.append(labels)

        all_labelwise_embeddings = []
        for labels, last_hidden_state in tqdm(zip(all_labels, all_last_hidden_states), total=len(all_labels)):
            labels = labels.squeeze(0).tolist()
            last_hidden_state = last_hidden_state.squeeze(0)
            current_tag = None
            current_id = None
            current_sum = None
            count = 0
            for index, label_id in enumerate(labels):
                if label_id == -100 or label_id == 68:
                    continue
                tag = self.dataset.dataset.idx2labels[label_id].split("-")[-1].strip()
                if label_id != current_id and tag != current_tag:
                    if current_id is not None:
                        averaged_embedding = current_sum / count
                        all_labelwise_embeddings.append((tag, averaged_embedding))
                    current_tag = tag
                    current_id = label_id   
                    current_sum = last_hidden_state[index]
                    count = 1
                elif label_id != current_id and tag == current_tag:
                    current_id = label_id
                    current_sum += last_hidden_state[index]
                    count += 1
                else:
                    current_sum += last_hidden_state[index]
                    count += 1
            if current_id is not None:
                averaged_embedding = current_sum / count
                all_labelwise_embeddings.append((tag, averaged_embedding))

        entity_tags_list, entity_embeddings_list = zip(*all_labelwise_embeddings)

        entity_embeddings = torch.stack(entity_embeddings_list)
        entity_embeddings = entity_embeddings.detach().to("cpu").numpy()

        tsne = TSNE(n_components=2, random_state=42)
        embeddings_transform = tsne.fit_transform(entity_embeddings)

        embeddings_transform_df = pd.DataFrame(embeddings_transform, columns=["Dimension 1", "Dimension 2"])
        embeddings_transform_df["label"] = entity_tags_list

        if aggregate_categories:
            embeddings_transform_df["label"] = embeddings_transform_df.apply(
                lambda row: find_aggregate_category(row["label"]),
                axis=1
            )

        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=embeddings_transform_df,
                        x="Dimension 1",
                        y="Dimension 2",
                        hue="label",
                        palette="tab20",
                        markers="o",
                        s=100,
                        legend="full")
        plt.title("t-SNE Cluster Representation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
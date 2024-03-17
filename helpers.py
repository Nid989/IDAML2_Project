import os
import pickle
import yaml
import pandas as pd
import numpy as np
from typing import List

# import seaborn as sns
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer

from models.modeling_xlm_roberta import XLMRobertaForTokenClassification

from skopt import dump, load

def check_and_create_directory(path_to_folder):
    """
    check if a nested path exists and create 
    missing nodes/directories along the route
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    return path_to_folder

def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_from_pickle(data, filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

general_config_data = load_from_yaml("./configurations/config.yaml")
transformers_config_data = load_from_yaml("./configurations/transformers_config.yaml")
lstm_cnn_config_data = load_from_yaml("./configurations/lstm_cnn_config.yaml")

def save_skopt_results(results, filename: str="optimization_results.pkl", **kwargs):
    if "logging" in kwargs:
        kwargs["logging"].info("Saving the scikit-optimize (bayesian optimization) results w/ filename")
    dump(results, filename)

def load_skopt_results(filename: str="optimization_results.pkl"):
    return load(filename)

def save_results(path_to_file: str, results: dict):
    classnames, f1_scores, precision_scores, recall_scores, numbers = [], [], [], [], []
    for key, value in results.items():
        if not key.startswith("overall"):
            classnames.append(key)
            f1_scores.append(results[key]["f1"])
            precision_scores.append(results[key]["precision"])
            recall_scores.append(results[key]["recall"])
            numbers.append(results[key]["number"])

    results_df = pd.DataFrame({
        "CLASS": classnames,
        "F1": f1_scores,
        "PRECISION": precision_scores,
        "RECALL": recall_scores,
        "COUNT": numbers
    })
    results_df.set_index("CLASS", inplace=True)
    results_df.to_csv(path_to_file)
    print(f"File saved at location: {path_to_file}")

def print_model_details(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {pytorch_total_trainable_params}")

def load_glove_embeddings(embedding_file, dimension):
    path_to_embedding_file = os.path.join("./embeddings/", embedding_file)
    embeddings_index = {}
    with open(path_to_embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create a numpy array to store the embeddings
    embedding_matrix = np.zeros((len(embeddings_index), dimension))
    words_list = []

    for i, word in enumerate(embeddings_index):
        embedding_matrix[i] = embeddings_index[word]
        words_list.append(word)

    return embedding_matrix, words_list

# def plot_attentions(input_text: List[str],
#                     target_word: str,
#                     model: XLMRobertaForTokenClassification,
#                     tokenizer: AutoTokenizer, 
#                     aggregate_heads: bool=False):
#     if target_word not in input_text:
#         raise ValueError("parameter `target_word` is not present in the provided `input_text`.")
    
#     tokens = tokenizer.tokenize(input_text, is_split_into_words=True, add_special_tokens=True)
#     encodings = tokenizer(input_text, is_split_into_words=True)
#     word_ids = encodings.word_ids()
#     # index of `target_word` in `input_text` (original)
#     target_word_idx = input_text.index(target_word)
#     # index(s) of tokenized `target_word` in tokenized representation
#     target_word_new_idxs = [index for index, word_id in enumerate(word_ids) if word_id == target_word_idx]

#     # compute output_attentions & hidden_states        
#     with torch.no_grad():
#         outputs = model(**tokenizer([input_text], return_tensors="pt", is_split_into_words=True), output_attentions=True)
#         attentions = outputs.attentions
#         attentions = torch.cat(attentions).to("cpu")

#     attentions = attentions.permute(2, 1, 0, 3)
#     layers = len(attentions[0][0])
#     heads = len(attentions[0])
#     seqlen = len(attentions)

#     # attention weights corresponding to the `target_word` token index(s)
#     attentions_pos = attentions[target_word_new_idxs]

#     if not aggregate_heads:
#         cols = 2
#         rows = int(heads/cols)
#         for index, pos in enumerate(target_word_new_idxs):
#             attentions_pos_ = attentions_pos[index]
#             fig, axes = plt.subplots(rows, cols, figsize=(15,35))
#             axes = axes.flat
#             print (f'Attention weights for token {tokens[pos]}')
#             for i, att in enumerate(attentions_pos_):
#                 im = axes[i].imshow(att, cmap='gray')
#                 sns.heatmap(att, vmin=0, vmax=1, ax=axes[i], xticklabels=tokens)
#                 axes[i].set_title(f'head - {i} ' )
#                 axes[i].set_ylabel('layers')
#     else:
#         plt.figure(figsize=(7,4))
#         for index, pos in enumerate(target_word_new_idxs):
#             attentions_pos_ = attentions_pos[index]            
#             avg_attention = attentions_pos_.mean(dim=0)
#             sns.heatmap(avg_attention, vmin=0, vmax=1, xticklabels=tokens)

#     plt.show()
import os
import pickle
import yaml
import pandas as pd
import numpy as np

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
import pickle

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(data, filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
import pandas as pd
import numpy as np

def index_vocab(vocab):
    vocab2id = {word: i for i, word in enumerate(sorted(vocab))}
    id2vocab = {v: k for k, v in vocab2id.items()}
    return vocab2id, id2vocab

def read_csv(f):
    try:
        return pd.read_csv(f, header=None).values
    except pd.errors.EmptyDataError:
        return []

def read_isets(f, id2vocab):
    lines = read_csv(f)
    if len(lines) == 0:
        return []
    sets = [[id2vocab[int(w)] for w in x.split()] 
            for x in lines[:,0]]
    return sets

def read_topics(f, id2vocab):
    lines = read_csv(f)
    if len(lines) == 0:
        return []
    topics = [[id2vocab[int(w)] for w in x.split()] 
            for x in lines[:,0]] 
    scores = [ float(x) for x in lines[:,1]]
    return list(zip(topics,scores))
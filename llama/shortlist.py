from algo.extract import MMEKC
from algo.topics import get_topics
import pickle
import pandas as pd
import numpy as np
import os
from functools import partial
from multiprocessing import Pool
from time import time
from llama import Tokenizer


def shortlist_decode(vocab2id, tokenizer_path, extra_vocab=[]):

    tokenizer = Tokenizer(model_path=tokenizer_path)

    vocab2token = {}

    for word in vocab2id:
        if word.isalpha() and '\n' not in word:
            vocab2token[word] = tokenizer.encode(word, bos=True, eos=False)[1:]
            
    if len(extra_vocab) > 0:
        vocab2token = {k:vocab2token[k] for k in extra_vocab 
                       if k in vocab2token}

    common_idx2vocab = {i: vocab2id[word] for i,word in enumerate(vocab2token.keys())}
    vocab2sum = np.array([len(v) for k,v in vocab2token.items()])
    vocab2idx = np.zeros((len(vocab2sum), 32000), dtype=bool)
    for i, v in enumerate(vocab2token.values()):
        vocab2idx[i, v] = 1
    return {'vocab2sum': vocab2sum,
            'vocab2idx': vocab2idx,
            'idx2vocab': common_idx2vocab}


def process_beta_llama(beta, shortlist_decoder):
    beta_index = np.zeros((32000,), dtype=bool)
    beta_index[beta] = 1
    arr = np.where(np.equal(
        np.sum(beta_index & shortlist_decoder['vocab2idx'], axis=1),
        shortlist_decoder['vocab2sum'])
        )[0]
    return arr


def process_beta_llama_efficient(beta, old_arr, shortlist_decoder):
    beta_index = np.zeros((32000,), dtype=bool)
    beta_index[beta] = 1
    arr = np.where(np.equal(
        np.sum(beta_index & shortlist_decoder['vocab2idx'][old_arr],
               axis=1),
        shortlist_decoder['vocab2sum'][old_arr])
        )[0]
    return old_arr[arr]


def multi_helper(inp, tau, dest_dir, thresholds,
                 size_limits, shortlist_decoder,
                 graph_dir, num_windows, min_freq,
                 single_prob, time_limit_s, verbose):

    name, arg_matrix = inp
    cut_neg = cut_pos = tau
    neg = process_beta_llama(arg_matrix[:cut_neg], shortlist_decoder)
    while len(neg) > tau and cut_neg > 0:
        cut_neg -= 50
        neg = process_beta_llama_efficient(arg_matrix[:cut_neg], 
                                           neg, shortlist_decoder)
        print(len(neg))
    neg = [shortlist_decoder['idx2vocab'][w] for w in neg]

    pos = process_beta_llama(arg_matrix[-cut_pos:], shortlist_decoder)
    while len(pos) > tau and cut_pos > 0:
        cut_pos -= 50
        pos = process_beta_llama_efficient(arg_matrix[-cut_pos:], 
                                           pos, shortlist_decoder)
    pos = [shortlist_decoder['idx2vocab'][w] for w in pos]

    pickle.dump(neg, open(f"{dest_dir}/{name}_neg.pkl", 'wb'))
    pickle.dump(pos, open(f"{dest_dir}/{name}_pos.pkl", 'wb'))

    start = time()
    if verbose >= 2:
        print('Start :', name)
        print(f"BEGIN: {name} :", len(pos), len(neg))
    results, isets = MMEKC(get_topics(neg, graph_dir, num_windows,
                                      min_freq, single_prob),
                           thresholds, size_limits, time_limit_s)

    results = [((" ".join([str(x) for x in r]), s)) for r, s in results]
    isets = [" ".join([str(x) for x in iset]) for iset in isets]
    pd.DataFrame(results).to_csv(
        f"{dest_dir}/{name}_neg_topics.csv", header=None, index=None)
    pd.DataFrame(isets).to_csv(
        f"{dest_dir}/{name}_neg_isets.csv", header=None, index=None)

    if verbose >= 2:
        print(f"HALF : {name} : {round(time()-start)}s")

    results, isets = MMEKC(get_topics(pos, graph_dir, num_windows,
                                      min_freq, single_prob), thresholds, size_limits, time_limit_s)
    results = [((" ".join([str(x) for x in r]), s)) for r, s in results]
    isets = [" ".join([str(x) for x in iset]) for iset in isets]
    pd.DataFrame(results).to_csv(
        f"{dest_dir}/{name}_pos_topics.csv", header=None, index=None)
    pd.DataFrame(isets).to_csv(
        f"{dest_dir}/{name}_pos_isets.csv", header=None, index=None)

    if verbose >= 1:
        print(f"STOP : {dest_dir}/{name} : {round(time()-start)}s")


def shortlist_solve(betas, dest_dir, tau, thresholds, size_limits,
                    graph_dir, num_windows, min_freq, single_prob,
                    shortlist_decoder, time_limit_s=180,
                    workers=4, names=[], verbose=0):

    arg_matrix = np.argsort(betas)
    os.makedirs(dest_dir, exist_ok=True)

    if len(names) != len(betas):
        inputs = list(enumerate(arg_matrix))
    else:
        inputs = list(zip(names, arg_matrix))

    with Pool(processes=workers) as pool:
        print('processing:', len(inputs))
        pool.map(partial(multi_helper, dest_dir=dest_dir, tau=tau,
                         thresholds=thresholds, size_limits=size_limits,
                         shortlist_decoder=shortlist_decoder,
                         graph_dir=graph_dir, num_windows=num_windows,
                         min_freq=min_freq, single_prob=single_prob,
                         time_limit_s=time_limit_s, verbose=verbose),
                 inputs)

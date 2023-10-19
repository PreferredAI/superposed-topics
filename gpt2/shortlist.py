from algo.extract import MMEKC
from algo.topics import load_joint_prob_graph, npmi, create_graph_with
from gpt2.bpe_encoder import get_encoder
import pickle
import pandas as pd
import numpy as np
import os
from functools import partial
from multiprocessing import Pool
from time import time
from tqdm import tqdm


def shortlist_decode(vocab2id, model_size, models_dir, extra_vocab=[]):
    encoder = get_encoder(model_size, models_dir)
    processed_decoded = {k: v[1:].lower() if v.startswith(
        "Ġ") else v.lower() for k, v in encoder.decoder.items()}

    common = set(processed_decoded.values()).intersection(set(vocab2id.keys()))
    if len(extra_vocab) > 0:
        common = common.intersection(set(extra_vocab))

    shortlist_decoder = {k: vocab2id[v]
                         for k, v in processed_decoded.items()
                         if v in common}

    return shortlist_decoder


def get_topics(indices, graph_dir, num_windows, min_freq, single_prob):
    joint_prob = load_joint_prob_graph(graph_dir, num_windows, min_freq,
                                       shortlist=indices)
    g = create_graph_with(npmi, joint_prob, single_prob,
                          smooth=True, shortlist=indices)
    return g


def multi_helper(inp, tau, dest_dir, thresholds, 
                 size_limits, shortlist_decoder, 
                 graph_dir, num_windows, min_freq, 
                 single_prob, time_limit_s, verbose):

    name, arg_matrix = inp
    # neg = process_beta(arg_matrix[:tau], shortlist_decoder)
    # pos = process_beta(arg_matrix[-tau:], shortlist_decoder)

    neg = {shortlist_decoder[k] for k in arg_matrix[:tau] 
           if k in shortlist_decoder}
    pos = {shortlist_decoder[k] for k in arg_matrix[-tau:] 
           if k in shortlist_decoder}

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
                         thresholds=thresholds,size_limits=size_limits,
                         shortlist_decoder=shortlist_decoder,
                         graph_dir=graph_dir, num_windows=num_windows, 
                         min_freq=min_freq, single_prob=single_prob,
                         time_limit_s=time_limit_s, verbose=verbose),
                 inputs)

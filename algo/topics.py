'''
Code modified from
https://github.com/PreferredAI/topic-metrics/blob/main/topic_metrics/
MIT
Selected choice functions
'''
import os
import pickle
import numpy as np
import pandas as pd
from functools import partial, reduce
from math import log

EPS = EPSILON = 1e-12

def npmi(p_a_b, p_a, p_b, smooth=True, default_to=0):
    """ Normalised Pointwise-mutual information (Bouma, 2009)

    Parameters
    ----------
    p_a_b: float
        joint co-occurence probability for word a & b
    p_a: float
        probability for word a
    p_b: float
        probability for word b
    smooth : bool
        decides the use of epsilon=1e-12 or default
    default_to : float
        value used if smooth == False
    Return
    ------
    score : float

    """
    if smooth:
        p_a_b += EPS
    if p_a == 0 or p_b == 0 or p_a_b == 0:
        return default_to
    return log(p_a_b / (p_a * p_b)) / -log(p_a_b)

def create_prob_graph(graph, num_windows, min_freq):
    """ Create prob. graph from count graph fulfilling minimum freq. count

    Parameters
    ----------
    graph : Dict[int, int]
        count graph
    num_windows : int
        num_windows number of sliding windows in corpus
    min_freq : int
        lower bound to exclude rare words with counts less than

    Return
    ------
    probability graph : Dict[int, float]   
    """
    return {k: v/num_windows if v >= min_freq else 0 for k, v in graph.items()}

def aggregate_prob_graph(graph, item, num_windows, min_freq):
    """ 
    Parameters
    ----------
    graph : Dict[int, Dict[int, float]]
    item : Tuple[int, Dict[int, float]]
        vocab id : count co-occurrences
    num_windows: int
        total number of windows in corpus
    min_freq : int 
        lower bount to consider co-occurrence counts

    Returns
    -------
    graph: Dict[int, Dict[int, float]]
    """
    graph[item[0]] = create_prob_graph(item[1], num_windows, min_freq)
    return graph

def iload_id_and_graph(paths):
    """ iterative loader for name and .pkl graph

    Parameters
    ----------
    paths : List[str]
        filepaths

    Yield
    ------
    str
        name of graph
    dict
        mapping of key,value
    """
    for f in paths:
        yield int(f.split('/')[-1].rstrip('.pkl')), pickle.load(open(f, 'rb'))

def load_joint_prob_graph(graph_dir, num_windows, min_freq,
                          shortlist=[], existing_graph={}):
    """ Load and build probability graphs from count graphs

    Parameters
    ----------
    graph_dir : str
        path to directory cotaining only .pkl joint graphs
    num_windows: int
        total number of windows in corpus
    min_freq : int 
        lower bount to consider co-occurrence counts
    shortlist : List[int]
        list of shortlisted vocab ids
    existing_graph : Dict[int, Dict[int, float]]

    Returns
    -------
    joint co-occurence probability graph : Dict[int, Dict[int, float]]
    """
    if len(shortlist) == 0:
        paths = [f"{graph_dir}/{f}" for f in os.listdir(graph_dir)]
    else:
        paths = [f"{graph_dir}/{f}.pkl" for f in shortlist if os.path.exists(
            os.path.join(graph_dir, f"{f}.pkl"))]
    graph = reduce(partial(aggregate_prob_graph, num_windows=num_windows,
                           min_freq=min_freq), iload_id_and_graph(paths),
                   existing_graph)
    return graph

def create_graph_with(score_func, co_occ, occ, smooth=True, shortlist=[]):
    """ create a scored graph from probability graphs

    Parameters
    ----------
    score_func : function -> float
        Defined as f(p_a_b, p_a, p_b, **kwargs)
        Function takes in joint co-occ and prior word probabilities
        Generates a graph based on your scoring function
    co_occ: Dict[int, Dict[int, float]]
        joint co-occurence probability graph
    occ: Dict[int, float]
        prior probability graph
    smooth : bool
        Decides the use of epsilon=1e-12 or default if required
    shortlist : List[int]
        list of shortlisted vocab ids

    Return
    ------
    scored co-occurrence graph : Dict[int, Dict[int, float]]

    Benchmark
    ---------
    ~33 minutes to calculate 40K Wikipedia graphs using AMD EPYC 7502 @ 2.50GHz
    """
    if len(shortlist) > 0:
        co_occ = {s: {k2: v for k2, v in co_occ[s].items()
                      if k2 in shortlist}
                  for s in shortlist}
    graph = {
        i: {j: score_func(co_occ[i][j] if j in co_occ[i] else 0,
                          occ[i], occ[j], smooth=smooth)
            for j in co_occ.keys()}
        for i in co_occ.keys()
    }
    return graph

def get_total_windows(histogram_path, window_size):
    """ Calculate number of sliding windows based on document lengths

    Parameters
    ----------
    histogram_path : str
        path of histogram file, csv format
    window_size : int
        hyper-parameter for counting, sliding window size

    Return
    ------
    num_windows: int
        total number of windows in corpus

    Note
    ----
    as per Palmetto (Röder et al., 2015)
    """
    histogram = pd.read_csv(histogram_path, header=None).values
    if window_size == 0:
        return histogram[:, 1].sum()
    return (histogram[:, 1] * np.maximum([1], histogram[:, 0] - (window_size - 1))).sum()

def load_graph(path):
    """ load .pkl dictionary graph

    Parameters
    ----------
    path : str
        filepath

    Return
    ------
    dict
        mapping of key,value
    """
    return pickle.load(open(path, 'rb'))

def single_count_setup(histogram_path, single_count_path,
                       window_size, min_freq):
    """ 
    Parameters
    ----------
    histogram_path : str
        path of histogram file, csv format
    single_count_path : str
        path to .pkl containing prior counts
    window_size : int 
        hyper-parameter for counting, sliding window size
    min_freq : int 
        lower bount to consider co-occurrence counts
    Returns
    -------
    num_windows: int
        total number of windows in corpus
    single_prob: Dict[int,float]
        prior probability graph
    """
    num_windows = get_total_windows(histogram_path, window_size)
    single_prob = create_prob_graph(load_graph(
        single_count_path), num_windows, min_freq)
    return num_windows, single_prob
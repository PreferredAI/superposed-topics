# superposed-topics

## Disentangling Transformer Language Models as Superposed Topic Models, EMNLP'23

This repo serves as the accompany code to the paper. The pursuit of interpretability in most alignment research is very similar to that of topic modelling. Our work applies the methodologies used in topic modelling to 'explain' neurons in Transformer Language Models, discovering superpositions in individual neurons.

### Summary

For each neuron and its path of activation, we:

1. Project to a corpus -- From the activated tokens, we contruct a word pool w.r.t the corpus

```
accounts administration agreement artist asks attributed bass basses bassist beat beatings bird blocks bloomberg book booked booking bookings boxes broad broadband cat circuit client clients commented connecting consist consists crowd cursed curses direct doesn ease entire facebook finding finds fragments gentle guest habit helps historians histories impression includes increases intention intermediate joins leave leaves lift loves loving moment mounted net occupied officials oil organ pick picking platforms plays plus preserve preserved prevent prevented protest protested protesting provides registers relationships reports sales sam says series shows size skeletal sleep sleeping sons sparked stop strikes students tells threatened touch track tracked treat treaties tries users uses valuable victim videos views villa vista voice witness witnessed witnesses
```

2. Shortlist plausible word sets -- Using our _star_ heuristic, we shortlist disjoint word sets from the pools
```
book boot box certain current display displays editing intended introduced like menu operating possible provides registry reviewed save secret shows similar single special target temporary folder (26)
absolute branch computation continuous direct discussed discussion exact fact geometry introduce note property showed showing speaking straightforward subject true work algebraic (21)
...
```

3. Exact solving -- Find an optimistic set of words that can be used for comparisons and 'infer' the role(s) of the neuron (LLaMA-13B 3-226)
```
certain intended menu boot possible display displays operating folder target 0.0738
note geometry true continuous straightforward absolute computation fact exact algebraic 0.0876
...
```

### Additional Resources

[Wikipedia-V40K](https://static.preferred.ai/jiapeng/wiki.tar.gz) (18.1 GB), 40K count graphs of vocabulary and some entities

Our [results](https://static.preferred.ai/jiapeng/gpt2_900.tar.gz) (3.9 GB) from mining GPT-2 via projecting to Wikipedia-V40K

Our [results](https://static.preferred.ai/jiapeng/llama_900.tar.gz) (4.3 GB) from mining LLaMA via projecting to Wikipedia-V40K

Results structured as {size}/{mode}/{layer}\_{id}\_{polarity}{file}, empty files means nothing extracted

* mode: i/ii/iii/iiii (Hi-C, Hi-I, Lo-C, Lo-I respectively)
* size: e.g. 1558M, 7B
* layer/id: int
* polarity: 'pos' or 'neg'
* file: pool ('.pkl') or '_topics.csv' or '_isets.csv'

### Resources used

Jay Mody's [picoGPT](https://github.com/jaymody/picoGPT) for GPT-2 (As we only need the forward pass, it is easier to tinker with small code)

Meta's [LLaMA v1](https://github.com/facebookresearch/llama/tree/llama_v1) (Note: forward pass modifications to it falls under GNU GPL v3)

Diversity-contrained Greedy (EMNLP'22) from https://github.com/PreferredAI/ReIntNTM

PreferredAI topic-metrics (ACL'23) from https://github.com/PreferredAI/topic-metrics/

For convenience, we include the [CVXPY](https://www.cvxpy.org/index.html) implementation that serves as an interface for [different](https://www.cvxpy.org/install/#install-with-cvxopt-and-glpk-support) solvers (Gurobi, CPLEX, etc.). For non-commercial solvers, we recommend [SCIP](https://www.scipopt.org/index.php#welcome)'s [PySCIPOpt](https://github.com/scipopt/PySCIPOpt/tree/master), [link](https://dl.acm.org/doi/abs/10.1145/3585516) to report, Apache 2.0 license.

### Citation

If you had found the resources helpful, we'd appreciate a citation!

```
@inproceedings{lim-lauw-2023-disentangling,
    title = "Disentangling Transformer Language Models as Superposed Topic Models",
    author = "Lim, Jia Peng  and
      Lauw, Hady",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics"
}
```

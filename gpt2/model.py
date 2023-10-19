'''
Original Code from:
https://jaykmody.com/blog/gpt-from-scratch/
https://github.com/jaymody/picoGPT
MIT license
'''

from functools import partial
from jax import vmap
from gpt2.utils import load_encoder_hparams_and_params
import jax.numpy as np
import numpy as onp
from tqdm import tqdm


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(
        x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10
    out_heads = np.hstack([attention(q, k, v, causal_mask)
                          for q, k, v in zip(*qkv_heads)])
    x = linear(out_heads, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x += mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x += ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    print(x.shape)
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def generate(inputs, params, n_head, n_tokens_to_generate):
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs = np.append(inputs, [next_id])
    return list(inputs[len(inputs) - n_tokens_to_generate:])


'''
Modifications
For gpt-2 mods, we do not require positional embeddings wpe or text inputs
'''


def ffn_mod(x, c_proj):
    return linear(x, **c_proj)


def transformer_block_mod(x, mlp, attn, ln_1, ln_2, n_head):
    return ffn_mod(x[None, :], mlp['c_proj'])


def ffn2(x, c_fc, c_proj):
    mlp = gelu(linear(x, **c_fc))
    return linear(mlp, **c_proj), mlp


def gpt2_mod_i(x, wte, blocks, ln_f, n_head, inject):
    x = transformer_block_mod(x, **blocks[inject], n_head=n_head)
    for j in range(inject+1, len(blocks)):
        x = transformer_block(x, **blocks[j], n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def gpt2_mod_ii(x, wte, blocks, ln_f, n_head, inject):
    x = transformer_block_mod(x, **blocks[inject], n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def gpt2_mod_iii(x, wte, blocks, ln_f, n_head, inject):
    for j in range(inject, len(blocks)):
        x = transformer_block(x, **blocks[j], n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def gpt2_mod_iv(x, wte, blocks, ln_f, n_head, inject):
    if inject < len(blocks):
        x = transformer_block(x, **blocks[inject], n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T


def gpt2_mod_zstm(inputs, wte, wpe, blocks, n_head, votes=5):
    '''
    Returns
        Indices of top activated neurons in each low-dim. layer
    '''
    x = wte[inputs] + wpe[range(len(inputs))]
    ctx = []
    for j in range(0, len(blocks)):
        x = transformer_block(x, **blocks[j], n_head=n_head)
        ctx.append(onp.array(x))
    return onp.argsort(onp.array(ctx))[:, :, -votes:]


'''
Main methods
'''


def gpt2_crawl(model_dir, model_size, mode, layers=[], batch_size=32):
    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, model_dir)

    if mode == 1 or mode == 2:
        MLP = params['blocks'][0]['mlp']['c_proj']['w'].shape[0]
        batches = np.eye(MLP, MLP)
    else:
        HDN = params['blocks'][0]['ln_1']['b'].shape[0]
        batches = np.eye(HDN, HDN)
    
    batches = batches[:, None, :]
    max_layer = hparams["n_layer"]
    if len(layers) == 0:
        layers = range(max_layer)
    else:
        layers = [layer for layer in layers if layer < max_layer]


    func_options = {1: gpt2_mod_i, 2: gpt2_mod_ii,
                    3: gpt2_mod_iii, 4: gpt2_mod_iv}
    if mode not in func_options:
        raise Exception("mode requires Integer [1,4]")
    else:
        func = func_options[mode]

    output_in_layer = {}
    for i in layers:
        final_logits = []
        batch_func = vmap(partial(func, wte=params['wte'],
                                  blocks=params['blocks'],
                                  ln_f=params['ln_f'],
                                  n_head=hparams["n_head"],
                                  inject=i))
        for j in tqdm(range(0, len(batches), batch_size)):
            batch_input = batches[j:j+batch_size]
            final_logits.append(onp.array(batch_func(batch_input)))
        output_in_layer[i] = onp.squeeze(onp.vstack(final_logits))

    return output_in_layer


def gpt2_get_votes(model_dir, model_size, corpus, votes):
    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, model_dir)
    input_ids_list = []
    for prompt in corpus:
        input_ids_list.append(encoder.encode(prompt)[:1024])
    input_ids_arr = onp.array(input_ids_list, dtype=object)
    counts = []
    for j in tqdm(range(0, len(input_ids_list))):
        counts.append(gpt2_mod_zstm(input_ids_arr[j], wte=params['wte'],
                                    wpe=params['wpe'], blocks=params['blocks'],
                                    n_head=hparams["n_head"], votes=votes))
    return counts

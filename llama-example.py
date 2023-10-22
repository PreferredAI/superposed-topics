from typing import Tuple
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", dest='mode', type=int, metavar='<int>')
parser.add_argument("--size", dest='model_size', type=str, metavar='<str>', default="7B")
parser.add_argument("--dest", dest='dest_dir', type=str, metavar='<str>', default="./outputs_llama")
parser.add_argument("--layers", dest='layers', type=str, metavar='<str>', default="")
args = parser.parse_args()

"""
Example commands
torchrun --nproc_per_node 1 llama-example.py --mode 3 --size 7B --layers 1,2
torchrun --nproc_per_node 2 llama-example.py --mode 1 --size 13B --layers 3
"""

import torch
import time
import json
import pickle
from tqdm import tqdm
import numpy as np

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()
ckpt_dir = f"/data/LLaMA/{args.model_size}"
tokenizer_path = "/data/LLaMA/tokenizer.model"
tokenizer = Tokenizer(model_path=tokenizer_path)

start_time = time.time()
checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

ckpt_path = checkpoints[0]
print("Loading...")
checkpoint = torch.load(ckpt_path, map_location="cpu")
with open(Path(ckpt_dir) / "params.json", "r") as f:
    params = json.loads(f.read())

max_seq_len: int = 10
max_batch_size: int = 32
model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
)
tokenizer = Tokenizer(model_path=tokenizer_path)
model_args.vocab_size = tokenizer.n_words
torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)
model.load_state_dict(checkpoint, strict=False)
print(f"Loaded in {time.time() - start_time:.2f} seconds")

if args.mode == 3 or args.mode == 4:
    ctx_dim = model_args.dim
    eye_input = torch.eye(ctx_dim).half()[:,None,:]
    print('CTX_DIM:', ctx_dim)
elif args.mode == 1 or args.mode == 2:
    hidden_dim = model.layers[0].feed_forward.hidden_dim
    eye_input = torch.eye(hidden_dim).half()[:,None,:]
    print('HID_DIM:', hidden_dim)
else:
    raise Exception("mode requires Integer [1,4]")


func_options = {1: model.single_neuron_mod_i, 
                2: model.single_neuron_mod_ii,
                3: model.single_neuron_mod_iii, 
                4: model.single_neuron_mod_iv}

output_dir = f'{args.dest_dir}/{args.model_size}'
os.makedirs(output_dir, exist_ok=True)

if len(args.layers) == 0:
    layers = range(model_args.n_layers)
else:
    layers = [int(layer) for layer in args.layers.split(',') 
              if int(layer) < model_args.n_layers]

print('mode:', args.mode)
for start_layer in layers:
    results = []
    for i in tqdm(range(0, len(eye_input), max_batch_size)):
        results.append(np.squeeze(func_options[args.mode](
            eye_input[i:i+max_batch_size,:,:].cuda(),start_layer).cpu().numpy()
            ))
    results = np.vstack(results)
    pickle.dump(results, open(f'{output_dir}/{"i"*args.mode}_{start_layer}.pkl','wb'))
    print(f'{output_dir}/{"i"*args.mode}_{start_layer}.pkl', results.shape)
print('completed.')

from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
import numpy as np
from generation import generate_inv, generate_gum, generate_rnd
from sampling import transform_sampling, transform_key_func, transform_Y
from sampling import gumbel_sampling, gumbel_key_func, gumbel_Y
import argparse

results = defaultdict(dict)
## We only generate the data.

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="gumbel",type=str,choices=["gumbel", "transform", "raw"])
parser.add_argument('--task',default="QA",type=str)
parser.add_argument('--access', default="", type=str, help='Huggingface access token')
parser.add_argument('--model',default="facebook/opt-13b",type=str)
parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=4,type=int)
parser.add_argument('--temperature',default=1.0,type=float)

parser.add_argument('--batch_size',default=1,type=int)
parser.add_argument('--seed_way',default="skipgram_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)

parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)

parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--truncate_vocab',default=8,type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)
print(args)

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access)
if tokenizer.pad_token is None:     # for llama we need to set padding tokens manually
    tokenizer.pad_token = tokenizer.bos_token

model = AutoModelForCausalLM.from_pretrained(args.model, token=args.access).to(device)
model_name = args.model.split('/')[1]

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
print(f'Loaded the model (t = {time()-t0} seconds)')
print()
print("The vocabulary size is", vocab_size)
print()
if args.task == 'QA':
    dataset = load_dataset("rexarski/eli5_category", split="train", trust_remote_code=True, streaming=True)
elif args.task == 'CS':
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

T = args.T                                    # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
prompt_tokens = args.prompt_tokens            # minimum prompt length
new_tokens = args.m                           # number of tokens to generate
buffer_tokens = args.buffer_tokens 


if args.method == "transform":
    generate_null = False
    generate_watermark = lambda prompt : generate_inv(model,
                                                  prompt,
                                                  vocab_size,
                                                  new_tokens,
                                                  transform_key_func,
                                                  transform_sampling,
                                                  transform_Y,
                                                  key=args.seed,
                                                  c=args.c,
                                                  seeding_scheme=args.seed_way,
                                                  temperature=args.temperature)

elif args.method == "gumbel":
    generate_null = False
    generate_watermark = lambda prompt : generate_gum(model,
                                                  prompt,
                                                  vocab_size,
                                                  new_tokens,
                                                  gumbel_key_func,
                                                  gumbel_sampling,
                                                  gumbel_Y,
                                                  key=args.seed,
                                                  c=args.c,
                                                  seeding_scheme=args.seed_way,
                                                  temperature=args.temperature)
elif args.method == "raw":
    generate_null = True                                                                  
else:
    raise

ds_iterator = iter(dataset)

t1 = time()
prompts = []
itm = 0
while itm < T:
    example = next(ds_iterator)
    if args.task == 'CS':
        text = example['text']
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        if len(tokens) < prompt_tokens + new_tokens:
            continue
        prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
    elif args.task == 'QA':
        #when task is QA
        text = example['title']
        instruct = 'Answer the following question: '
        text = instruct + text

        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        if len(tokens) > prompt_tokens:
            continue
        prompt = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=prompt_tokens, padding='max_length', padding_side='left')[0]
    prompts.append(prompt)
    itm += 1
prompts = torch.vstack(prompts)
results['prompts'] = copy.deepcopy(prompts)

if not generate_null:
    ## If we need to generate watermarked samples

    if args.method == "transform":
        watermarked_samples = []
        generated_Ys = []
        generated_Us = []
        generated_etas = []
        generated_top_probs = []
        for batch in tqdm(range(n_batches)):
            idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))

            generated_tokens, Ys, Us, etas, top_probs = generate_watermark(prompts[idx])
            watermarked_samples.append(generated_tokens[:,prompt_tokens:])
            generated_Ys.append(Ys.squeeze())
            generated_Us.append(Us.squeeze())
            generated_etas.append(etas.squeeze())
            generated_top_probs.append(top_probs.squeeze())

        watermarked_samples = torch.vstack(watermarked_samples)
        generated_Ys = torch.vstack(generated_Ys)
        generated_Us = torch.vstack(generated_Us)
        generated_etas = torch.vstack(generated_etas)
        generated_top_probs = torch.vstack(generated_top_probs)

        ## Save generated texts and pivotal statsitics
        results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
        results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
        results['watermark']['Us'] = copy.deepcopy(generated_Us)
        results['watermark']['etas'] = copy.deepcopy(generated_etas)
        results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)

    elif args.method == "gumbel":
        watermarked_samples = []
        generated_Ys = []
        generated_top_probs = []
        for batch in tqdm(range(n_batches)):
            idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))

            generated_tokens, Ys, top_probs = generate_watermark(prompts[idx])
            watermarked_samples.append(generated_tokens[:,prompt_tokens:])
            generated_Ys.append(Ys.squeeze())
            generated_top_probs.append(top_probs.squeeze())

        ## Save generated texts and pivotal statsitics
        watermarked_samples = torch.vstack(watermarked_samples)
        generated_Ys = torch.vstack(generated_Ys)
        generated_top_probs = torch.vstack(generated_top_probs)

        results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
        results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
        results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)
    else:
        raise ValueError(f"This watermark method is not implemented: {args.method}.")
    
    print(f'Generated watermarked samples in (t = {time()-t1} seconds)')

    ## Name the experiment with configuration
    exp_name = f"results/{model_name}-{args.method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{args.temperature}-{args.task}.pkl"
    pickle.dump(results,open(exp_name,"wb"))

else:
    ## If we need to generate unwatermarked samples 
    ## We don't adjust the temperature parameter here
    null_samples = []
    for batch in tqdm(range(n_batches)):
        idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))
        null_samples.append(generate_rnd(prompts[idx],new_tokens+buffer_tokens,model)[:,prompt_tokens:])

    null_samples = torch.vstack(null_samples)
    results['null']['tokens'] = copy.deepcopy(null_samples)

    print(f'Generated samples in (t = {time()-t1} seconds)')

    ## Name the experiment with configuration
    exp_name = f"results/{model_name}-raw-m{args.m}-T{args.T}.pkl"
    pickle.dump(results,open(exp_name,"wb"))

import gc
import torch
import os
import json
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from datasets import load_dataset, Dataset
import copy
import numpy as np
from tqdm import tqdm
import pickle
import argparse

results = defaultdict(dict)
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="SynthID",type=str)
parser.add_argument('--task',default="QA",type=str)
parser.add_argument('--access', default="", type=str, help='Huggingface access token')
parser.add_argument('--model',default="facebook/opt-13b",type=str) # facebook/opt-1.3b, meta-llama/Llama-3.1-8B
parser.add_argument('--temp',default=1.0,type=float)
parser.add_argument('--top_k',default=100,type=int) # we use top_k sampling
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=60,type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)
print(args)

T = args.T                                    # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
prompt_tokens = args.prompt_tokens            # minimum prompt length
new_tokens = args.m                           # number of tokens to generate
buffer_tokens = args.buffer_tokens 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access)
if tokenizer.pad_token is None:     # for llama we need to set padding tokens manually
    tokenizer.pad_token = tokenizer.eos_token

transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained(args.model, token=args.access).to(device),
                          tokenizer=tokenizer,
                          vocab_size=len(tokenizer),
                          device=device,
                          max_new_tokens=new_tokens+buffer_tokens,
                          min_length=new_tokens+buffer_tokens,
                          do_sample=True,
                          temperature=args.temp,
                          top_k=args.top_k)

if args.task == 'QA':
    dataset = load_dataset("rexarski/eli5_category", split="train", trust_remote_code=True, streaming=True)
elif args.task == 'CS':
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    # with open('dataset/c4/c4.json', 'r') as f:
    #     lines = f.readlines()
    #     data = [json.loads(line) for line in lines]
    # dataset = Dataset.from_list(data)


print('Preparing the prompts...')
ds_iterator = iter(dataset)
prompts = []
for i in range(T):
    example = next(ds_iterator)
    if args.task == 'CS':
        prompt = example['text']
    elif args.task == 'QA':
        prompt = example['title']
        instruct = 'Answer the following question: '
        prompt = instruct + prompt
    prompts.append(prompt)
results['prompts'] = prompts

print('Loading the watermarking algorithm...')
myWatermark = AutoWatermark.load(args.method, algorithm_config=f'config/{args.method}.json', transformers_config=transformers_config)

watermarked_texts = []
watermarked_tokens = []
generated_gs = []
print('Generating the watermarked texts...')
for batch in tqdm(range(n_batches)):
    batch_prompts = prompts[batch*args.batch_size:(batch+1)*args.batch_size]
    batch_watermarked_texts, batch_watermarked_tokens, batch_generated_gs = myWatermark.generate_watermark(batch_prompts)
    watermarked_texts.extend(batch_watermarked_texts)
    watermarked_tokens.extend(batch_watermarked_tokens.cpu())
    generated_gs.extend(batch_generated_gs.cpu())

results['watermarked_texts'] = watermarked_texts
results['watermarked_tokens'] = watermarked_tokens
results['generated_gs'] = generated_gs

print('Saving the results...')
results_file = f"results/full_results/{args.method}_{args.task}_{args.model.split('/')[-1]}_{args.T}_{args.m}_temp-{args.temp}.pkl"
# make sure the directory exists
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f'Results saved to {results_file}')
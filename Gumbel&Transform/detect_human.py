import os
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
from attack_utils import get_score_inverse, get_score_gumbel
from sampling import transform_key_func, transform_Y, gumbel_key_func, gumbel_Y
from detect_utils import ConfigManager, StatisticalTests, PhiDivergenceTest
import argparse

results = defaultdict(dict)
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="transform",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
parser.add_argument('--task',default="CS",type=str)
parser.add_argument('--access', default="", type=str, help='Huggingface access token')
parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=4,type=int)
parser.add_argument('--seed_way',default="skipgram_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--drop_repeat',default=390,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--truncate_vocab',default=8,type=int)
parser.add_argument('--alpha', default=0.01, type=float)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)
print(args)

# fix the random seed for reproducibility
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access)
if tokenizer.pad_token is None:     # for llama we need to set padding tokens manually
    tokenizer.pad_token = tokenizer.bos_token

vocab_size_dic = {'meta-llama/Llama-3.1-8B': 128256,
                  'facebook/opt-13b': 50272,
                  'facebook/opt-1.3b': 50272 }
vocab_size = vocab_size_dic[args.model]
eff_vocab_size = vocab_size - args.truncate_vocab
print("The vocabulary size is", vocab_size)

if args.task == 'QA':
    dataset = load_dataset("rexarski/eli5_category", split="train", trust_remote_code=True, streaming=True)
elif args.task == 'CS':
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

# define the function to compute pivotal statistics
if args.method == "transform":
    get_score = lambda text, prompt : get_score_inverse(text, prompt, tokenizer, vocab_size, transform_key_func, transform_Y, args.seed, args.c, args.seed_way, input_type='token')

elif args.method == "gumbel":
    get_score = lambda text, prompt : get_score_gumbel(text, prompt, tokenizer, vocab_size, gumbel_key_func, gumbel_Y, args.seed, args.c, args.seed_way, input_type='token')
else:
    raise ValueError(f"Unknown watermark: {args.method}")

T = args.T                                    # number of prompts/generations
prompt_tokens = args.prompt_tokens            # minimum prompt length
new_tokens = args.m                           # number of tokens to generate
buffer_tokens = args.buffer_tokens 

ds_iterator = iter(dataset)
y_list = []
itm = 0
pbar = tqdm(total=T, desc="Processing data")
while itm < T:
    example = next(ds_iterator)
    if args.task == 'CS':
        text = example['text']
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        if len(tokens) < prompt_tokens + new_tokens:
            continue
        prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
        human_text = tokens[-new_tokens:]
    elif args.task == 'QA':
        text = example['title']
        instruct = 'Answer the following question: '
        text = instruct + text
        human_answer = max(example['answers']['text'], key=len)
        human_token = tokenizer.encode(human_answer, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        # For QA dataset, we want the question to be shorter than the prompt requirement
        if len(tokens) > prompt_tokens or len(human_token) < new_tokens:
            continue
        prompt = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=prompt_tokens, padding='max_length', padding_side='left')[0]
        human_text = human_token[:new_tokens]
    # get pivotal score after attack
    y = get_score(human_text, prompt)
    assert len(y) == new_tokens, f"Expected the length of pivotal statistics to be {new_tokens}, but got {len(y)}."
    if args.drop_repeat:
        if len(set(y)) < args.drop_repeat:
            continue
    # y_list.append(y)
    y_list.append(list(set(y))[:args.drop_repeat])
    itm += 1
    pbar.update(1)
pbar.close()
print("Finish computing pivotal statistics")

# adversarial edits
data = np.array(y_list)

# get the score
save_name = f"human_{args.task}_{args.method}_drop{args.drop_repeat}.pkl"
result_dic = {}
config_manager = ConfigManager()
if args.alpha != 0.05:
    for test_name in config_manager.configs.keys():
        config_manager.set_config(test_name, alpha=args.alpha)
if args.method == 'transform':
    config_manager.set_config('sum_based_its', delta=0.5)
    save_name = f'{save_name[:-4]}_d0.5.pkl'
    data = -data
    stat_tests = StatisticalTests(data, config_manager, 'ITSEdit')
    tests_list = ['sum_based_its', 'its_neg', 'phi_divergence', 'kuiper', 'kolmogorov', 'anderson',
                 'cramer', 'watson', 'neyman', 'chi_squared', 'rao', 'greenwood']
else:
    stat_tests = StatisticalTests(data, config_manager, 'EXP')
    tests_list = ['sum_based', 'log', 'ars', 'phi_divergence', 'kuiper', 'kolmogorov', 'anderson',
                 'cramer', 'watson', 'neyman', 'chi_squared', 'rao', 'greenwood']
for test in tqdm(tests_list):
    results, _ = stat_tests.perform_statistical_test(test)
    result_dic[test] = results

save_path = 'results/scores/human'
# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)
with open(f"{save_path}/{save_name}", 'wb') as f:
    pickle.dump(result_dic, f)

with open(f"{save_path}/Y_human_{args.task}_{args.method}_drop{args.drop_repeat}.pkl", 'wb') as f:
    pickle.dump(y_list, f)

print('result saved')
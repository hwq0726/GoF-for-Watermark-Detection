import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import nltk
import numpy as np
import argparse
import pickle
from attack_utils import get_score_inverse, SynonymSubstitution, DipperParaphraser, WordDeletion
from sampling import transform_key_func, transform_Y, gumbel_key_func, gumbel_Y
nltk.download('punkt_tab')

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--task',default="QA",type=str)
parser.add_argument('--method',default="transform",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
parser.add_argument('--access', default="", type=str, help='Huggingface access token')
parser.add_argument('--temperature',default=1.0,type=float)
parser.add_argument('--attacker', default="Dipper", type=str)
parser.add_argument('--Y_min_len', default=300, type=int, help='The min length of Y') #250 for deletion

# Dipper paraphraser settings
parser.add_argument('--order', default=20, type=int)        # take values in [0, 20, 40, 60, 80, 100]
parser.add_argument('--lex', default=20, type=int)     # take values in [0, 20, 40, 60, 80, 100]
# Deletion and Substitution settings
parser.add_argument('--attacker_rate', default=0.3, type=float) 
args = parser.parse_args()
print(args)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access)

vocab_size_dic = {'meta-llama/Llama-3.1-8B': 128256,
                  'facebook/opt-13b': 50272,
                  'facebook/opt-1.3b': 50272 }
vocab_size = vocab_size_dic[args.model]

# define the function to compute pivotal statistics
if args.method == "transform":
    get_score = lambda text, prompt : get_score_inverse(text, prompt, tokenizer, vocab_size, transform_key_func, transform_Y, 15485863, 4, "skipgram_prf")

elif args.method == "gumbel":
    get_score = lambda text, prompt : get_score_gumbel(text, prompt, tokenizer, vocab_size, gumbel_key_func, gumbel_Y, 15485863, 4, "skipgram_prf")
else:
    raise ValueError(f"Unknown watermark: {args.method}")

# Load the attacker
if args.attacker == 'WordDeletion':
    attacker = WordDeletion(ratio=args.attacker_rate)
elif args.attacker == 'SynonymSubstitution':
    attacker = SynonymSubstitution(ratio=args.attacker_rate)
elif args.attacker == 'Dipper':
    attacker = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('google/t5-v1_1-xxl'),
                                 model=T5ForConditionalGeneration.from_pretrained('kalpeshk2011/dipper-paraphraser-xxl'),
                                 lex_diversity=args.lex, order_diversity=args.order,
                                 sent_interval=1, max_new_tokens=400, do_sample=True, top_p=0.75, top_k=None)

# Get watermarked text (use the default seed and hash scheme)
data_dir = f"results/{args.model.split('/')[1]}-{args.method}-c4-m400-T1000-skipgram_prf-15485863-temp{args.temperature}-{args.task}.pkl"
with open(data_dir, 'rb') as f:
    data = pickle.load(f)

# prompts: tensor with shape 1000*50; texts: tensor with shape 1000*400
prompts_token = data['prompts']
texts_token = data['watermark']['tokens']

# Attack the watermark
y_list = []
for index in tqdm(range(len(texts_token))):
    prompt_token = prompts_token[index]
    text_token = texts_token[index]

    # decoed into text, skip specail tokens will cause the loss of several siganls but it's ok for long enough text
    prompt = tokenizer.decode(prompt_token, skip_special_tokens=True)
    text = tokenizer.decode(text_token, skip_special_tokens=True)
    text_attack = attacker.edit(text, prompt)
    if len(text_attack.split()) == 0:
        continue
    # get pivotal score after attack
    y = get_score(text_attack, prompt_token)
    if len(y) < args.Y_min_len:
        continue
    y = y[:args.Y_min_len]
    y_list.append(y)

if args.attacker in ['WordDeletion', 'SynonymSubstitution']:
    save_path = f"results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temperature}_{args.method}_{args.task}_{args.attacker}-{args.attacker_rate}.pkl" 
elif args.attacker == 'Dipper':
    save_path = f"results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temperature}_{args.method}_{args.task}_{args.attacker}-{args.order}-{args.lex}.pkl" 

os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'wb') as f:
    pickle.dump(y_list, f)

print(f'The length of y_list is {len(y_list)}')


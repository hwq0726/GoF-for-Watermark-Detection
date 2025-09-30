import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import nltk
import numpy as np
import argparse
import pickle
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.text_editor import WordDeletion, SynonymSubstitution, DipperParaphraser

nltk.download('punkt_tab')

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--task',default="QA",type=str)
parser.add_argument('--method',default="SynthID",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
parser.add_argument('--access', default="", type=str, help='Huggingface access token')
parser.add_argument('--temp',default=1.0,type=float)
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
assert device == "cuda", "the device config should be the same as the one used in generating the data"
print(f'Using device: {device}')

# Load the watermark algorith
tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access)
if tokenizer.pad_token is None:     # for llama we need to set padding tokens manually
    tokenizer.pad_token = tokenizer.eos_token

transformers_config = TransformersConfig(model=None,
                          tokenizer=tokenizer,
                          vocab_size=len(tokenizer),
                          device=device,)

algorithm_name = 'SynthID'
myWatermark = AutoWatermark.load(f'{algorithm_name}', algorithm_config=f'config/{algorithm_name}.json', transformers_config=transformers_config)

# Load the attacker
if args.attacker == 'WordDeletion':
    attacker = WordDeletion(ratio=args.attacker_rate)
elif args.attacker == 'SynonymSubstitution':
    attacker = SynonymSubstitution(ratio=args.attacker_rate)
elif args.attacker == 'Dipper':
    attacker = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('google/t5-v1_1-xxl', token=args.access),
                                 model=T5ForConditionalGeneration.from_pretrained('kalpeshk2011/dipper-paraphraser-xxl').to(device),
                                 lex_diversity=args.lex, order_diversity=args.order,
                                 sent_interval=1, max_new_tokens=400, do_sample=True, top_p=0.75, top_k=None)
else:
    raise ValueError('Invalid attacker name')

# Get watermarked text
root_path = 'set you root path here'
data_dir = f"{root_path}/results/full_results/SynthID_{args.task}_{args.model.split('/')[1]}_1000_400_temp-{args.temp}.pkl"
with open(data_dir, 'rb') as f:
    data = pickle.load(f)
watermarked_texts = data['watermarked_texts']
full_prompts = data['prompts']
print(f'The length of watermarked_text is {len(watermarked_texts)}')

encoded_prompts = tokenizer(full_prompts, return_tensors="pt", add_special_tokens=True, padding='max_length', padding_side='left', truncation=True, max_length=50)
prompts = tokenizer.batch_decode(encoded_prompts['input_ids'], skip_special_tokens=True)

# Attack the watermark
g_value_list = []
pivotal_list = []
edited_texts = []
for index in tqdm(range(len(watermarked_texts))):
    prompt = prompts[index]
    text = watermarked_texts[index]
    watermarked_text = text[len(prompt):]
    # decoed into text, skip specail tokens will cause the loss of several siganls but it's ok for long enough text
    
    text_attack = attacker.edit(watermarked_text, prompt)
    if len(text_attack.split()) == 0:
        continue
    # get pivotal score after attack
    edited_text = prompt + text_attack
    pivotal, g_value = myWatermark.get_pivotal(edited_text)
    # y = get_score_inverse(text, prompt_token, tokenizer, vocab_size, transform_key_func, transform_Y, 15485863, 4, "skipgram_prf")
    if pivotal.shape[0] < args.Y_min_len:
        continue
    pivotal = pivotal[:args.Y_min_len]
    pivotal_list.append(pivotal)
    g_value_list.append(g_value)
    edited_texts.append(edited_text)

if args.attacker in ['WordDeletion', 'SynonymSubstitution']:
    save_path = f"{root_path}/results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temp}_{args.method}_{args.task}_{args.attacker}-{args.attacker_rate}.pkl"
    text_save_path = f"{root_path}/results/attack/{args.attacker}/texts_{args.model.split('/')[1]}_{args.temp}_{args.method}_{args.task}_{args.attacker}-{args.attacker_rate}.pkl" 
elif args.attacker == 'Dipper':
    save_path = f"{root_path}/results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temp}_{args.method}_{args.task}_{args.attacker}-{args.order}-{args.lex}.pkl" 
    text_save_path = f"{root_path}/results/attack/{args.attacker}/texts_{args.model.split('/')[1]}_{args.temp}_{args.method}_{args.task}_{args.attacker}-{args.order}-{args.lex}.pkl"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
result = {'pivotal': pivotal_list, 'g_value': g_value_list}
with open(save_path, 'wb') as f:
    pickle.dump(result, f)

with open(text_save_path, 'wb') as f:
    pickle.dump(edited_texts, f)

# with open('results/test.pkl', 'wb') as f:
#     pickle.dump(y_list, f)

print(f'The length of pivotal_list is {len(pivotal_list)}')


import os
import pickle
import torch
import numpy as np
from scipy.stats import chi2
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from detect_utils import ConfigManager, StatisticalTests, PhiDivergenceTest

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--task', default='CS', type=str)
parser.add_argument('--no_repeat', default=None, type=int)
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)        # change needed
parser.add_argument('--alpha', default=0.01, type=float)
# parser.add_argument('--human', default=False, type=bool)
parser.add_argument('--method', default="transform", type=str)
parser.add_argument('--prompt_len', default=50, type=int)
parser.add_argument('--temperature', default=0.1, type=float)
# WordDeletion and SynonymSubstitution settings
parser.add_argument('--attacker', default=None, type=str)     # change needed
parser.add_argument('--attacker_rate', default=0.3, type=float)
# Dipper paraphraser settings
parser.add_argument('--order', default=60, type=int)        # take values in [0, 20, 40, 60, 80, 100]
parser.add_argument('--lex', default=60, type=int)
# Detection method setting
parser.add_argument(
        '--detection',
        nargs='+',  # Accepts multiple values
        default=['sum_based', 'log', 'ars', 'phi_divergence', 'kuiper', 'kolmogorov', 'anderson',
                 'cramer', 'watson', 'neyman', 'chi_squared', 'rao', 'greenwood'],
        help="List of algorithms to use for detection. Default: all algorithms."
    )
parser.add_argument('--save', default="results/scores", type=str)
args = parser.parse_args()
print(args)

# Get the data path
if args.attacker is None:
    data_dir = f"results/{args.model.split('/')[1]}-{args.method}-c4-m400-T1000-skipgram_prf-15485863-temp{args.temperature}-{args.task}.pkl"
    save_name = f"{args.model.split('/')[1]}_{args.task}_{args.temperature}_{args.method}_{args.alpha}.pkl"
elif args.attacker in ['WordDeletion', 'SynonymSubstitution']:
    data_dir = (f"results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temperature}_{args.method}_{args.task}_{args.attacker}-{args.attacker_rate}.pkl")
    save_name = f"{args.model.split('/')[1]}_{args.task}_{args.temperature}_{args.method}_{args.attacker}_{args.attacker_rate}_{args.alpha}.pkl"
elif args.attacker == 'Dipper':
    data_dir = (f"results/attack/{args.attacker}/Y_{args.model.split('/')[1]}_{args.temperature}_{args.method}_{args.task}_{args.attacker}-{args.order}-{args.lex}.pkl")
    save_name = f"{args.model.split('/')[1]}_{args.task}_{args.temperature}_{args.method}_{args.attacker}_{args.order}_{args.lex}_{args.alpha}.pkl"
else:
    raise ValueError("Invalid value for attacker")

# if args.human:
#     data_dir = f'dataset/Y_score/human_text_{args.method}_{args.task}.pkl'
#     save_name = f'human_text_{args.method}_{args.task}_{args.alpha}.pkl'

# if args.data == 'new':
#     data_dir = f'dataset/Y_score/new_inverse/Y_{model_name}_{args.temperature}_{args.algorithm}.pkl'
#     save_name = f'{model_name}_{args.temperature}_{args.algorithm}_{args.alpha}.pkl'

# Load the Y score
with open(data_dir, 'rb') as f:
    data = pickle.load(f)

    if args.attacker is None:
        data = data['watermark']['Ys']

if args.method == 'transform':
    data = -np.array(data)
else:
    data = np.array(data)
# if args.data == 'new':
#     data = -data

# Drop repeating pivotal values, truncate the length to 200
if args.no_repeat:
    data_np = []
    for i in range(len(data)):
        d = set(data[i])
        if len(d) > args.no_repeat:
            data_np.append(list(d)[:args.no_repeat])
    data = np.array(data_np)
    print(f'Evaluate non-repeating pivotal values, the length of processed data is {data.shape[0]}')
    save_name = f'{save_name[:-4]}_np{args.no_repeat}.pkl'

result_dic = {}
config_manager = ConfigManager()
if args.alpha != 0.05:  # default alpha is 0.05
    for test_name in config_manager.configs.keys():
        config_manager.set_config(test_name, alpha=args.alpha)
if args.method == 'transform':
    config_manager.set_config('sum_based_its', delta=0.5)
    save_name = f'{save_name[:-4]}_d0.5.pkl'
    stat_tests = StatisticalTests(data, config_manager, 'ITSEdit')
else:
    stat_tests = StatisticalTests(data, config_manager, 'EXP')
for test in tqdm(args.detection):
    results, _ = stat_tests.perform_statistical_test(test)
    result_dic[test] = results

save_path = args.save
os.makedirs(save_path, exist_ok=True)
with open(f'{save_path}/{save_name}', 'wb') as f:
    pickle.dump(result_dic, f)

print('result saved')

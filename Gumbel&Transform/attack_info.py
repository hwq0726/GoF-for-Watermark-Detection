import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from attack_utils import replace_random
from detect_utils import ConfigManager, StatisticalTests, PhiDivergenceTest

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--task',default="CS",type=str)
parser.add_argument('--method',default="transform",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
parser.add_argument('--temperature',default=1.0,type=float)
parser.add_argument('--rate',default=5.0,type=float)
parser.add_argument('--alpha', default=0.01, type=float)
# Detection method setting
parser.add_argument(
        '--detection',
        nargs='+',  # Accepts multiple values
        default=['sum_based', 'log', 'ars', 'phi_divergence', 'kuiper', 'kolmogorov', 'anderson',
                 'cramer', 'watson', 'neyman', 'chi_squared', 'rao', 'greenwood'],
        help="List of algorithms to use for detection. Default: all algorithms."
    )
args = parser.parse_args()
print(args)

# Get Ys
data_dir = f"results/{args.model.split('/')[1]}-{args.method}-c4-m400-T1000-skipgram_prf-15485863-temp{args.temperature}-{args.task}.pkl"
with open(data_dir, 'rb') as f:
    data = pickle.load(f)
data = data['watermark']['Ys']
# adversarial edits
data = replace_random(data, args.rate, args.method)

# get the score
save_name = f"{args.model.split('/')[1]}_{args.task}_{args.temperature}_{args.method}_adversary_{args.rate}_{args.alpha}.pkl"
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

save_path = 'results/scores/adversary'
# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)
with open(f"{save_path}/{save_name}", 'wb') as f:
    pickle.dump(result_dic, f)

print('result saved')

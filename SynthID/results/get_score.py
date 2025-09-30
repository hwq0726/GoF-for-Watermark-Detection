import os
import pickle
import numpy as np
from scipy.stats import chi2
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import math
from math import comb, floor
from detect_utils import ConfigManager, StatisticalTests, PhiDivergenceTest

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--task', default='CS', type=str)
parser.add_argument('--no_repeat', default=None, type=int)  # take values in [10, 30, 50]
parser.add_argument('--info_attack', default=None, type=int)
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)        # change needed
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--human', default=False, type=bool)
parser.add_argument('--algorithm', default="EXP", type=str)
parser.add_argument('--temperature', default=0.1, type=float)
# WordDeletion and SynonymSubstitution settings
parser.add_argument('--attacker', default=None, type=str)     # change needed
parser.add_argument('--attacker_rate', default=0.3, type=float)
# Dipper paraphraser settings
parser.add_argument('--order', default=20, type=int)        # take values in [0, 20, 40, 60, 80, 100]
parser.add_argument('--lex', default=20, type=int)
# Detection method setting
parser.add_argument(
        '--detection',
        nargs='+',  # Accepts multiple values
        default=['sum_based', 'phi_divergence', 'kuiper', 'kolmogorov', 'anderson',
                 'cramer', 'watson', 'neyman', 'chi_squared', 'rao', 'greenwood'],
        help="List of algorithms to use for detection. Default: all algorithms."
    )
parser.add_argument('--mean', default=True, type=bool)
parser.add_argument('--weighted_mean', default=True, type=bool)
args = parser.parse_args()
print(args)

# define the function used to convert the sample into unifrom distribution
def irwin_hall_cdf(s, n=30):
    """
    Compute the CDF of the Irwin–Hall distribution at point s,
    for the sum of n i.i.d. Uniform(0,1) random variables.
    
    F_S(s) = (1 / n!) * sum_{k=0 to floor(s)} [(-1)^k * C(n, k) * (s - k)^n],
    valid for 0 <= s <= n.
    """
    if s <= 0:
        return 0.0
    if s >= n:
        return 1.0
    
    cdf_val = 0.0
    k_upper = floor(s)
    for k in range(k_upper + 1):
        term = ((-1)**k) * comb(n, k) * max(s - k, 0)**n
        cdf_val += term
    return cdf_val / math.factorial(n)

def Y_cdf(y):
    """
    CDF of Y = (X_1 + ... + X_30)/30.
    We have Y in [0, 1], and F_Y(y) = P(Y <= y) = F_S(30*y),
    where S ~ Irwin-Hall(n=30).
    """
    if y <= 0:
        return 0.0
    if y >= 1:
        return 1.0
    return irwin_hall_cdf(30*y, n=30)

def transform_samples_Y_to_uniform(y_array):
    """
    Apply the probability integral transform elementwise to an array of samples
    from Y, returning an array of the same shape containing Uniform(0,1) samples.
    
    Parameters
    ----------
    y_array : np.ndarray
        An array of shape (m, n) containing samples from Y.
    
    Returns
    -------
    np.ndarray
        A new array of shape (m, n), where each element is F_Y(y_ij).
    """
    # Vectorize the Y_cdf function for elementwise application
    vectorized_cdf = np.vectorize(Y_cdf, otypes=[float])
    return vectorized_cdf(y_array)

# Load the data
model_name = args.model.split('/')[-1]
if not args.attacker:
    data_dir = f"no_modified_pivotal/SynthID_{args.task}_{model_name}_1000_400_temp-{args.temperature}_generated_gs.pkl"
    save_name = f"detection_results/original/SynthID_{args.task}_{model_name}_temp-{args.temperature}.pkl"
elif args.attacker == 'Dipper':
    data_dir = f"attack/Dipper/Y_{model_name}_{args.temperature}_SynthID_{args.task}_Dipper-{args.order}-{args.lex}.pkl"
    save_name = f"detection_results/Dipper/SynthID_{args.task}_{model_name}_temp-{args.temperature}_Dipper-{args.order}-{args.lex}.pkl"
elif args.attacker in ['WordDeletion', 'SynonymSubstitution']:
    data_dir = f"attack/{args.attacker}/Y_{model_name}_{args.temperature}_SynthID_{args.task}_{args.attacker}-{args.attacker_rate}.pkl"
    save_name = f"detection_results/{args.attacker}/SynthID_{args.task}_{model_name}_temp-{args.temperature}_{args.attacker}-{args.attacker_rate}.pkl"
else:
    raise ValueError('Invalid attacker name')

# Load the values, the target file is a dictionary with keys 'pivotal' and 'g_value'
with open(data_dir, 'rb') as f:
    data = pickle.load(f)

if not args.attacker:
    g_values = np.array(data['g_value'][:1000])
    pivotals = np.array(data['pivotal'][:1000])
elif args.attacker in ['Dipper', 'SynonymSubstitution']:
    g_values = np.array([data['g_value'][i][-300:] for i in range(len(data['g_value']))])
    pivotals = np.mean(g_values, axis=-1) # here since when store the pivotao, there is a little mistake, should take the lated 300 tokens
elif args.attacker == 'WordDeletion':
    g_values = np.array([data['g_value'][i][-250:] for i in range(len(data['g_value']))])
    pivotals = np.mean(g_values, axis=-1)
else:
    raise ValueError('Invalid attacker name')

# Drop repeating pivotal values, truncate the length to 200
if args.no_repeat:
    print('Perform non-repeating pivotal values')
    data_np = []
    for i in range(len(pivotals)):
        d = set(pivotals[i])
        if len(d) > args.no_repeat:
            data_np.append(list(d)[:args.no_repeat])
    pivotals = np.array(data_np)
    print(f'Evaluate non-repeating pivotal values, the length of processed data is {pivotals.shape[0]}')
    save_name = save_name.replace('.pkl', f'_no_repeat-{args.no_repeat}.pkl')

# transform the pivotal values to uniform distribution under null hypothesis, shape of the pivotal is (num_samples, num_tokens)
transformer_pivotals = transform_samples_Y_to_uniform(pivotals)

# if need to perform information attack
if args.info_attack:
    print('Perform information attack')
    assert args.attacker is None, 'Information attack is only for original data'
    n_rows, n_cols = transformer_pivotals.shape
    num_to_replace = int(np.ceil(args.info_attack / 100 * n_cols))
    modified_data = transformer_pivotals.copy()
    generate_samples = lambda size: np.random.uniform(0, 1, size)
    for i in range(n_rows):
        # Get the indices of the top k% values in the row
        top_k_indices = np.argsort(modified_data[i])[-num_to_replace:]
        # Replace these values with random numbers
        modified_data[i, top_k_indices] = generate_samples(num_to_replace)
    transformer_pivotals = modified_data
    save_name = f"detection_results/info/SynthID_{args.task}_{model_name}_temp-{args.temperature}_info-{args.info_attack}.pkl"

# Perform Goodness-of-Fit tests
results_dict = {}
config_manager = ConfigManager()
if args.alpha != 0.05:
    for test_name in config_manager.configs.keys():
        config_manager.set_config(test_name, alpha=args.alpha)
stat_tests = StatisticalTests(transformer_pivotals, config_manager, args.algorithm)
for test in tqdm(args.detection):
    results, _ = stat_tests.perform_statistical_test(test)
    results_dict[test] = results

if args.mean:
    print('Using mean')
    # g_values = np.array(data['g_value'][:1000]) # shape of g_values is (1000, 400, 30)
    token_length = g_values.shape[-2]
    watermarking_depth = g_values.shape[-1]
    pivotal = np.mean(g_values, axis=-1)
    if args.info_attack:
        print('Perform information attack with pivotal sum')
        num_to_replace = int(np.ceil(args.info_attack / 100 * token_length))
        for i in range(len(pivotal)):
            # Get the indices of the top k% values in the row
            top_k_indices = np.argsort(pivotal[i])[-num_to_replace:]
            # Replace these values with random numbers
            pivotal[i, top_k_indices] = np.mean(np.random.uniform(0, 1, (num_to_replace, watermarking_depth)), axis=1)
    sum_pivotal = np.cumsum(pivotal, axis=1)
    s = np.random.uniform(0, 1, (10000, token_length, watermarking_depth))
    s = np.mean(s, axis=-1)
    simulation_outcome = np.cumsum(s, axis=1)
    quantile = np.quantile(simulation_outcome, 1 - args.alpha, axis=0)
    r = np.mean(sum_pivotal >= quantile, axis=0)
    results_dict['mean'] = r

if args.weighted_mean:
    print('Using weighted mean')
    # g_values = np.array(data['g_value'][:1000]) # shape of g_values is (1000, 400, 30)
    token_length = g_values.shape[-2]
    watermarking_depth = g_values.shape[-1]
    # use the default weights
    weights = np.linspace(start=10, stop=1, num=watermarking_depth)
    weights *= watermarking_depth / np.sum(weights)
    g_values *= weights[None, None, :]
    weighted_pivotal = np.mean(g_values, axis=-1)
    sum_weighted_pivotal = np.cumsum(weighted_pivotal, axis=1)
    s = np.random.uniform(0, 1, (10000, token_length, watermarking_depth))
    s *= weights[None, None, :]
    s = np.mean(s, axis=-1)
    simulation_outcome = np.cumsum(s, axis=1)
    quantile = np.quantile(simulation_outcome, 1 - args.alpha, axis=0)
    r = np.mean(sum_weighted_pivotal >= quantile, axis=0)
    results_dict['weighted_mean'] = r

# make sure the directory exists
os.makedirs(os.path.dirname(save_name), exist_ok=True)
with open(save_name, 'wb') as f:
    pickle.dump(results_dict, f)

print('result saved to', save_name)

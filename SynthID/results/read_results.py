import pickle
import torch
import numpy as np

temps = [0.1, 0.3, 0.7, 1.0]
tasks = ['CS', 'QA']
models = ['opt-1.3b', 'opt-13b', 'Llama-3.1-8B']

for temp in temps:
    for task in tasks:
        for model in models:
            print(f"task: {task}, model: {model}, temp: {temp}")
            target_file = f"full_results/SynthID_{task}_{model}_1000_400_temp-{temp}.pkl"
            target_key = 'generated_gs'
            with open(target_file, 'rb') as f:
                results = pickle.load(f)

            target_value = results[target_key]
            pivotal = []
            g_values = []
            for i in range(len(target_value)):
                pivotal.append(np.mean(np.array(target_value[i]), axis=-1)[50:450])
                g_values.append(np.array(target_value[i][50:450]))
            pivotal = np.array(pivotal)
            g_values = np.array(g_values)
            results = {'pivotal': pivotal, 'g_value': g_values}
            # save the target value into .pkl file
            results_file = f"no_modified_pivotal/SynthID_{task}_{model}_1000_400_temp-{temp}_generated_gs.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

            # target_text = results['watermarked_texts']
            # # save the target value into .pkl file
            # results_file = f"SynthID_{task}_{model}_1000_400_temp-{temp}_watermarked_text.pkl"
            # with open(results_file, 'wb') as f:
            #     pickle.dump(target_text, f)

# temp = 1.0
# target_file = f"full_results/SynthID_QA_opt-1.3b_1000_400_temp-{temp}.pkl"
# target_key = 'generated_gs'
# with open(target_file, 'rb') as f:
#     results = pickle.load(f)

# print(results.keys())

# target_value = results[target_key]
# target_value = np.mean(np.array(target_value), axis=-1)
# print(target_value.shape)
# # # save the target value into .pkl file
# results_file = f"SynthID_QA_opt-1.3b_1000_400_temp-{temp}_generated_gs.pkl"
# with open(results_file, 'wb') as f:
#     pickle.dump(target_value, f)

# target_text = results['watermarked_texts']
# # save the target value into .pkl file
# results_file = f"SynthID_QA_opt-1.3b_1000_400_temp-{temp}_watermarked_text.pkl"
# with open(results_file, 'wb') as f:
#     pickle.dump(target_text, f)

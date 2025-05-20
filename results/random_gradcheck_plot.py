import matplotlib.pyplot as plt
import numpy as np
import re
import os

folder = os.path.dirname(__file__)

def parse_gradcheck_file(filename):
    trials = []
    biases_diff = []
    weights_diff = []
    
    with open(filename, 'r') as f:
        content = f.read()
        
    # Find all trial numbers, bias and weight differences using regex
    trial_matches = re.findall(r'Trial #(\d+)', content)
    bias_matches = re.findall(r'Biases Diff: ([\d.e+-]+)', content)
    weight_matches = re.findall(r'Weights Diff: ([\d.e+-]+)', content)
    
    # Convert strings to appropriate types
    trials = [int(x) for x in trial_matches][-50:]
    biases_diff = [float(x) for x in bias_matches][-50:]
    weights_diff = [float(x) for x in weight_matches][-50:]
    
    return trials, biases_diff, weights_diff

# Read data from file
trials, biases_diff, weights_diff = parse_gradcheck_file('results/random_gradcheck.txt')

# Plot
plt.figure(figsize=(16, 6))
bar_width = 0.4
index = np.arange(len(trials))

plt.bar(index, biases_diff, bar_width, label='Biases Diff', color='skyblue')
plt.bar(index + bar_width, weights_diff, bar_width, label='Weights Diff', color='salmon')

plt.xlabel('Trial Number')
plt.ylabel('Gradient Check Relative Error')
plt.title('Gradient Check Relative Error by Trial (Trial #50 to #99)')
plt.xticks(index + bar_width / 2, trials, rotation=45)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig(os.path.join(folder, "RandGradientCheck.png"))

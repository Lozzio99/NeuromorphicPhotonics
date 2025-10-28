import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.config import RESULTS_DIRECTORY

fp1 = os.path.join(RESULTS_DIRECTORY, f'spike_probability_by_pulse_10_runs.csv')
fp2 = os.path.join(RESULTS_DIRECTORY, f'double_spike_10_runs.csv')

data1 = pd.read_csv(fp1)
data2 = pd.read_csv(fp2)

print(data1, data2)
high_threshold = 0.8
low_threshold = 0.2

merged = pd.merge(data1, data2, on=['length', 'strength'], suffixes=('_single', '_double')).dropna()
print(merged)


unique_lengths = merged['length'].unique()
unique_strengths = merged['strength'].unique()

print(unique_lengths, unique_strengths)
results = []

for L in unique_lengths:
    for S in unique_strengths:
        pos = merged[(merged['length'] == L) & (merged['strength'] == S)]
        neg = merged[(merged['length'] == L) & (merged['strength'] == -S)]

        if S < 0: pos, neg = neg, pos

        if pos.empty or neg.empty:
            continue


        # Check condition in csv1
        single_firing_positive = pos['spike_probability_single'].values[0] >= high_threshold
        single_firing_negative = neg['spike_probability_single'].values[0] >= high_threshold

        if single_firing_positive and single_firing_negative:

            double_firing_positive = pos['spike_probability_double'].values[0] <= low_threshold
            double_firing_negative = neg['spike_probability_double'].values[0] <= low_threshold

            # Now check condition in csv2
            if double_firing_positive and double_firing_negative:
                results.append({
                    "length": L,
                    "strength": S,
                    "P_single_pos": pos['spike_probability_single'].values[0],
                    "P_single_neg": neg['spike_probability_single'].values[0],
                    "P_double_pos": pos['spike_probability_double'].values[0],
                    "P_double_neg": neg['spike_probability_double'].values[0]
                })

results_df = pd.DataFrame(results)
print(results_df)


plt.figure(figsize=(10,6))

# Compute difference
merged['prob_diff'] = merged['spike_probability_single'] - merged['spike_probability_double']

scatter = plt.scatter(merged['strength'],
                      merged['length'],
                      c=merged['prob_diff'],
                      cmap='coolwarm', s=50)

if not results_df.empty:
    # Highlight the points that meet condition
    plt.scatter(results_df['strength'], results_df['length'],
                facecolors='none', edgecolors='black', s=120, label=f"single >{high_threshold} & double_pulse <{low_threshold}")

plt.colorbar(scatter, label='Pulse Input Comparison Spike Probability Difference (Single - Double)')
plt.xlabel("Strength")
plt.ylabel("Length")
plt.legend()
plt.title("Comparison of Spike Probabilities")
plt.show()



# interval
# phase space
# more range
# logical gates

##
# run again big plot (double-single) with 4 lasers
# bigger interval range (improve plot)
# bigger gates - binary output as pulse for next cycle
# write something about the pulse response thing (pos vs neg)
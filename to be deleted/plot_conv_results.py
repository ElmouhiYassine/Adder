import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('to be deleted/fashion_positive_100_images.csv')


name_map = {
    'Naive': 'Naive',
    'Sobo': 'Sobociński',
    'Bochvar': 'Bochvar external',
    'Gaines': 'Gaines–Rescher',
    'Sette': 'Sette',
    'SK': 'Strong Kleene'
}

ordered_keys = ['Naive', 'Sobo', 'Bochvar', 'Gaines', 'Sette', 'SK']

style_map = {
    'Naive': {'color': 'black', 'marker': 'o'},
    'Sobo': {'color': 'blue', 'marker': '^'},
    'Bochvar': {'color': 'orange', 'marker': 'D'},
    'Gaines': {'color': 'brown', 'marker': '*'},
    'Sette': {'color': 'purple', 'marker': 'p'},
    'SK': {'color': 'red', 'marker': 's'}
}

plt.figure(figsize=(12, 7))

x_values = df['Uncertainty Level']
n_adders = len(ordered_keys)
jitter_step = 0.3

for i, key in enumerate(ordered_keys):
    avg_col = f'Avg Error {key}'
    sd_col = f'SD Error {key}'
    full_name = name_map[key]

    offset = (i - (n_adders - 1) / 2) * jitter_step
    style = style_map[key]

    plt.errorbar(
        x_values,
        # df[avg_col],
        yerr=df[sd_col],
        label=full_name,
        marker=style['marker'],
        color=style['color'],
        capsize=3,
        # linestyle='-',
        # linewidth=1.5,
        alpha=0.8
    )

plt.xticks(x_values)
plt.xlabel('Uncertainty (max value)')
plt.ylabel('Average Error (MAE)')
plt.title('Adder Performance Comparison on Fashion MNIST using Positive Kernels')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('adder_performance_comparison_fashion_positive.png')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Signal data
# signals = {
#     '2Q': [0,0,0,1,0.25,0.25],
#     '1Q': [0,1,0,0,0.5,0.5],
#     'N': [0,0,0,1,1,0],
#     'E': [0,1,1,0,1,0],
#     '2D': [0,1,1,1,0.25,0.5],
#     '1D': [0,1,0,0,0.5,0.55],
# } 

# signals = {
#     'Q': [0, 0, 0, 0.666, 0.666, 0.333, 0, 1],
#     'CLK': [0, 1, 0, 1, 0, 1, 0, 1],
#     'CLR': [0, 0, 1, 1, 1, 1, 0, 1],
#     'D': [0, 0.666, 0.666, 0.666, 0.333, 0.333, 1, 1],
#
# }

# signals = {
#     'Out': [0,0,1,1,0],
#     'E': [0,1,1,0,0],
#     'C': [0,0,1,0,1],
# }

# signals = {
#     'Out': [0,0,1,1,1,1,1,0],
#     'E': [0,1,1,1,1,0,0,0],
#     'C': [0,0,1,0,1,1,0,1],
# }

# signals = {
#     'Out': [0,0,1,1,1,1,1,0,0,1],
#     'C': [0,0,1,0,0,1,0,1,0,1],
#     'E': [0,1,1,1,0,0,0,0,1,1],
# }

# signals = {
#     'Out': [0,0,1,0,1,1,0],
#     'E':   [0,1,1,1,1,0,0],
#     'C':   [0,0,1,0,1,1,0],
# }

signals = {
    'q[0]': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'q[1]': [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    'q[2]': [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    'q[3]': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    'q': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'clk': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
} 

# Color palette for signals
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

# Variable to control graph height
graph_height_multiplier = 0.5

# Create the plot with black background
plt.figure(figsize=(16, len(signals) * graph_height_multiplier), facecolor='black', edgecolor='none')
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['text.color'] = 'white'
plt.title('Timing Diagram', fontsize=15, color='white')

# Plot each signal
for i, (signal_name, signal_data) in enumerate(signals.items()):
    # Offset each signal vertically
    y_offset = i * 1.2

    # Extend x-axis to include last step
    x = np.arange(len(signal_data) + 1) * 600
    y = np.append(signal_data, signal_data[-1]) + y_offset

    # Use different color for each signal
    color = colors[i % len(colors)]

    plt.step(x, y, where='post', linewidth=2, color=color)

    # Add signal name on the left side
    plt.text(-200, y_offset + 0.5, signal_name,
             verticalalignment='center',
             horizontalalignment='right',
             color='white')

    # Add solid horizontal line to the right of Y-axis labels
    plt.hlines(y=y_offset + 0.05, xmin=-1000, xmax=0, colors='white', linewidth=0.8)

# Customize the plot
plt.xlim(-300, len(list(signals.values())[0]) * 600)
plt.ylim(-0.5, len(signals) * 1.2)

# Set x-axis ticks every 600
# Set x-axis ticks every 600
x_ticks = np.arange(0, len(list(signals.values())[0]) * 600 + 600, 600)
plt.xticks(x_ticks, labels=[f"{tick}m" for tick in x_ticks], color='white')


# Move x-axis labels to the top
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')

# Remove y-axis ticks and labels
plt.gca().yaxis.set_ticks([])
plt.gca().yaxis.set_ticklabels([])

# Customize tick colors
plt.tick_params(axis='both', colors='white')

# Add solid line below X-axis labels
plt.hlines(y=7.1, xmin=-300, xmax=len(list(signals.values())[0]) * 600, colors='white', linewidth=0.8)

# Add dashed vertical lines from each X-axis label
for tick in x_ticks:
    plt.axvline(x=tick, color='white', linestyle='--', linewidth=0.8)

plt.tight_layout()

# Show the plot
plt.show()

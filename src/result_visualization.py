import numpy as np
import matplotlib.pyplot as plt

# Set style parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Define the actual results
LAYERS = [0, 7, 13, 19, 27]
ACCURACIES = {
    0: 0.704225352112676,
    7: 0.6901408450704225,
    13: 0.676056338028169,
    19: 0.7183098591549296,
    27: 0.7887323943661971
}

def plot_probe_performance():
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot data
    layers = list(ACCURACIES.keys())
    acc_values = [ACCURACIES[l] * 100 for l in layers]
    
    # Plot line connecting points
    ax.plot(layers, acc_values, '-', color='#2E86C1', linewidth=2.5, label='Probe Accuracy')
    
    # Plot points
    ax.plot(layers, acc_values, 'o', color='#2E86C1', markersize=8)
    
    # Add random chance baseline
    ax.axhline(y=50, color='#E74C3C', linestyle='--', alpha=0.8, label='Random Chance')
    
    # Highlight best performing layer
    best_layer = 27
    best_acc = ACCURACIES[best_layer]
    ax.plot(best_layer, best_acc * 100, 'o', color='#27AE60', markersize=10,
            label=f'Best Layer (#{best_layer})')
    
    # Customize the plot
    ax.set_xlabel('Model Layer', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Faithfulness Detection Accuracy at Key Model Layers', fontsize=14, pad=15)
    
    # Set axis limits with some padding
    ax.set_xlim(-1, 28)
    ax.set_ylim(45, 85)
    
    # Customize ticks
    ax.set_xticks(layers)
    ax.set_yticks(np.arange(45, 85, 5))
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add annotations for best layer
    ax.annotate(f'Peak Accuracy: {best_acc*100:.1f}%',
                xy=(best_layer, best_acc * 100),
                xytext=(best_layer-8, best_acc * 100 + 3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Add analysis summary
    acc_values_list = list(ACCURACIES.values())
    summary_text = (f'Mean Accuracy: {np.mean(acc_values_list)*100:.1f}%\n'
                   f'Std Dev: {np.std(acc_values_list)*100:.1f}%')
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('probe_performance_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate the plot
plot_probe_performance()

# Print summary statistics
acc_values_list = list(ACCURACIES.values())
print(f"\nSummary Statistics:")
print(f"Mean Accuracy: {np.mean(acc_values_list)*100:.1f}%")
print(f"Standard Deviation: {np.std(acc_values_list)*100:.1f}%")
print(f"Best Layer: 27")
print(f"Best Accuracy: {ACCURACIES[27]*100:.1f}%")

print(f"\nAccuracies by layer:")
for layer, acc in sorted(ACCURACIES.items()):
    print(f"Layer {layer}: {acc*100:.1f}%")
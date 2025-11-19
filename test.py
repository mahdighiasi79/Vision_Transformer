import torch
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    x_axis = np.arange(1, 12, dtype=int)
    y_axis = np.arange(1, 12, dtype=int)
    plt.figure(figsize=(8, 4))  # Create a new figure
    plt.plot(x_axis, y_axis, label='Sine Wave', color='blue')
    plt.title('Line Plot of X vs Y Data')
    plt.xlabel('X-axis Data (Time)')
    plt.ylabel('Y-axis Data (Amplitude)')
    plt.grid(True)
    plt.legend()
    plt.savefig('line_plot.png')  # Save the plot to a file

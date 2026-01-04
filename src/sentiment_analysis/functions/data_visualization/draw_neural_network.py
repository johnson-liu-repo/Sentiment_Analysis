


####################################################################################
# Notes
#######
# 1. How does this code work?
####################################################################################


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class NeuralNetworkVisualizer:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.layer_names = []

    def add_layer(self, num_neurons, layer_name=""):
        """Add a layer to the neural network."""
        self.layers.append(num_neurons)
        self.layer_names.append(layer_name)

    def initialize_weights(self, initial_weights):
        """Initialize weights from the first frame of weight_frames."""
        if len(self.layers) < 2:
            raise ValueError("At least two layers are required to initialize weights.")
        if len(initial_weights) != len(self.layers) - 1:
            raise ValueError("The number of weight matrices must match the number of layer connections.")
        for i, weight_matrix in enumerate(initial_weights):
            expected_shape = (self.layers[i], self.layers[i + 1])
            if weight_matrix.shape != expected_shape:
                raise ValueError(f"Weight matrix dimensions {weight_matrix.shape} do not match the expected dimensions {expected_shape}.")
        self.weights = initial_weights

    def draw(self, neuron_spacing=0.04, animate=False, save_figure=False, frames=50, weight_frames=None, log_scale=False):
        """Draw the neural network with optional animation and log scale for weights."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('equal')
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Calculate layer positions.
        x_positions = np.linspace(0, 1, len(self.layers))
        y_positions = []
        for num_neurons in self.layers:
            total_height = 0.8
            start_y = 0.5 - (num_neurons - 1) * neuron_spacing / 2
            y_positions.append([start_y + i * neuron_spacing for i in range(num_neurons)])

        # Draw neurons.
        neuron_radius = 0.02
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            for neuron_y in y:
                circle = plt.Circle((x, neuron_y), neuron_radius, color='black', fill=False, lw=1.5)
                ax.add_artist(circle)
            if self.layer_names[i]:
                label_y_position = min(y) - 0.05
                ax.text(x, label_y_position, self.layer_names[i], ha='center', fontsize=10)

        # Draw weights.
        lines = []
        for i, weight_matrix in enumerate(self.weights):
            for j, y1 in enumerate(y_positions[i]):
                for k, y2 in enumerate(y_positions[i + 1]):
                    weight = weight_matrix[j, k]
                    color = plt.cm.bwr((weight + 1) / 2)
                    x_start = x_positions[i] + neuron_radius
                    x_end = x_positions[i + 1] - neuron_radius
                    line, = ax.plot([x_start, x_end], [y1, y2], color=color, lw=0.5)
                    lines.append((line, i, j, k))

        # Determine global min and max weights across all frames.
        if weight_frames:
            all_weights = np.concatenate([np.concatenate([w.flatten() for w in frame]) for frame in weight_frames])
            if log_scale:
                all_weights = np.log(np.abs(all_weights) + 1e-8)  # Apply log scale
            global_min = np.min(all_weights)
            global_max = np.max(all_weights)
        else:
            global_min, global_max = -1, 1  # Default range if no weight frames are provided.

        # Add colorbar for weights with consistent scale.
        sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Log Weight Value' if log_scale else 'Weight Value', fontsize=10)

        if animate:
            if not weight_frames:  # Check if weight_frames is empty.
                print("weight_frames is empty. Showing a still image.")
                plt.show()
                return

            if len(weight_frames) != frames:
                raise ValueError("weight_frames must be provided and match the number of frames.")

            def update(frame):
                # Update weights for animation.
                current_weights = weight_frames[frame]
                if log_scale:
                    current_weights = [np.log(np.abs(w) + 1e-8) for w in current_weights]  # Apply log scale.
                for line, i, j, k in lines:
                    weight = current_weights[i][j, k]
                    color = plt.cm.bwr((weight - global_min) / (global_max - global_min))
                    line.set_color(color)
                # Update the title with the current epoch.
                ax.set_title(f"Neural Network Training - Epoch {frame + 1}", fontsize=12)

            anim = FuncAnimation(fig, update, frames=frames, interval=200, repeat=True)
            plt.show()
        else:
            # Plot only the last epoch.
            if weight_frames:
                last_weights = weight_frames[-1]
                if log_scale:
                    last_weights = [np.log(np.abs(w) + 1e-8) for w in last_weights]  # Apply log scale.
                for line, i, j, k in lines:
                    weight = last_weights[i][j, k]
                    color = plt.cm.bwr((weight - global_min) / (global_max - global_min))
                    line.set_color(color)
                ax.set_title("Neural Network Training - Final Epoch", fontsize=12)
            plt.show()

        if save_figure:
            # Save as GIF if animate, else as PNG.
            if animate:
                anim.save("neural_network_animation.gif", writer='Pillow', fps=10)
            else:
                plt.savefig("neural_network_visualization.png", bbox_inches='tight', dpi=300)


# Normalize weights across all epochs.
def normalize_weights(weight_frames):
    """Normalize weights across all epochs to a consistent range."""
    all_weights = np.concatenate([np.concatenate([w.flatten() for w in frame]) for frame in weight_frames])
    min_weight = np.min(all_weights)
    max_weight = np.max(all_weights)

    # Normalize each frame's weights.
    normalized_frames = []
    for frame in weight_frames:
        normalized_frame = [(w - min_weight) / (max_weight - min_weight) * 2 - 1 for w in frame]
        normalized_frames.append(normalized_frame)
    return normalized_frames


# Example usage.
if __name__ == "__main__":
    nn_viz = NeuralNetworkVisualizer()
    
    a = 8
    b = 10
    z = 1

    nn_viz.add_layer(a, "Input Layer")
    nn_viz.add_layer(b, "Hidden Layer 1")
    nn_viz.add_layer(b, "Hidden Layer 2")
    nn_viz.add_layer(z, "Output Layer")

    frames = 100
    # Generate weight frames for animation.
    weight_frames = []
    for _ in range(frames):
        frame_weights = [
            np.random.uniform(-5, 5, (a, b)),  # Example range of weights.
            np.random.uniform(-5, 5, (b, b)),
            np.random.uniform(-5, 5, (b, z))
        ]
        weight_frames.append(frame_weights)

    # Normalize weights across all frames.
    normalized_weight_frames = normalize_weights(weight_frames)

    # Initialize weights using the first frame.
    nn_viz.initialize_weights(normalized_weight_frames[0])

    # Pass normalized weight frames to the draw method.
    nn_viz.draw(neuron_spacing=0.06, animate=True, save_figure=True, frames=frames, weight_frames=normalized_weight_frames, log_scale=False)

    nn_viz.draw(neuron_spacing=0.06, animate=False, save_figure=True, frames=frames, weight_frames=normalized_weight_frames, log_scale=False)
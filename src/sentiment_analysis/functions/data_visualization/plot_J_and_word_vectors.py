import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_J_over_time(data_file='data/J_over_time.npy', show=False):
    """
    Plots the values of J_over_time and log(J_over_time) over iterations as an animated graph.

    Args:
        1. data_file (str): Path to the .npy file containing the J_over_time data. Default is 'data/J_over_time.npy'.
        2. show (bool): If True, displays the plot after creating the animation. Default is False.

    The Function:
        1. Loads J_over_time data from the specified .npy file.
        2. Computes the logarithm of J_over_time and scales it for visibility.
        3. Creates an animated plot showing both J_over_time and the scaled log(J_over_time) over iterations.
        4. Saves the animation as a GIF file named 'J_and_log_J_over_time_animation.gif'.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the animation as a GIF file named 'J_and_log_J_over_time_animation.gif'.
    """
    # Get J_over_time from the npy file.
    J_over_time = np.load(data_file)

    trajectory_length = len(J_over_time)

    # Compute log(J_over_time) and normalize it.
    min_J_over_time = min(J_over_time)
    max_J_over_time = max(J_over_time)
    log_J_over_time = np.log(J_over_time)
    min_log_J_over_time = min(log_J_over_time)
    max_log_J_over_time = max(log_J_over_time)
    

    log_J_over_time_normalized = (
                                    (log_J_over_time - min_log_J_over_time) /
                                    (max_log_J_over_time - min_log_J_over_time) *
                                    (max_J_over_time - min_J_over_time) + min_J_over_time
                                )

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('J and log(J) over time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.grid(True)

    # Initialize the line plots.
    line_J, = ax.plot([], [], color='blue', label='J over time')
    line_log_J, = ax.plot([], [], color='red', label='log(J) over time (scaled)')
    ax.legend()

    # Set axis limits.
    ax.set_xlim(0, len(J_over_time))
    ax.set_ylim(min_J_over_time - 100, max_J_over_time + 100)

    # Update function for the animation.
    def update(frame):
        line_J.set_data(range(frame), J_over_time[:frame])
        line_log_J.set_data(range(frame), log_J_over_time_normalized[:frame])
        return line_J, line_log_J

    # Create the animation.
    ani = FuncAnimation(fig, update, frames=trajectory_length, interval=50, blit=True)

    # Save the animation as a GIF.
    ani.save('J_and_log_J_over_time_animation.gif', writer='imagemagick')

    if show==True:
        plt.show()


def plot_word_vectors_over_time(data_file='data/word_vectors_over_time.npy', show=False):
    """
    Animates the word_vectors_over_time as a heatmap.

    Args:
        1. data_file (str): Path to the .npy file containing the word_vectors_over_time data. Default is 'data/word_vectors_over_time.npy'.
        2. show (bool): If True, displays the animation after creating it. Default is False.

    The Function:
        1. Loads word_vectors_over_time data from the specified .npy file.
        2. Converts each dictionary in the list to a 2D array.
        3. Calculates global min and max values to ensure consistent color scale across all frames.
        4. Creates an animated heatmap showing the evolution of word vectors over time with a smooth color gradient.
        5. Saves the animation as a GIF file named 'word_vectors_over_time_animation.gif'.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the animation as a GIF file named 'word_vectors_over_time_animation.gif'.
    """

    # Load word_vectors_over_time from the .npy file.
    word_vectors_over_time = np.load(data_file, allow_pickle=True)
    
    trajectory_length = len(word_vectors_over_time)

    # Function to convert a dictionary to a 2D array
    def dict_to_2d_array(word_vectors_dict):
        # Get all words (keys)
        words = sorted(list(word_vectors_dict.keys()))
        
        number_of_words = len(words)

        # Create an empty 2D array with shape (num_words, vector_dimension)
        vector_dim = len(word_vectors_dict[words[0]])
        array_2d = np.zeros(number_of_words, vector_dim)
        
        # Fill the array with vector values
        for i, word in enumerate(words):
            array_2d[i] = word_vectors_dict[word]
        
        return array_2d, words
    
    # Calculate global min and max values across all frames for consistent color scale
    global_min = float('inf')
    global_max = float('-inf')
    
    # Pre-process all frames to find global min/max values
    all_arrays = []
    for frame_dict in word_vectors_over_time:
        array_2d, _ = dict_to_2d_array(frame_dict)
        all_arrays.append(array_2d)
        
        frame_min = np.min(array_2d)
        frame_max = np.max(array_2d)
        
        if frame_min < global_min:
            global_min = frame_min
        if frame_max > global_max:
            global_max = frame_max
    
    # Convert the first dictionary to get dimensions and words
    first_array, words = dict_to_2d_array(word_vectors_over_time[0])
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Word Vectors Over Time', fontsize=14)
    
    # Initialize the heatmap with consistent color scale and smoother colormap
    # Using 'plasma' for smoother gradient transitions
    heatmap = ax.imshow(first_array, cmap='plasma', aspect='auto', 
                        vmin=global_min, vmax=global_max, interpolation='bilinear')
    
    # Add a colorbar with more ticks for better gradient visualization
    cbar = plt.colorbar(heatmap, ax=ax, label='Vector Value', 
                        ticks=np.linspace(global_min, global_max, 10))
    cbar.ax.tick_params(labelsize=10)
    
    # Set axis labels and ticks
    ax.set_xlabel('Vector Dimension', fontsize=12)
    ax.set_ylabel('Words', fontsize=12)
    
    # Add word labels on y-axis (if not too many)
    if len(words) <= 30:  # Only show labels if there aren't too many
        ax.set_yticks(range(number_of_words))
        ax.set_yticklabels(words)
    
    # Add a text annotation for iteration counter
    iteration_text = ax.text(   0.02, 0.98, 'Iteration: 1', transform=ax.transAxes,
                                fontsize=12, verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)   )
    
    # Update function for the animation
    def update(frame):
        # Get the pre-processed array for this frame
        array_2d = all_arrays[frame]
        
        # Update the heatmap data without modifying the color scale
        heatmap.set_array(array_2d)
        
        # Update the iteration counter
        iteration_text.set_text(f'Iteration: {frame + 1}')
        
        # Update the title to show progress
        ax.set_title(f'Word Vectors Over Time (Frame {frame + 1}/{len(word_vectors_over_time)})', 
                     fontsize=14)
        
        return [heatmap, iteration_text]
    
    # Create the animation with longer interval for better observation
    ani = FuncAnimation(fig, update, frames=range(0, trajectory_length, 10), 
                        interval=300, blit=True)
    
    # Save the animation as a GIF
    ani.save('word_vectors_over_time_animation.gif', writer='imagemagick')
    
    if show:
        plt.show()


def plot_word_vectors_difference(data_file='data/word_vectors_over_time.npy', show=False):
    """
    Animates the differences between consecutive word vector frames as a heatmap.

    Args:
        1. data_file (str): Path to the .npy file containing the word_vectors_over_time data. Default is 'data/word_vectors_over_time.npy'.
        2. show (bool): If True, displays the animation after creating it. Default is False.

    The Function:
        1. Loads word_vectors_over_time data from the specified .npy file.
        2. Computes the differences between consecutive frames.
        3. Creates an animated heatmap showing the differences with a smooth color gradient.
        4. Saves the animation as a GIF file named 'word_vectors_difference_animation.gif'.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the animation as a GIF file named 'word_vectors_difference_animation.gif'.
    """
    # Load word_vectors_over_time from the .npy file.
    word_vectors_over_time = np.load(data_file, allow_pickle=True)

    trajectory_length = len(word_vectors_over_time)

    # Function to convert a dictionary to a 2D array
    def dict_to_2d_array(word_vectors_dict):
        words = sorted(list(word_vectors_dict.keys()))
        vector_dim = len(word_vectors_dict[words[0]])
        array_2d = np.zeros((len(words), vector_dim))
        for i, word in enumerate(words):
            array_2d[i] = word_vectors_dict[word]
        return array_2d, words

    # Preprocess the data to store every 10th frame.
    sampled_frames = [word_vectors_over_time[i] for i in range(0, trajectory_length, 10)]
    all_arrays = [dict_to_2d_array(frame_dict)[0] for frame_dict in sampled_frames]

    trimmed_length = len(all_arrays)

    # Calculate the differences between consecutive frames (after the trimming).
    differences = [all_arrays[i] - all_arrays[i - 1] for i in range(1, trimmed_length)]

    #############################################################################
    ### This doesn't seem to be working for getting a consistent color scale. ###
    #############################################################################
    # Get global min and max values for consistent color scale.
    global_min = min(np.min(diff) for diff in differences)
    global_max = max(np.max(diff) for diff in differences)

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Difference Between Frames', fontsize=14)

    # Initialize the heatmap.
    heatmap = ax.imshow(differences[0], cmap='coolwarm', aspect='auto',
                        vmin=global_min, vmax=global_max, interpolation='bilinear')

    # Add a colorbar.
    cbar = plt.colorbar(heatmap, ax=ax, label='Difference Value',
                        ticks=np.linspace(global_min, global_max, 10))
    cbar.ax.tick_params(labelsize=10)
    
    #############################################################################

    # Set axis labels.
    ax.set_xlabel('Vector Dimension', fontsize=12)
    ax.set_ylabel('Words', fontsize=12)

    # Add a text annotation for iteration counter.
    iteration_text = ax.text(0.02, 0.98, 'Iteration: 1', transform=ax.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Update function for the animation.
    def update(frame):
        heatmap.set_array(differences[frame])
        iteration_text.set_text(f'Iteration: {frame + 1}')
        return [heatmap, iteration_text]

    # Create the animation.
    ani = FuncAnimation(fig, update, frames=trimmed_length, interval=300, blit=True)

    # Save the animation as a GIF.
    ani.save('word_vectors_difference_animation.gif', writer='imagemagick')

    if show:
        plt.show()


def plot_single_word_vector_over_time(word, data_file='word_vectors_over_time.npy', show=False):
    """
    Plots the evolution of a single word vector over time.

    Args:
        1. word (str): The word whose vector evolution is to be plotted.
        2. data_file (str): Path to the .npy file containing the word_vectors_over_time data. Default is 'word_vectors_over_time.npy'.
        3. show (bool): If True, displays the plot after creating it. Default is False.

    The Function:
        1. Loads word_vectors_over_time data from the specified .npy file.
        2. Extracts the vector for the specified word across all frames.
        3. Plots the evolution of the word vector over time.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the plot as a PNG file named '{word}_vector_evolution.png'.
    """

    # Convert the word to all lowercase.
    word = word.lower()

    # Load the word_vectors_over_time data from the .npy file.
    word_vectors_over_time = np.load(data_file, allow_pickle=True)

    trajectory_length = len(word_vectors_over_time)

    # Load the first frame to check for the word.
    first_frame = word_vectors_over_time[0]
    if word not in first_frame:
        print(f"The word '{word}' is not present in the dictionary.")
        return

    # Extract the vector for the specified word across all frames
    single_word_vector_over_time = [frame_dict[word] for frame_dict in word_vectors_over_time]

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'Evolution of "{word}" Vector Over Time', fontsize=14)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Vector Value', fontsize=12)

    dimensionality = len(single_word_vector_over_time[0])

    # Plot each dimension of the vector separately.
    for dim in range(dimensionality):
        ax.plot( range(trajectory_length),
                 [vec[dim] for vec in single_word_vector_over_time],
                 label=f'Dimension {dim + 1}' )

    # ax.legend()
    ax.grid(True)

    if show:
        plt.show()
    
    # Save the plot as a PNG file.
    plt.savefig(f'{word}_vector_evolution.png')


def plot_single_word_vector_diffence_over_time(word, data_file='word_vectors_over_time.npy', show=True):
    """
    Plots the difference between the current frame's word vector and the previous frame's word vector for each dimension.

    Args:
        1. word (str): The word whose vector evolution is to be plotted.
        2. data_file (str): Path to the .npy file containing the word_vectors_over_time data. Default is 'word_vectors_over_time.npy'.
        3. show (bool): If True, displays the plot after creating it. Default is False.

    The Function:
        1. Loads word_vectors_over_time data from the specified .npy file.
        2. Extracts the vector for the specified word across all frames.
        3. Plots the change in each dimension of the word vector over time.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the plot as a PNG file named '{word}_vector_change_per_dimension.png'.
    """

    # Convert the word to all lowercase.
    word = word.lower()

    # Load the word_vectors_over_time data from the .npy file.
    word_vectors_over_time = np.load(data_file, allow_pickle=True)

    trajectory_length = len(word_vectors_over_time)

    # Load the first frame to check for the word.
    first_frame = word_vectors_over_time[0]
    if word not in first_frame:
        print(f"The word '{word}' is not present in the dictionary.")
        return

    # Extract the vector for the specified word across all frames.
    single_word_vector_over_time = [frame_dict[word] for frame_dict in word_vectors_over_time]

    # Calculate differences between consecutive frames for each dimension.
    dimensionality = len(single_word_vector_over_time[0])
    differences = [
        [single_word_vector_over_time[i][dim] - single_word_vector_over_time[i - 1][dim]
         for i in range(1, trajectory_length)]
        for dim in range(dimensionality)
    ]

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'Change in "{word}" Vector Over Time (Per Dimension)', fontsize=14)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Change in Vector Value', fontsize=12)

    # Plot the change for each dimension.
    for dim in range(dimensionality):
        # Plot the regular differences.
        ax.plot(range(1, trajectory_length), differences[dim], label=f'Dimension {dim + 1}')
        
        # # Normalize the log differences to fit in the same range as the regular differences.
        # # Add a small value to avoid log(0).
        # log_differences = np.log(np.abs(differences[dim]) + 1e-10)
        # log_differences_normalized = (
        #     (log_differences - np.min(log_differences)) /
        #     (np.max(log_differences) - np.min(log_differences)) *
        #     (np.max(differences[dim]) - np.min(differences[dim])) +
        #     np.min(differences[dim])
        # )
        
        # Plot the normalized log differences.
        # ax.plot(range(1, trajectory_length), log_differences_normalized, linestyle='--', label=f'Log Dimension {dim + 1}')

    # ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    # Save the plot as a PNG file.
    plt.savefig(f'{word}_vector_change_per_dimension.png')


def plot_single_vector_dimension_change_over_time(dimension, data_file='word_vectors_over_time.npy', show=True):
    """
    Plots the change in a single vector dimension over time.

    Args:
        1. dimension (int): The index of the vector dimension to be plotted.
        2. data_file (str): Path to the .npy file containing the word_vectors_over_time data. Default is 'word_vectors_over_time.npy'.
        3. show (bool): If True, displays the plot after creating it. Default is False.

    The Function:
        1. Loads word_vectors_over_time data from the specified .npy file.
        2. Extracts the vector for the specified word across all frames.
        3. Plots the change in a single dimension of the word vector over time.

    Returns:
        To terminal:
            1. None
        To file:
            1. Saves the plot as a PNG file named '{word}_vector_change_per_dimension.png'.
    """
    # Load the word_vectors_over_time data from the .npy file.
    word_vectors_over_time = np.load(data_file, allow_pickle=True)

    trajectory_length = len(word_vectors_over_time)

    words = word_vectors_over_time[0].keys()
    # print(words)

    element_over_time = [frame[word][dimension] for frame in word_vectors_over_time for word in words]

    '''
    differences = [
        single_word_vector_over_time[i][dimension] - single_word_vector_over_time[i - 1][dimension]
        for i in range(1, trajectory_length)
    ]

    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'Change in Vector Dimension {dimension + 1} Over Time', fontsize=14)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Change in Vector Value', fontsize=12)

    # Plot the change for the specified dimension.
    ax.plot(range(1, trajectory_length), differences, label=f'Dimension {dimension + 1}')

    if show:
        plt.show()

    # Save the plot as a PNG file.
    plt.savefig(f'dimension_{dimension + 1}_change.png')
    '''


if __name__ == "__main__":
    # Call the function to plot J_over_time.
    J_data_file = 'testing_scrap_misc/scrap_01/J_over_time.npy'
    plot_J_over_time(J_data_file, show=True)


    # Call the function to animate word_vectors_over_time.
    # plot_word_vectors_over_time()

    # Call the function to animate word_vectors_over_time differences.
    # plot_word_vectors_difference()

    # data_file='data/project_data/training_data/test/project_word_vectors_over_time_01.npy'
    # plot_single_word_vector_over_time(word='two', data_file=data_file)
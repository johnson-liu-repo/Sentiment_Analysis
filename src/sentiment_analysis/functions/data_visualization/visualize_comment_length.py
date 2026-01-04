import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_comment_length(data_file_name, output_image_file):
    """
    Visualizes the relative frequency of words in sarcastic and non-sarcastic comments.

    Args:
        1. data_file_name (str): Path to the CSV file containing the data.
        2. output_image_file (str): Path to save the output visualization image.

    The Function:
        1. Reads the CSV file into a pandas DataFrame.
        2. Ensures all elements in the 'comment' and 'parent_comment' columns are strings.
        3. Removes rows where the number of words in the 'comment' column is greater than 30.
        4. Creates histograms for sarcastic and non-sarcastic comments.
        5. Saves the visualization as an image file.

    Returns:
        1. None
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file_name)

    # Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
    df['comment'] = df['comment'].astype(str)
    df['parent_comment'] = df['parent_comment'].astype(str)

    # Remove rows where the number of words in the 'comment' column is greater than 30.
    df = df[df['comment'].map(lambda x: len(x.split())) <= 30]

    # Create subplots for sarcastic and non-sarcastic comments.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Sarcastic comments.
    text_len = df[df['label'] == 1]['comment'].str.split().map(lambda x: len(x))
    counts, bins, _ = ax1.hist(text_len, color='red', bins=30, edgecolor='black', density=True)
    ax1.set_title('Sarcastic comment')

    # Non-sarcastic comments.
    text_len = df[df['label'] == 0]['comment'].str.split().map(lambda x: len(x))
    counts, bins, _ = ax2.hist(text_len, color='green', bins=30, edgecolor='black', density=True)
    ax2.set_title('Not Sarcastic comment')

    # Add a title and save the figure.
    fig.suptitle('Relative Frequency of Words in Comments')
    plt.savefig(output_image_file, dpi=300, bbox_inches='tight')


# Sample usage of the visualize_comment_length function.
if __name__ == "__main__":
    
    data_file_name = '../data/trimmed_training_data.csv'
    output_image_file = 'words_in_comments.png'
    visualize_comment_length(data_file_name, output_image_file)
    print(f"Visualization saved to {output_image_file}")
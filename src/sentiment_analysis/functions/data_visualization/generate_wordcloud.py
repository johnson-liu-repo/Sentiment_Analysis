import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_wordcloud(data_file_name, label, output_file_name):
    """
    Generates a word cloud for the given label and saves it to a file.

    Args:
        1. data_file_name (str): Path to the CSV file containing the data.
        2. label (int): The label for which to generate the word cloud (0 for non-sarcastic, 1 for sarcastic).
        3. output_file_name (str): The name of the file to save the word cloud.

    The Function:
        1. Reads the CSV file into a DataFrame.
        2. Ensures all elements in the 'comment' and 'parent_comment' columns are strings.
        3. Generates a word cloud for the specified label.
        4. Saves the figure to a file.
    
    Returns:
        None
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(data_file_name)

    # Ensure all elements in the 'comment' and 'parent_comment' columns are strings.
    df['comment'] = df['comment'].astype(str)
    df['parent_comment'] = df['parent_comment'].astype(str)

    # Generate the word cloud for the specified label.
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(df[df.label == label].comment))
    plt.imshow(wc, interpolation='bilinear')

    # Save the figure to a file.
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Define the output CSV file name.
    data_file_name = '../data/trimmed_training_data.csv'

    # Generate word cloud for non-sarcastic comments.
    generate_wordcloud(data_file_name, label=0, output_file_name='wordcloud_not_sarcastic.png')

    # Generate word cloud for sarcastic comments.
    generate_wordcloud(data_file_name, label=1, output_file_name='wordcloud_sarcastic.png')
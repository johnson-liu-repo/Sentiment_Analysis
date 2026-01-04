
import pandas as pd
import matplotlib.pyplot as plt




def filter_comments_by_length(comments, min_len=4, max_len=100):
    """
    Filters comments based on token count.

    Args:
        comments (list of str): Original comments.
        min_len (int): Minimum number of words required.
        max_len (int): Maximum number of words allowed.

    Returns:
        filtered_comments (list of str)
    """
    return [
        comment for comment in comments
        if min_len <= len(comment.split()) <= max_len
    ]



data_file = 'data/project_data/raw_data/trimmed_training_data.csv'

df = pd.read_csv(data_file)
comments_list = df['comment'].dropna().astype(str).tolist()

filtered_comments_list = filter_comments_by_length(comments_list, min_len = 4, max_len=100)

lengths = [len(sentence.split()) for sentence in filtered_comments_list]
# print(lengths)
# print(max(lengths))

plt.hist(lengths, bins=96, edgecolor='black')
plt.title("Distribution of Comment Lengths (in Tokens)")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.show()

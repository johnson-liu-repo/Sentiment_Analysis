import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open('data/sentences.txt', 'r') as f:
    comments = [ comment.strip() for comment in f.readlines() ]

with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
    trained_word_vectors = np.load(f, allow_pickle=True)[-1]


words_in_comments = [ comment.split() for comment in comments ]

words_in_comments = [
    [word.strip('.,!?()[]{}"\'').lower() for word in comment]
    for comment in words_in_comments
]

vectors_in_comments = [
    [ trained_word_vectors[word] for word in comment]
    for comment in words_in_comments
]

frechet_mean_vectors = [
    np.mean(np.array(vectors), axis=0) if len(vectors) > 0 else np.zeros(trained_word_vectors.shape[1])
    for vectors in vectors_in_comments
]

# Define the number of comments to visualize.
n = 5  # Change this value to the desired number of comments.

# Create a 3D plot.
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Loop through the first n comments.
for comment_idx in range(min(n, len(vectors_in_comments))):
    # Extract the word vectors and Frechet mean for the current comment.
    comment_vectors = vectors_in_comments[comment_idx]
    comment_frechet_mean = frechet_mean_vectors[comment_idx]

    # Ensure the vectors are in 3D (truncate or pad if necessary).
    comment_vectors = [vec[:3] for vec in comment_vectors]
    comment_frechet_mean = comment_frechet_mean[:3]

    # Plot the word vectors for the current comment.
    for vec in comment_vectors:
        ax.scatter(vec[0], vec[1], vec[2], color='blue', alpha=0.6)

        # Draw a dashed line from the word vector to the Frechet mean.
        ax.plot(
            [vec[0], comment_frechet_mean[0]],
            [vec[1], comment_frechet_mean[1]],
            [vec[2], comment_frechet_mean[2]],
            linestyle='--',
            color='gray',
            alpha=0.5
        )

    # Plot the Frechet mean for the current comment.
    ax.scatter(
        comment_frechet_mean[0],
        comment_frechet_mean[1],
        comment_frechet_mean[2],
        color='red',
        s=100
    )

# Set plot labels and title.
ax.set_title(f'Word Vectors and Frechet Means for First {n} Comments', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)

# Add a legend.
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels in the legend.
ax.legend(by_label.values(), by_label.keys())

# Show the plot.
plt.show()

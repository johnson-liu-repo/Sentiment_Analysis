


def frechet_mean(
        vectors_in_comments: list,
        word_vector_length: int
    ):
    """
    _summary_

    Args:
        1. vectors_in_comments (list): _description_
        2. word_vector_length (int): _description_

    The Function:
        1. 

    Returns:
        _type_: _description_
    """

    import numpy as np

    # Compute the Fréchet mean for each comment.
    # Use a zero vector as the fallback for empty comments.
    frechet_mean_for_each_comment = [
        np.mean(comment, axis=0) if len(comment) > 0 else np.zeros(word_vector_length) for comment in vectors_in_comments
    ]

    # Convert the list structure to an array.
    X = np.array(frechet_mean_for_each_comment)

    return X
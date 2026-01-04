



#######################################################################################################################
def g(x:float, x_max:float, alpha:float):
    """
    Computes a scaling factor based on the input value `x`, a maximum value `x_max`, and an exponent `alpha`.

    Args:
        1. x (float): The input value.
        2. x_max (float): The maximum value for scaling.
        3. alpha (float): The exponent used for scaling.

    The Function:
        1. If `x` is less than `x_max`, computes `(x / x_max) ** alpha`.
        2. Otherwise, returns 1.0.

    Returns:
        1. float: The computed scaling factor.
    """
    if x < x_max:
        return (x / x_max) ** alpha
    else:
        return 1.0


#######################################################################################################################
def descent(
        unique_words:list, 
        word_vectors:dict,
        word_vector_length:int,
        probabilities,
        x_max,
        alpha,
        eta,
        iter
    ):
    """
    Gradient descent algorithm to optimize word vectors based on probabilities.
    
    Args:
        1. word_vectors (dict): Dictionary of word vectors.
        2. probabilities (DataFrame): DataFrame of probabilities between words.
        3. x_max (float): Maximum value for the function g.
        4. alpha (float): Parameter for the function g.
        5. eta (float): Learning rate.
        6. iter (int): Number of iterations for gradient descent.

    The Function:
        1. Initializes a new dictionary for word vectors.
        2. Iteratively updates the word vectors based on the gradient of the cost function.
        3. Computes the cost function value at each iteration and stores it.

    Returns:
        1. J_over_time (list): List of cost function values over time.
        2. word_vectors_over_time (list): List of word vectors over time.
    """

    import numpy as np


    new_word_vectors = word_vectors.copy()

    word_vectors_over_time = []
    word_vectors_over_time.append(word_vectors)

    J_over_time = []

    for t in range(iter):
        print(f'Performing gradient descent ...\nIteration: {t}\n')
        for i in range(len(probabilities.columns)):
            a = np.zeros(word_vector_length)

            for j in range(len(probabilities.columns)):
                if i != j:
                    if probabilities.iloc[i][j] != 0:
                        dot_product = np.dot(word_vectors[unique_words[i]], word_vectors[unique_words[j]])

                        log_prob = np.log(probabilities.iloc[i][j])
                        g_value = g(probabilities.iloc[i][j], x_max, alpha)

                        a += (dot_product - log_prob) * g_value * word_vectors[unique_words[j]]

            new_word_vectors[unique_words[i]] = word_vectors[unique_words[i]] - eta * 2*a
        
        J = 0
        for i in range(len(probabilities.columns)):
            for j in range(len(probabilities.columns)):
                if i != j:
                    if probabilities.iloc[i][j] != 0:
                        dot_product = np.dot(new_word_vectors[unique_words[i]], new_word_vectors[unique_words[j]])
                        log_prob = np.log(probabilities.iloc[i][j])
                        g_value = g(probabilities.iloc[i][j], x_max, alpha)

                        J +=  g_value * (dot_product - log_prob) ** 2
            
        J_over_time.append(J)

        word_vectors = new_word_vectors.copy()
        word_vectors_over_time.append(word_vectors)

    return J_over_time, word_vectors_over_time
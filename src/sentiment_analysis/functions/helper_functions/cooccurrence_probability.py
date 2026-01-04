


from scipy.sparse import csr_matrix


def cooccurrence_probability_sparse(cooc_matrix_sparse):
    row_sums = np.array(cooc_matrix_sparse.sum(axis=1)).flatten()  # Sum over rows

    # Avoid divide-by-zero
    row_sums[row_sums == 0] = 1

    # Build a new sparse probability matrix
    data = cooc_matrix_sparse.data / row_sums[cooc.row]
    prob_matrix_sparse = csr_matrix(
        ( data, (cooc_matrix_sparse.row, cooc_matrix_sparse.col) ),
        shape=cooc_matrix_sparse.shape
    )

    return prob_matrix_sparse

    # from collections import defaultdict

    # totals = defaultdict(float)
    # probabilities = defaultdict(dict)

    # for i, j, val in zip(cooc_matrix.row, cooc_matrix.col, cooc_matrix.data):
    #     totals[i] += val

    # for i, j, val in zip(cooc_matrix.row, cooc_matrix.col, cooc_matrix.data):
    #     probabilities[i][j] = val / totals[i] if totals[i] > 0 else 0

    # return totals, probabilities

########################################################################################################
'''
def cooccurrence_probability(cooccurrence_matrix:dict):
    """
    Calculate the co-occurrence probability of terms in a given co-occurrence matrix.

    Args:
        1. cooccurrence_matrix (dict): A dictionary representing the co-occurrence matrix.

    The Function:
        1. Iterates through each row of the co-occurrence matrix.
        2. Calculates the total count of terms in each row.
        3. Computes the probability of each term given the row by dividing the count of each term by the total count.
        4. Stores the total counts and probabilities in separate dictionaries.

    Returns:
        1. totals: A dictionary with the total counts of each term.
        2. probabilities: A dictionary with the probabilities of each term given the row.
    """
    
    totals = {}
    probabilities = {}

    for row in cooccurrence_matrix:
        total_count = sum(cooccurrence_matrix[row].values())
        totals[row] = total_count

        row_probabilities = {}

        # Calculate the probability of each term in the row.
        for term, count in cooccurrence_matrix[row].items():
            if total_count > 0:
                row_probabilities[term] = count / total_count
            else:
                row_probabilities[term] = 0


        probabilities[row] = row_probabilities

    return totals, probabilities
'''
########################################################################################################

# Example usage:
if __name__ == "__main__":
    cooccurrence_csv_file_name = 'data/test_cooccurrence_matrix.csv'

    # Extract the co-occurrence matrix from the CSV file along with the words.
    cooccurrence_matrix = pd.read_csv(cooccurrence_csv_file_name, index_col=0).to_dict()
    terms = list(cooccurrence_matrix.keys())

    totals, probabilities = cooccurrence_probability(cooccurrence_matrix)

    # Save the probabilities to a CSV file.
    probabilities_df = pd.DataFrame(probabilities)
    probabilities_df.to_csv('data/test_cooccurrence_probabilities.csv', index=True, header=True)
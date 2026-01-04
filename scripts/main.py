

if __name__ == "__main__":
    ############################################################
    import os
    import sys
    import argparse
    ############################################################
    import numpy as np
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    # For consistency in testing.
    import random
    random.seed(1994)
    ############################################################


    ############################################################
    parser_description = "Sarcasm/Sentiment Analysis."
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument(
        "--part",
        help = "Tell the script which modular job to perform.",
        choices =   {
                        'preprocess',
                        'train_vectors',
                        'vectorize_comments',
                        'train_fnn',
                        'stack_word_vectors',
                        'train_cnn',
                        'train_rnn'
                    }
    )

    args = parser.parse_args()
    part = args.part

    ############################################################
    if part == 'preprocess':
        from scipy.sparse import save_npz
        
        from sentiment_analysis.functions.helper_functions import data_preprocessing
        ########################################################

        data_file_name='data/project_data/raw_data/trimmed_training_data.csv'
        print(f"Preprocessing training data extracting from {data_file_name}...")

        comments_limit=1010771 # Make sure there are enough rows in the CSV file, or else this will drop data.
        window_size=15
        min_len=4
        max_len=100
        min_word_count=50

        print(f"We are using a maximum of -{comments_limit}- data points...")
        print(f"Co-occurrence computed with a window size of -{window_size}-...")
        print(f"We are throwing out comments that have less than {min_len} words or greater than {max_len} words...")
        print(f"We are keeping only words that occur at least {min_word_count} times in the comments...")

        unique_words, cooc_matrix_sparse, filtered_comments, filtered_labels = (
            data_preprocessing.data_preprocessing(
                data_file_name,
                comments_limit,
                window_size,
                min_len,
                max_len,
                min_word_count
            )
        )

        save_dir = 'data/training_data/test_training_02/preprocessing/'

        unique_words_save_file = save_dir + 'unique_words.npy'
        cooccurrence_matrix_save_file = save_dir + 'cooccurrence_matrix.npz'
        text_save_file = save_dir + 'text.npy'
        labels_save_file = save_dir + 'labels.npy'

        print(f"Saving preprocessed data to files in {save_dir}...")        
        np.save( unique_words_save_file, unique_words )
        save_npz( cooccurrence_matrix_save_file, cooc_matrix_sparse )
        np.save( text_save_file, filtered_comments )
        np.save( labels_save_file, filtered_labels )
    ############################################################


    ############################################################
    elif part == 'train_vectors':
        from scipy.sparse import load_npz

        from sentiment_analysis.functions.machine_learning import LogBilinearModel
        ########################################################


        preprocess_save_dir = 'data/training_data/test_training_02/preprocessing/'
        unique_words_save_file = preprocess_save_dir + 'unique_words.npy'
        cooccurrence_matrix_save_file = preprocess_save_dir + 'cooccurrence_matrix.npz'
        text_save_file = preprocess_save_dir + 'text.npy'

        # Load the preprocessed data.
        print(f"Loading preprocessed data from files in {preprocess_save_dir}...")
        unique_words = np.load(unique_words_save_file, allow_pickle=True)
        cooc_matrix_sparse = load_npz(cooccurrence_matrix_save_file)

        training_save_dir = 'data/training_data/test_training_02/word_vector_training/'

        # Train the word vectors using PyTorch.
        print("Training word vectors through log bilinear regression...")
        word_vectors_over_time = LogBilinearModel.train_sparse_glove(
            cooc_sparse=cooc_matrix_sparse,
            embedding_dim=200,
            epochs=100,
            batch_size=256,
            learning_rate=0.001,
            x_max=100,
            alpha=0.75,
            num_workers=4,
            training_save_dir=training_save_dir,
            use_gpu=True,
            resume_checkpoint=True,
            checkpoint_interval=2
        )
    ############################################################

    ############################################################
    elif part == 'vectorize_comments':
        import torch
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentiment_analysis.functions.comment_representation import tf_idf_vectorization



        text_save_file = 'testing_scrap_misc/training_01/preprocessing/text.npy'
        text = np.load(text_save_file, allow_pickle=True)

        non_string_indices = [(i, t) for i, t in enumerate(text) if not isinstance(t, str)]
        if non_string_indices:
            print("Non-string values found in 'text' at the following indices:")
            for idx, val in non_string_indices:
                print(f"Index {idx}: {val} (type: {type(val)})")
        else:
            print("All values in 'text' are strings.")

        # ---------------------------
        # Step 1: Fit a TF-IDF Vectorizer
        # ---------------------------
        # This calculates both TF and IDF values across the corpus.
        print("Initializing TF-IDF vectorizer...")
        # This calculates both TF and IDF values across the corpus.
        print("Initializing TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(lowercase=True, tokenizer=str.split, token_pattern=None)
        vectorizer.fit(text)

        # ---------------------------
        # Step 2: Build the Vocabulary
        # Step 2: Build the Vocabulary
        # ---------------------------
        # Vocabulary: word → index
        print("Indexing unqiue words...")
        print("Indexing unqiue words...")
        word_to_idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}

        print("Loading the trained word vectors...")
        trained_word_vectors_file = 'testing_scrap_misc/training_01/word_vector_training/training_logs/weights_epoch_57.pt'
        word_vectors_matrix = torch.load(trained_word_vectors_file)

        print("Creating the vectorized comments...")
        output_file_name = 'testing_scrap_misc/training_01/vectorized_comments.npy'
        tf_idf_vectorization.vectorize_comments_with_tfidf(
            text, vectorizer, word_vectors_matrix, output_file_name )
# ----> should have the function return the data and save the data here in main?
    ############################################################

    ############################################################
    elif part == 'train_fnn':
        from sentiment_analysis.models import feedforward_neural_network



        print("Loading vectorized comments and corresponding labels...")
        vectorized_comments_file_name = 'testing_scrap_misc/training_01/fnn/vectorized_comments.npy'
        vectorized_comments = np.load(vectorized_comments_file_name)
        labels = np.load('testing_scrap_misc/training_01/preprocessing/labels.npy')

        num_zeros = np.sum(labels == 0)
        num_ones = np.sum(labels == 1)
        print(f"Number of 0 labels: {num_zeros}")
        print(f"Number of 1 labels: {num_ones}")

        print(f"Number of training datapoints...{len(labels)}")

        print("Training the feedforward neural network...")
        feedforward_neural_network.custom_fnn(
                vectorized_comments,
                labels,
                epochs = 100,
                patience = 10,
                batch_size = 32,
                learning_rate = 0.01,
                num_workers = 0,
                seed = 94
            )
    ############################################################
    
    ############################################################
    elif part == 'stack_word_vectors':
        import torch
        import torch.nn as nn



        text_save_file = 'testing_scrap_misc/scrap_02/text.npy'
        text = np.load(text_save_file, allow_pickle=True)

        def generate_word_to_idx(vocab, add_special_tokens=True):
            """
            Generate a word-to-index dictionary from a list of unique words.

            Args:
                vocab (List[str]): List of unique words.
                add_special_tokens (bool): Whether to add <PAD> and <UNK> tokens at the start.

            Returns:
                Dict[str, int]: Mapping from word to integer index.
            """
            word_to_idx = {}

            idx = 0

            if add_special_tokens:
                word_to_idx['<PAD>'] = idx
                idx += 1
                word_to_idx['<UNK>'] = idx
                idx += 1

            for word in vocab:
                if word not in word_to_idx:  # skip if already in (e.g. duplicates)
                    word_to_idx[word] = idx
                    idx += 1
            
            return word_to_idx

        def encode_sentence(sentence, word_to_idx, embedding_layer, sequence_length):
            """
            Encode a sentence into a stacked word vector matrix using nn.Embedding.

            Args:
                sentence (str): The input sentence.
                word_to_idx (Dict[str, int]): Mapping from word to index.
                word_vectors_matrix:
                sequence_length (int): Desired fixed length.

            Returns:
                torch.Tensor: Shape [sequence_length, embedding_dim]
            """
            tokens = sentence.lower().split()
            indices = [word_to_idx.get(tok, word_to_idx.get('<UNK>', 0)) for tok in tokens]

            # Pad or truncate
            if len(indices) < sequence_length:
                indices += [word_to_idx.get('<PAD>', 0)] * (sequence_length - len(indices))
            else:
                indices = indices[:sequence_length]

            idx_tensor = torch.tensor(indices, dtype=torch.long)  # shape: [sequence_length]
            return embedding_layer(idx_tensor)  # shape: [sequence_length, embedding_dim]

            # matrix = []

            # for index in indices:
            #     # print(index)
            #     matrix.append(word_vectors_matrix[index])

            # return matrix

        print("Extracting list of unique words from file...")
        unique_words = np.load('testing_scrap_misc/scrap_02/unique_words.npy')

        print("Creating dictionary to map word to index...")
        word_to_idx = generate_word_to_idx(unique_words)

        print("Loading trained word vectors...")
        trained_word_vectors_file = 'testing_scrap_misc/scrap_02/training_logs/final_word_vectors.pt'
        word_vectors_matrix = torch.load(trained_word_vectors_file)

        print("Adding <PAD> and <UNK> to word_vectors_matrix...")
        # If word_vectors_matrix is a torch.Tensor of shape [vocab_size, embedding_dim]
        num_special_tokens = 2  # <PAD> and <UNK>
        embedding_dim = word_vectors_matrix.shape[1]
        if word_vectors_matrix.shape[0] < len(word_to_idx):
            extra_rows = torch.zeros((len(word_to_idx) - word_vectors_matrix.shape[0], embedding_dim))
            word_vectors_matrix = torch.cat([extra_rows, word_vectors_matrix], dim=0)

        print("Converting word vectors matrix to torch.nn.Embedding layer...")
        # Not sure why we need to do this...
        embedding_layer = nn.Embedding.from_pretrained(word_vectors_matrix, padding_idx=word_to_idx['<PAD>'])

        # matrix = encode_sentence(text[0], word_to_idx, embedding_layer, 5)

        # Converting all sentences to matrices of stacked word vectors.
        stack_size = 20
        sentence_matrices = [ encode_sentence(
                                sentence,
                                word_to_idx,
                                embedding_layer,
                                stack_size ) for sentence in text ]
    
        # Save to file.
        stacked_word_vectors_save_file_name = 'testing_scrap_misc/scrap_02/cnn/stacked_word_vectors.npy'
        np.save(stacked_word_vectors_save_file_name, sentence_matrices)
    ############################################################

    ############################################################
    elif part == 'train_cnn':
        pass
    ############################################################

    ############################################################
        from sentiment_analysis.models import feedforward_neural_network

        labels = np.load('testing_scrap_misc/training_01/preprocessing/labels.npy')

        num_zeros = np.sum(labels == 0)
        num_ones = np.sum(labels == 1)
        print(f"Number of 0 labels: {num_zeros}")
        print(f"Number of 1 labels: {num_ones}")

        print(f"Number of training datapoints...{len(labels)}")

        print("Training the feedforward neural network...")
        feedforward_neural_network.custom_fnn(
                vectorized_comments,
                labels,
                epochs = 100,
                patience = 10,
                batch_size = 32,
                learning_rate = 0.01,
                num_workers = 0,
                seed = 94
            )
    ############################################################
    
    ############################################################
    elif part == 'stack_word_vectors':
        import torch
        import torch.nn as nn



        text_save_file = 'testing_scrap_misc/scrap_02/text.npy'
        text = np.load(text_save_file, allow_pickle=True)

        def generate_word_to_idx(vocab, add_special_tokens=True):
            """
            Generate a word-to-index dictionary from a list of unique words.

            Args:
                vocab (List[str]): List of unique words.
                add_special_tokens (bool): Whether to add <PAD> and <UNK> tokens at the start.

            Returns:
                Dict[str, int]: Mapping from word to integer index.
            """
            word_to_idx = {}

            idx = 0

            if add_special_tokens:
                word_to_idx['<PAD>'] = idx
                idx += 1
                word_to_idx['<UNK>'] = idx
                idx += 1

            for word in vocab:
                if word not in word_to_idx:  # skip if already in (e.g. duplicates)
                    word_to_idx[word] = idx
                    idx += 1
            
            return word_to_idx

        def encode_sentence(sentence, word_to_idx, embedding_layer, sequence_length):
            """
            Encode a sentence into a stacked word vector matrix using nn.Embedding.

            Args:
                sentence (str): The input sentence.
                word_to_idx (Dict[str, int]): Mapping from word to index.
                word_vectors_matrix:
                sequence_length (int): Desired fixed length.

            Returns:
                torch.Tensor: Shape [sequence_length, embedding_dim]
            """
            tokens = sentence.lower().split()
            indices = [word_to_idx.get(tok, word_to_idx.get('<UNK>', 0)) for tok in tokens]

            # Pad or truncate
            if len(indices) < sequence_length:
                indices += [word_to_idx.get('<PAD>', 0)] * (sequence_length - len(indices))
            else:
                indices = indices[:sequence_length]

            idx_tensor = torch.tensor(indices, dtype=torch.long)  # shape: [sequence_length]
            return embedding_layer(idx_tensor)  # shape: [sequence_length, embedding_dim]

            # matrix = []

            # for index in indices:
            #     # print(index)
            #     matrix.append(word_vectors_matrix[index])

            # return matrix

        print("Extracting list of unique words from file...")
        unique_words = np.load('testing_scrap_misc/scrap_02/unique_words.npy')

        print("Creating dictionary to map word to index...")
        word_to_idx = generate_word_to_idx(unique_words)

        print("Loading trained word vectors...")
        trained_word_vectors_file = 'testing_scrap_misc/scrap_02/training_logs/final_word_vectors.pt'
        word_vectors_matrix = torch.load(trained_word_vectors_file)

        print("Adding <PAD> and <UNK> to word_vectors_matrix...")
        # If word_vectors_matrix is a torch.Tensor of shape [vocab_size, embedding_dim]
        num_special_tokens = 2  # <PAD> and <UNK>
        embedding_dim = word_vectors_matrix.shape[1]
        if word_vectors_matrix.shape[0] < len(word_to_idx):
            extra_rows = torch.zeros((len(word_to_idx) - word_vectors_matrix.shape[0], embedding_dim))
            word_vectors_matrix = torch.cat([extra_rows, word_vectors_matrix], dim=0)

        print("Converting word vectors matrix to torch.nn.Embedding layer...")
        # Not sure why we need to do this...
        embedding_layer = nn.Embedding.from_pretrained(word_vectors_matrix, padding_idx=word_to_idx['<PAD>'])

        # matrix = encode_sentence(text[0], word_to_idx, embedding_layer, 5)

        # Converting all sentences to matrices of stacked word vectors.
        stack_size = 20
        sentence_matrices = [ encode_sentence(
                                sentence,
                                word_to_idx,
                                embedding_layer,
                                stack_size ) for sentence in text ]
    
        # Save to file.
        stacked_word_vectors_save_file_name = 'testing_scrap_misc/scrap_02/cnn/stacked_word_vectors.npy'
        np.save(stacked_word_vectors_save_file_name, sentence_matrices)
    ############################################################

    ############################################################
    elif part == 'train_cnn':
        pass
    ############################################################





######################################################################################################
# Notes
#######
# 1. Three things to try:
#   a. FNN
#      Feedforward Neural Network using aggregated word vectors for each comment.
            # Pros:
                # Simple and fast to train
                # Works well with aggregated embeddings (e.g., TF-IDF-weighted GloVe vectors)
                # Fewer parameters → less overfitting on small datasets
            # Cons:
                # Ignores word order and syntax
                # Cannot model phrases like “I just love waiting in line” where sarcasm depends on context
                # Performance plateaus if the model lacks sequential information
            # Use FNNs if:
                # You're using averaged or TF-IDF-weighted embeddings
                # You want a fast, simple baseline
#   b. CNN
#      Convolutional Neural Networks using stacks of word vectors with padding for shorter comments.
            # Pros:
                # Captures local n-gram patterns that are useful for sarcasm (e.g., “great job”, “love that”)
                # Faster to train than RNNs
                # Some resistance to word order noise
            # Cons:
                # Limited to local context (can’t model long-range dependencies)
                # Can miss sarcasm that builds over multiple clauses
            # Use CNNs if:
                # You have access to sequence-preserving embeddings (e.g., [sequence_length, embedding_dim])
                # Sarcasm is often expressed in short phrases
#   c. RNN
#      Recurrent Neural Networks feeding word vectors in sequentially.
            # Pros:
                # Models word order and long-range dependencies
                # Good for subtle sarcasm that builds over the sentence
                # LSTM/GRU handles negations, sentiment shifts, and intensifiers (e.g., “Oh yeah, that’s exactly what I wanted…”)
            # Cons:
                # Slower training than CNNs/FNNs
                # More prone to overfitting, especially with small data
                # Vanilla RNNs can struggle with long sequences (use LSTM or GRU)
            # Use RNNs if:
                # Sarcasm depends on word order or context buildup
                # You’re okay with slightly longer training times
# 
# 1. Revise the structure/logic of the argument handling in the beginning parts of main.py.
#    Make it so that the user can run each part of the code sequentially.
#
# 1. Go through __name__ == "__main__" in each file to make sure the code still works / is up-to-date.
#
# ------------------------------------------------------------------- #
# ---------------------- Possible Improvements ---------------------- #
# ------------------------------------------------------------------- #
# 1. CNNs over Word Embeddings
#       - Cannot squash comments into uniform comment vectors. Need full comment and padding.
#       - Apply 1D convolution filters to detect n-gram patterns (e.g., sarcasm markers).
#       - Captures local word patterns
# 2. Recurrent Neural Networks (RNNs / LSTMs / GRUs)
#       Feed word embeddings sequentially into an RNN to produce a context-sensitive representation.
#       Captures word order
#       Maintains directional context
#       Final hidden state or an attention-weighted sum can represent the whole comment

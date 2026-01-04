
### Old applicaiton. Switched to using Torch.

def custom_fnn(
        X:list,
        labels:list
    ):
    """_summary_

    Args:
        X (list): _description_
        labels (list): _description_
    """

    ##############################################################################################################
    import os
    ##############################################################################################################
    import numpy as np
    ##############################################################################################################
    import keras
    from keras import Model, Input
    from keras import layers
    ##############################################################################################################
    from tqdm.keras import TqdmCallback
    ##############################################################################################################
    import tensorflow as tf  # Add this import
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    ##############################################################################################################
    # Define the model sequentially.
    ################################
    # model = keras.Sequential(
    #     [
    #         keras.layers.Input(shape=(X.shape[1],)),
    #         keras.layers.Dense(512, activation="relu", name='dense_0'),
    #         # keras.layers.Dense(256, activation="relu", name='dense_1'),
    #         keras.layers.Dense(1, activation="sigmoid", name='output')  # For binary classification
    #     ]
    # )
    ##############################################################################################################


    ##############################################################################################################
    # Define the model using functional API.
    ########################################
    input1 = Input(shape=(X.shape[1],))
    layer1 = layers.Dense(8, activation='softmax', use_bias=False)(input1)
    layer2 = layers.Dense(8, activation='softmax', use_bias=False)(layer1)
    layer3 = layers.Dense(8, activation='softmax', use_bias=False)(layer2)
    output1 = layers.Dense(1, activation='softmax', use_bias=False)(layer3)
    ##############################################################################################################


    model = Model(inputs=input1, outputs=[output1])
    model.summary()

    # Create figure of neural network.
    keras.utils.plot_model(model, show_shapes=True)

    # Compile the model.
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Save weights for each epoch.
    checkpoint_dir = 'testing_scrap_misc/scrap_02/fnn/nn_weights_01/'
    checkpoint_filepath = os.path.join(checkpoint_dir, 'weights_{epoch:02d}.weights.h5')

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_freq='epoch'
    )

    # Train the model.
    model.fit(
        X,
        labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[model_checkpoint_callback, TqdmCallback()],
    )

    model.save("testing_scrap_misc/scrap_02/fnn/my_model.keras")



# Example usage of the custom_nn function.
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from sentiment_analysis.functions.helper_functions import frechet_mean

    # Get the trained word vectors from the file.
    with open('testing_scrap_misc/scrap_data_02/word_vectors_over_time.npy', 'rb') as f:
        trained_word_vectors = np.load(f, allow_pickle=True)[-1]


    data_file_name = 'data/project_data/raw_data/trimmed_training_data.csv'

    comments_limit = 10
    # Get the comments for which we want to train the neural network on.
    data = pd.read_csv(data_file_name)[:comments_limit]['comments']

    # Split each comment into their component words.
    words_in_comments = [ comment.split() for comment in data ]

    # Throw away any punctuation that are attached to the words.
    words_in_comments = [
        [ word.strip('.,!?()[]{}"\'').lower() for word in comment]
        for comment in words_in_comments
    ]

    # Retrieve the vector representation of each word in each comment.
    vectors_in_comments = [
        [ trained_word_vectors[word] for word in comment]
        for comment in words_in_comments
    ]

    # # Compute the Frechet mean for each comment.
    # # The Frechet mean in our context is just the mean of all of the word vectors in a comment.
    X = frechet_mean.frechet_mean(vectors_in_comments)

    labels = np.random.randint(0, 2, size=(X.shape[0],))

    # Train the neural network.
    custom_nn(trained_word_vectors, X, labels)

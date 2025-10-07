
<div align="justify">


# Sarcasm Detection in Social Media Text

<p align="center">
———Work in progress———
</p>

This project investigates sarcasm detection in social media text, with Reddit comments as a primary data source.
Sarcasm presents a distinctive challenge for natural language processing (NLP), since it often relies on contextual cues, irony, and pragmatic inference.
The approach combines word vector representations (trained locally or using pre-trained embeddings from libraries like Hugging Face's Transformers) with neural betwork architectures (feedforward, convolutional, and recurrent models) to classify comments as sarcastic or non-sarcastic.

Johnson Liu\
GitHub: [@johnson-liu-code](https://github.com/johnson-liu-code)



## Contents

1. [Project Overview](#project-overview)

2. [Background](#background)\
    2.1. [Natural Language Processing (NLP)](#natural-language-processing---nlp)\
&nbsp;&nbsp;&nbsp;2.1.1 [Word Embedding](#word-embedding)\
&nbsp;&nbsp;&nbsp;2.1.2 [Word2Vec](#word2vec)\
&nbsp;&nbsp;&nbsp;2.1.3 [GloVe](#glove)\
    2.2. [Data Used](#data-used)\
    2.3. [Mathematical Foundations](#mathematical-foundations)

3. [Workflow](#workflow)\
    3.1. [Data Preprocessing]()\
    3.2. [Word Vector Training]()\
    3.3. [Neural Network Training]()\
&nbsp;&nbsp;&nbsp;3.3.1 [Feedforward Neural Network]()\
&nbsp;&nbsp;&nbsp;3.3.2 [Convolutional Neural Network]()\
&nbsp;&nbsp;&nbsp;3.3.3 [Recurrent Neural Network]()

4. [Resources and Helpful References](#resources-and-helpful-references)\
    4.1. [Data](#data)\
    4.2. [Theoretical Foundations](#theoretical-foundations)\
&nbsp;&nbsp;&nbsp;4.2.1. [Natural Language Processing](#natural-language-processing)\
&nbsp;&nbsp;&nbsp;4.2.2. [word2vec Model](#word2vec-model)\
&nbsp;&nbsp;&nbsp;4.2.3. [GloVe Model](#glove-model)\
    4.3. [Sample Works](#sample-works)\
    4.4. [Documentation and Tutorials](#documentation-and-tutorials)\
&nbsp;&nbsp;&nbsp;4.4.1. [Neural Networks](#neural-networks)\
&nbsp;&nbsp;&nbsp;4.4.2. [Gensim](#gensim)\
    4.5. [Other Theoretical Backgrounds](#other-theoretical-backgrounds)\
    4.6. [Mathematical References](#mathematical-references)\
    4.7. [Graphical Visualization Guides](#graphical-visualization-guides)


5. [Future Direction and Possible Improvements](#future-direction-and-possible-improvements)


---



## Project Overview

This project aims to experiment with applying natural language processing techniques to classify comments found in informal, conversational, or otherwise casual text found through social media as either sarcastic or not sarcastic.
The current scope of this project involves training neural network models on prelabeled comments from a dataset retrieved from Kaggle and using the trained models to predict the presence of sarcastic intent in Reddit comments.
A possible future extension is applying sarcasm detection to broader sentiment analysis to general sentiment analysis in social media posts.
Another possible avenue to explore is sentiment analysis through multimodal input ( including images alongside text ).




<!-- --- Check this --- this is Johnson's rambling ... not sure if it is correct
------------------
The general idea is that we scan through a large amount of text, for each unique word in the vocabulary, tally up the number of times each other word (*context* words) in the vocabulary shows up within the local vicinity of the *target* word.
Take the word 'cake' as the target word, and consider the context words 'strawberry' and 'computer'.
A human reader would understand connections between the target word and the context words in different contexts.
The model learns these relationships through the frequency in which pairs of words show up in the training text.
By reading through a set of text, we would expect that the word 'strawberry' to show up near the word 'cake' more often than the word 'computer' would show up near 'cake'.
Although the model does not understand the specific meanings of 'cake', 'strawberry', and 'computer', it sees that the relationship between the words 'cake' and 'strawberry' is relatively stronger than the relationship between the words 'cake' and 'computer'.
The model does not understand why 'cake' is more related to 'strawberry' than 'computer' the same way that a human would, but it does recognize the relative strengths of the relationships between these pairs of words.

word vectors -
Consider the sentences 'this ice cream tastes great' and 'you need to take the general chemistry course before the physical chemistry course'. A simple sentence embedding can be obtained by averaging (or summing) its word vectors. This yields a fixed-length vector that roughly captures the overall meaning, though it ignores word order. -->


## Background


### Natural Language Processing (NLP)

Natural language processing (NLP) is a subfield of artificial intelligence focused on enabling computers to process and understand human language.
Sentiment analysis, an important NLP application, involves classifying text by emotional polarity ( positive, negative, or neutral ) or affective tone.
Sarcasm complicates sentiment analysis, as sarcastic expressions can invert literal sentiment ( e.g. “great job!” could be an insult in the context of a negative situation ).

—More on NPL here—



#### Word Embeddings
 
Word embeddings map words ( or tokens ) to real vectors in $n$-dimensional space.
These embeddings ( ideally ) capture semantic similarity between words by assigning similar words vectors that are close together in vector space.
These vectors are initialized ( often randomly ) and then trained so that words used in similar contexts end up with similar embeddings.
Traditional embeddings assign one vector per word type.
More recent contextual embeddings (e.g., ELMo, BERT) allow the same word to have multiple vectors depending on usage.



#### Word2Vec

Word2Vec is a model that learns word meanings based on the words that commonly appear around them.
For example, the words “*doctor*” and “*nurse*” often show up in similar contexts, so Word2Vec places them close together in vector space.
This helps the model understand how words relate to one another.
In sarcasm detection, understanding context is key, and Word2Vec helps capture that by noticing patterns in how words are used together.



#### GloVe

Global Vectors for Word Representation (GloVe), is another word embedding model.
Unlike Word2Vec, which focuses on small windows of text, GloVe looks at word co-occurrence across an entire dataset.
This gives it a more global understanding of word relationships.
GloVe can help highlight unusual or ironic word pairings—something that often happens in sarcastic sentences.

Some users on Kaggle have successfully used Word2Vec and GloVe to detect sarcasm in news headlines.
Some public Kaggle kernels ( e.g., detecting sarcasm in headlines ) report accuracies in the low- to mid-80% range.
In one project, the users trained their model using headlines from The Onion ( a satirical site ) and The Huffington Post ( a more traditional news source ).

---insert link/reference here

We can apply a similar approach in our project using Reddit posts.
Instead of news headlines, we’ll train our model on a dataset of labeled Reddit comments, using Word2Vec or GloVe to create embeddings.
This will allow our classifier to learn the difference between sincere and sarcastic language based on context, just like in the Kaggle example.




#### Data Used

The data that was used in this project was taken from the *Sarcasm on Reddit* notebook from Kaggle.com ( see [2.1 Data](#data) ).


---


## Variable Definitions

See reference [2.2.3. GloVe model](#theoretical-foundations) in the Theoretical Foundations section.

1. $C$ is the collection of all comments in the dataset.

2. $c \in C$ is comment.

3. $k$ is the number of words contained within comment $c$.

4. $W$ is the set of all unique words that appear in the corpus ( the collection of written text ).

5. $N = |W|$ is the cardinality of $W$.

6. $V$ is the set of all word vectors.

7. $v \in V$ is a word vector.
There is a word vector $v_i \in V$ associated with each $w_i \in W$.
Let $g$ be the mapping from $W$ to $V$.
Then

$$V \ = \ g(W)$$
$$ v_i \ = \ g(w_i) .$$

8. $m = \text{dim}(v)$ is the dimensionality of the word vectors.

9. $X$ is the co-occurrence matrix for every possible pair of words $(w_i, w_j) \in W^2$.

10. $X_{ij}$ is the $i$-th row, $j$-th column entry in $X$ which gives the number of times word $w_j$ appears in the context of word $w_i$.

11. $X_i$ is the summation over all $X_{ik}$ for $k \leq N$. In other words,

$$X_i \ \ = \ \ \sum_{\substack{k = 1, \\ k \neq i}}^{N} X_{ik} \ \ . $$

12. $P$ is the co-occurrence probabilities matrix for every possible pair of words $(w_i, w_j) \in W^2$.

13. $P_{ij}$ is the $i$-th row, $j$-th column entry in $P$ which gives the probability that word $w_j$ appears in the context of word $w_i$.
$P_{ij}$ can also be interpreted as the chance of finding word $w_j$ given that you have word $w_i$.
This probability is defined as

$$P_{ij} \ \ = \ \ P(\text{word}_j|\text{word}_i) \ \ \\ $$

$$P_{ij} \ \ = \ \ \frac{X_{ij}}{X_i} \ \ . $$


---


## Mathematical Foundations


#### Fréchet Mean
Given a set of vectors, the Fréchet mean is a single point that averages over all of the vectors.
The mean acts as a point of central tendency for the word vectors associated with a specific comment.


---


### Neural Networks

#### Multi-layer Perceptron (MLP)



#### Backpropagation





## Workflow


#### Collect Comments Classified as Sarcastic or Not Sarcastic

The dataset for this project was derived from the *Sarcasm on Reddit* dataset available on Kaggle, which compiles Reddit comments labeled as sarcastic or non-sarcastic by crowd-sourced annotation.
Datasets of this type are widely used in computational linguistics research to benchmark sarcasm detection tasks ( [Kaggle Sarcasm Dataset](https://www.kaggle.com/datasets/danofer/sarcasm) ).

The collected dataset is split into three partitions, consistent with standard practices in machine learning:

- **Training set**<br>
Used to fit the model to the labeled examples.

- **Validation set**<br>
Used to tune hyperparameters (e.g., learning rate, hidden layer size, dropout rates).

- **Test set**<br>
Reserved for final unbiased evaluation on unseen data.

This procedure follows the canonical machine learning paradigm of dataset partitioning, as outlined in [Wikipedia: Training, validation, and test sets](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets).


---



### GloVe Model

The **Global Vectors for word representation ( GloVe )** algorithm is used to create word embeddings that capture semantic relationships between words.
The general workflow of GloVe ( [Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf) ) is as follows:



#### Preprocess the Raw Data

1. **Stopword removal**<br>
Stopwords are common function words (e.g., “the”, “is”) that occur with high frequency but provide little semantic value when distinguishing sarcasm from non-sarcasm.
They are typically removed to reduce noise ( [Wikipedia: Stop words](https://en.wikipedia.org/wiki/Stop_words) ).

2. **Punctuation removal**<br>
While punctuation (e.g., “Yeah, right!”) can signal sarcasm, in this simplified project it is excluded.
Future extensions could explicitly model punctuation and emoji as sarcasm markers ( [Davidov et al., 2010](https://aclanthology.org/W10-2914/) ).

3. **Miscellaneous symbols**<br>
Emoticons, foreign-language words, and mathematical notation are treated as noise and removed.

4. **Vocabulary construction**<br>
Define the vocabulary as the set of unique tokens appearing in the dataset.
This vocabulary is central to building co-occurrence statistics.



#### Compute the Co-occurrence Matrix

- The **co-occurrence matrix** counts how often word $w_i$ appears in the context of another word $w_j$.
The context is defined by a sliding window ( e.g., ±5 tokens ).

- Unlike earlier probabilistic methods, GloVe directly uses **co-occurrence frequencies** rather than probabilities.
This distinction is crucial, as the algorithm is based on factorizing a weighted least-squares objective that relates co-occurrence frequencies to embedding geometry ( [Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf) ).  



#### Train the Word Vectors

- Each token is represented as a vector of fixed dimension (e.g., 50–300).
Larger dimensionality can capture richer semantic nuances but increases computation cost.

- Training minimizes a weighted least-squares loss designed to enforce that dot products of word vectors approximate the logarithm of co-occurrence counts.

- Evaluation of learned embeddings is performed using:

  - **Word similarity tests**<br>
  Measuring pairwise cosine similarity between embeddings to check semantic closeness.

  - **Word analogy tests**<br>
  Using vector arithmetic ( e.g., “king” - “ma” + “woman” ≈ “queen” ) to assess how well embeddings capture linear semantic regularities ( [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781) ).  


---


### Train the Neural Networks


Once embeddings are available, they serve as inputs to classification models tasked with distinguishing sarcastic from non-sarcastic comments.
Three architectures are explored: **Feedforward Neural Networks (FNNs)**, **Convolutional Neural Networks (CNNs)**, and **Recurrent Neural Networks (RNNs)**.


#### <ins>Feedforward Neural Network (FNN)</ins>


1. **Input representation**<br>
Since comments are variable length, each must be transformed into a fixed-size vector.
Strategies include:

   - **Averaging word vectors ( Fréchet mean )**<br>
   Compute the component-wise mean of all word embeddings in a comment.
   This provides a simple “bag of embeddings” representation.

   - **TF-IDF weighted averaging**<br>
   Scale embeddings by their term frequency–inverse document frequency scores so that rare but informative words have more influence ( [Wikipedia: TF–IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) ).

2. **Network architecture**
   - Input layer corresponding to the embedding dimension.
   - One or more hidden layers with nonlinear activation (ReLU or tanh).
   - Output layer with a sigmoid activation to predict probability of sarcasm.

3. **Loss function**<br>
Binary Cross-Entropy (BCE), defined as:

$$\textrm{BCE}(p,y) = -(y \cdot \log(p) + (1-y) \cdot \log(1-p))$$

4. **Regularization techniques**
   - **Batch Normalization** to stabilize training.  
   - **Dropout** and **weight decay** to prevent overfitting ( [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html) ).

5. **Troubleshooting and diagnostics**:
   - **Precision/recall metrics** highlight imbalances between sarcastic and non-sarcastic predictions.
   - **Confidence histograms** reveal uncertainty when probabilities cluster near 0.5.
   - **PCA analysis** of learned feature space can show whether embeddings form separable clusters by class.

#### <ins>Convolutional Neural Network (CNN)</ins>

- CNNs apply convolutional filters over sequences of embeddings to extract local n-gram features. For sarcasm, this can capture short sarcastic cues like “yeah right” or “as if.”

- CNNs have proven effective in sentence classification tasks ( [Kim, 2014](https://arxiv.org/abs/1408.5882) ).


#### <ins>Recurrent Neural Network (RNN)</ins>

- RNNs (and their variants such as LSTMs and GRUs) process sequences word by word, maintaining memory of prior context. This is particularly useful in sarcasm, where long-range dependencies (setup–punchline structures) exist.

- RNNs remain a standard choice for modeling sequential dependencies ( [Wikipedia: Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) ).


<!-- ##### <ins>Collect comments classified as sarcastic/not sarcastic</ins> -->

<!-- 1. Collect a dataset of comments that have been labeled as either sarcastic or not sarcastic.
    1. The dataset used in this project was taken from the *Sarcasm on Reddit* notebook from Kaggle.com ( see [2.1 Data](#data) in the References and Helpful Resources section ).

1. Split the dataset into training, validation, and test sets.
    1. The training set is used to train the model.
    1. The validation set is used to tune the model's hyperparameters.
    1. The test set is used to evaluate the model's performance on unseen data.

1.  -->

<!-- ##### <ins>GloVe model</ins> -->

<!-- General workflow when applying the GloVe model ... -->
<!-- The general workflow when applying the GloVe model to train word vectors is as follows:

1. Process raw data
    1. Remove stopwords, punctuation, and other words and/or characters that are deemed not important to the context of a comment.

        1. Stopwords are commonly used words within a language that appear frequently in many contexts. These words are assumed to be not important when discerning between comments that are meant to be sarcastic and comments that are not meant to be sarcastic.

        1. Although punctuations might be important in lending context to comments with sarcastic tones, we will assume that they do not in order to simplify our project. Future works might look into the importance of punctuations in sarcastic comments as well as developing a way to incorporate punctuations into the classification of such comments.

        1. Miscellaneous words/phrases/characters that are troublesome to work with ( such as emoticons, words from other languages, mathematical notation, etc. ) or otherwise deemed unimportant can be treated like stopwords and punctuations and removed from the data.

    1. Generate the vocabulary.

        1. The vocabulary is the set containing all of the words present within the corpus ( dataset ).

1. Compute the co-occurrence probability matrix.
    #################################################################################
    ####### NEED TO FIX THIS -                                                #######
    ####### WE DO NOT USE THE PROBABILITIES, WE USE THE FREQUENCIES DIRECTLY. #######
    #################################################################################
    #################################################################################
    ####### NEED TO FIX THIS -                                                #######
    ####### WE DO NOT USE THE PROBABILITIES, WE USE THE FREQUENCIES DIRECTLY. #######
    #################################################################################
    1. Compute the co-occurence matrix with a specified context window size.

        1. The co-occurence matrix tabulates the amount of times word $w_i$ appears in the context of word $w_j$.

        1. The context window is the range centered on the target word from which we count the number of times the context word shows up.

    1. Compute the pairwise co-occurrence probabilities using the co-occurrence matrix.

        1. ... text here about computing co-occurrence probabilities ...

1. Train word vectors for each unique word.

    1. We have to decide on the dimensionality of each word vector. Greater dimensionality might capture more nuances between words, but will also increase the demand on computational resources.

    1. ...text here...

    1. ...text here...

    1. Difficulties training the word vectors ... not sure what's going on ... could be due to small corpus (although there are hundreds of thousands of comments -- check this)

    1. Word similarity tests

    1. Word analogy tests (cosine similarity)

    1. Amount of tokens too little

    1. Trim vocabulary?

1. Train the neural networks.

    1. Feedforward Neural Network (FNN)
    1. Word similarity tests

    1. Word analogy tests

    1. Amount of tokens too little

    1. Trim vocabulary?

1. Train the neural networks.

    1. Feedforward Neural Network (FNN)

        1. In order to able to pass our data as input into the neural network, the input shape across all comments must be uniform. Since each comment in the dataset can have varying number of words, we have to decide on a way of aggregating all of the words in a comment into a single input.
        1. In order to able to pass our data as input into the neural network, the input shape across all comments must be uniform. Since each comment in the dataset can have varying number of words, we have to decide on a way of aggregating all of the words in a comment into a single input.

            1. ... text here on one possibility ... using the Frechet mean ... \
            One possibility is to take the average of all of the word vectors contained within a comment.
            1. ... text here on one possibility ... using the Frechet mean ... \
            One possibility is to take the average of all of the word vectors contained within a comment.

                1. ... text here about the Frechet mean being a measure of central tendency ...
                1. ... text here about the Frechet mean being a measure of central tendency ...

                1. In the context of this project, the Frechet mean for all of the word vectors associated with a specific comment is simply the arithmetic mean of the collection of vectors. This mean is found by taking the component-wise mean of each vector component.
                1. In the context of this project, the Frechet mean for all of the word vectors associated with a specific comment is simply the arithmetic mean of the collection of vectors. This mean is found by taking the component-wise mean of each vector component.

                1. ... TF-IDF ... weighting word vectors ...

        1. ... text here on structure of neural network ...
                1. ... TF-IDF ... weighting word vectors ...

        1. ... text here on structure of neural network ...

        1. Binary Cross Entropy
            BCE(p,y)=−(y⋅log(p)+(1−y)⋅log(1−p))

        1. Batch Normalization
        1. Regularization
            1. Dropout
            1. Weight decay
        
        1. Training struggles ...
            1. Per-Class Precision And Recall (Validation Set)
                <img src="figures/training_troubleshooting/fnn/per_class_precision_recall.png.png" width="50%" height="50%"/>

            1. Prediction Confidence Histogram
                <img src="figures/training_troubleshooting/fnn/prediction_confidence_histogram.png.png" width="50%" height="50%"/>


            1. Your model has learned some separation, but not enough to make confident predictions.

            Many predictions fall near the decision threshold (0.5), which is where models are most unsure.


            1. PCA analysis
            <img src="figures/training_troubleshooting/fnn/PCA_projection_of_real_feature_space.png.png" width="50%" height="50%"/>

            1. 

    1. Convolutional Neural Network (CNN)

    1. Recurrent Neural Network (RNN) -->


<!-- ##### <ins>Collect comments and classify them</ins>

1. ... text here ...

1. ... text here ... -->



## References and Helpful Resources


### Data

1. [_Sarcasm on Reddit_](https://www.kaggle.com/datasets/danofer/sarcasm/data?select=train-balanced-sarcasm.csv)<br>
— Kaggle dataset with Reddit posts classified as either sarcastic or not sarcastic ( _Kaggle_ website ).


---


### Theoretical Foundations


#### <ins>Natural Language Processing</ins>

1. [_Natural language processing_](https://en.wikipedia.org/wiki/Natural_language_processing)<br>
— Wikipedia article.

1. [_Text Classification & Sentiment Analysis_](https://mlarchive.com/natural-language-processing/text-classification-sentiment-analysis/)<br>
— Introduction to natural language processing techniques ( _ML Archive_ website ).

1. [_Text Embeddings: Comprehensive Guide_](https://towardsdatascience.com/text-embeddings-comprehensive-guide-afd97fce8fb5/)<br>
— Introductory guide to text embeddings for natural language processing ( _Towards Data Science_ website ).

1. [_Word Embeddings—Explained!_](https://towardsdatascience.com/word-embeddings-explained-c07c5ea44d64/)<br>
— Introductory guide to word embeddings for natural language processing ( _Towards Data Science_ website ).

1. [_Embeddings from Language Model ( ELMo )_](https://en.wikipedia.org/wiki/ELMo)<br>
— Wikipedia article.

1. [_Bidirectional encoder representations from transformers ( BERT )_](https://en.wikipedia.org/wiki/BERT_(language_model))<br>
— Wikipedia article. 



#### <ins>word2vec Model</ins>

1. [_Word2Vec: NLP with Contextual Understanding._](https://mlarchive.com/natural-language-processing/word2vec-nlp-with-contextual-understanding/)<br>
— Theoretical guide for word2vec and GloVe models ( _ML Archive_ website ).

1. [_Word2vec model_](https://en.wikipedia.org/wiki/Word2vec)<br>
— Wikipedia article.

1. [_CBOW – Word2Vec_](https://medium.com/@anmoltalwar/cbow-word2vec-854a043ee8f3)<br>
— Introduction to the continous bag of words (CBOW) and word2vec models ( _Medium_ website ).

1. [_Efficient Estimation of Word Representations in Vector Space_](https://arxiv.org/abs/1301.3781v3)<br>
— Original academic paper ( arxiv.org ).



#### <ins>GloVe Model</ins>

1. [_GloVe model_](https://en.wikipedia.org/wiki/GloVe)<br>
— Wikipedia article.

1. [_GloVe: Global Vectors for Word Representation_](https://nlp.stanford.edu/pubs/glove.pdf)<br>
— Original manusript/academic paper ( Standford NLP Group ).



#### <ins>Sentiment Analysis</ins>

1. [_Detection of emotion by text analysis using machine learning_](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1190326/full)<br>
— Article published in _Frontiers of Psychology_.


---


### Sample Works

1. [_Sarcasm Detection with GloVe/Word2Vec_](https://www.kaggle.com/code/madz2000/sarcasm-detection-with-glove-word2vec-83-accuracy)<br>
— Project on Kaggle applying the word2vec and GloVe models to classifying news headlines from _The Onion_ and the _The Huffington Post_.


---


### Documentation and Tutorials


#### <ins>Neural Networks</ins>

1. [_Your First Deep Learning Project in Python with Keras Step-by-Step_](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)<br>
— Guide to building and training a neural network in Python with Keras ( _Machine Learning Mastery_ website ).

1. [_Training a Neural Network using Keras API in Tensorflow_](https://www.geeksforgeeks.org/training-a-neural-network-using-keras-api-in-tensorflow/)<br>
— Guide to using Keras for neural network training ( _GeeksforGeeks_ website ).

1. [_Python AI: How to Build a Neural Network & Make Predictions_](https://realpython.com/python-ai-neural-network/)<br>
— Tutorial on building your own neural network in Python from scratch ( _Real Python_ website ).



#### <ins>Gensim</ins>

1. [_Gensim Word2Vec Tutorial_](https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial)<br>
— Notebook posted on Kaggle by one of the developers of Gensim ( _Kaggle_ website ).

1. [_Word2vec embeddings_](https://radimrehurek.com/gensim/models/word2vec.html)<br>
— Word2vec module documation ( _Gensim_ website ).

1. [_Word2Vec Model_](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)<br>
— Word2vec tutorial ( _Gensim_ website ).


---


### More Theoretical Background

1. [_Machine Learning Tutorial_](https://www.geeksforgeeks.org/machine-learning/)<br>
— General overview/tutorial on machine learning ( _GeeksforGeeks_ website ).

1. [_AI ML DS - How To Get Started?_](https://www.geeksforgeeks.org/ai-ml-ds/)<br>
— General overview on artificial intelligence, machine learning, and data science ( _GeeksforGeeks_ website ).

1. [_Bag of words model_](https://en.wikipedia.org/wiki/Bag-of-words_model)<br>
— Wikipedia article.

1. [_Logistic regression_](https://en.wikipedia.org/wiki/Logistic_regression)<br>
— Wikipedia article.

1. [_Multinomial logistic regression_](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)<br>
— Wikipedia article.

1. [_Least squares_](https://en.wikipedia.org/wiki/Least_squares)<br>
— Wikipedia article.

1. [_Tf-idf ( term frequency-inverse document frequency )_](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)<br>
— Wikipedia article.

1. [_Feedforward neural network_](https://en.wikipedia.org/wiki/Feedforward_neural_network)<br>
— Wikipedia article.

1. [_Convolutional neural network_](https://en.wikipedia.org/wiki/Convolutional_neural_network)<br>
— Wikipedia article.

1. [_Recurrent neural network_](https://en.wikipedia.org/wiki/Recurrent_neural_network)<br>
— Wikipedia article.


---


### Mathematical References

1. [_Dot product_](https://en.wikipedia.org/wiki/Dot_product)<br>
— Wikipedia article.

1. [_Cosine similarity_](https://en.wikipedia.org/wiki/Cosine_similarity)<br>
— Wikipedia article.

1. [_Linear least squares_](https://en.wikipedia.org/wiki/Linear_least_squares)<br>
— Wikipedia article.

1. [_Fréchet mean_](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean)<br>
— Wikipedia article.


---


### Graphical Visualization Guides
1. [_The Art of Effective Visualization of Multi-dimensional Data_](https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57/)<br>
— Guide on plotting multidimensional data ( _Towards Data Science_ website ).

1. [_Top Python Data Visualization Libraries in 2024: A Complete Guide_](https://www.fusioncharts.com/blog/best-python-data-visualization-libraries/)<br>
— ( _Fusion Charts_ website ).

1. [_A Complete Beginner’s Guide to Data Visualization_](https://www.analyticsvidhya.com/blog/2021/04/a-complete-beginners-guide-to-data-visualization/)<br>
— ( _Analytics Vidhya_ website ).

1. [_Tableau for Beginners: Data Visualisation Made Easy_](https://www.analyticsvidhya.com/blog/2017/07/data-visualisation-made-easy/)<br>
— ( _Analytics Vidhya_ website ).

1. [_Intermediate Tableau guide for data science and business intelligence professionals_](https://www.analyticsvidhya.com/blog/2018/01/tableau-for-intermediate-data-science/)<br>
— ( _Analytics Vidhya_ website ).



## Future Direction and Possible Improvements
1. Machine learning
    1. Extend project to sentiment and tone classificaiton of text.

    1. Extend project to multimodal classification where multiple input modalities (images, video, audio, etc.) are used together for prediction/classification.

1. Browser application
    1. Develop an app that can be used in a web browser that allows the user to directly take a comment and its associated data straight from the Reddit website (or from a screenshot).

    1. Develop the app further to display in real time the predicted tone of all of the comments seen in the current browser window.

    1. Extend project to multimodal classification where multiple input modalities (images, video, audio, etc.) are used together for prediction/classification.





</div>

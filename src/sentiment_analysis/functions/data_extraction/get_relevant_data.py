import extract_data
import csv
import pandas as pd
import nltk
import string


def process_data(
        input_csv:str,
        output_csv:str
    ):
    """_summary_
    Processes the input CSV file by removing punctuation and stopwords from the text columns 
    and saves the cleaned data to an output CSV file.

    Args:
        1. input_csv (str): Path to the input CSV file.
        2. output_csv (str): Path to the output CSV file where the processed data will be saved.

    The Function:
        1. Reads the input CSV file into a pandas DataFrame.
        2. Removes punctuation and stopwords from the 'comment' and 'parent_comment' columns.
        3. Saves the cleaned DataFrame to the output CSV file.

    Returns:
        1. n/a
    """
    # Extract data.
    data = extract_data.extract_data(csv_file_name=input_csv)

    # Write data to a temporary CSV file.
    temp_file = 'training_data.csv'
    data.to_csv(temp_file, index=False)

    # Open the temporary CSV file and extract data into a DataFrame using pandas.
    data = pd.read_csv(temp_file, encoding="utf-8")

    # Define words and punctuations to remove from the text in the data.
    stopwords_list = set(nltk.corpus.stopwords.words('english'))
    punctuation = list(string.punctuation)
    stopwords_list.update(punctuation)

    # Remove punctuation from individual words in the 'comment' and 'parent_comment' columns.
    data['comment'] = data['comment'].apply(lambda x: ''.join([char for char in x if char not in punctuation]))
    # data['parent_comment'] = data['parent_comment'].apply(lambda x: ''.join([char for char in x if char not in punctuation]))

    # Remove stopwords from the text in 'comment' and 'parent_comment' columns.
    data['comment'] = data['comment'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stopwords_list]))
    # data['parent_comment'] = data['parent_comment'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stopwords_list]))

    # Write the cleaned data to the output CSV file (only 'label' and 'comment').
    data[['label', 'comment']].to_csv(output_csv, index=False)


# Sample usage of the process_data function.
if __name__ == "__main__":
    input_csv = 'train-balanced-sarcasm.csv'
    output_csv = 'trimmed_training_data.csv'
    process_data(input_csv, output_csv)
    print(f"Processed data saved to {output_csv}")
import pandas as pd



def extract_data(csv_file_name):
    """
    Extracts the relevant data from the CSV file and returns a DataFrame.

    Args:
        1. csv_file_name (str): The name of the CSV file to extract data from.
    
    The Function:
        1. Loads the CSV file into a DataFrame.
        2. Extracts the relevant columns: "comment", "subreddit", "parent_comment", and "label".
        3. Checks for missing values in the DataFrame.
        4. Drops rows with missing values.
        5. Returns the cleaned DataFrame.

    Returns:
        1. data_redux: A DataFrame containing the extracted data.
    """

    # Load the CSV file into a DataFrame.
    data_all_columns = pd.read_csv(csv_file_name, encoding="utf-8")

    # Extract the columns we need.
    data_redux = data_all_columns[
        [
            "label",
            "comment",
            "subreddit",
            "parent_comment",
        ]
    ]

    # Find the number of rows with missing values.
    num_missing_rows = data_redux.isna().sum()

    # Drop rows with missing values.
    data_redux = data_redux.dropna()

    return data_redux

# Example usage:
if __name__ == "__main__":
    csv_file_name = "data/sentences.csv"
    data_redux = extract_data(csv_file_name)
    print(data_redux.head())
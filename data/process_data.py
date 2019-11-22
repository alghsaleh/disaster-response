import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    """
    """

    # Load csv files into DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge DataFrames
    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    """
    """

    # Create dataframe with column for each individual category
    categories = df.categories.str.split(';', expand=True)

    # Use first row to extract and rename column names
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])

    # Set and convert last character as a numeric value
    for column in categories:
        categories[column] = categories[column].str[-1].astype(np.int)

    # Drop 'child_alone' column because it only contains zeros
    categories.drop(columns='child_alone', inplace=True)

    # Drop original `categories` column from `df`
    df.drop(columns='categories', inplace=True)

    # Concat `df` with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows where `related` = 2 indicating non-english language
    df = df[df['related'] != 2]

    return df

def save_data(df, database_filename):
    """
    """

    # Create engine for SQLite database
    engine = create_engine(os.path.join('sqlite:///', database_filename))

    # Export DataFrame as database table
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

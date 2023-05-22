import sys
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories csv files and merges them on common id.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Returns:
    df: merged DataFrame of messages and categories
    
    """

    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df = messages.merge(categories, how ='outer', on =['id'])
    return df

def clean_data(df):
    """
    splits categories into separate category columns
    converts column from string to binary classes
    concatenate the original dataframe with the new `categories` dataframe

    Parameters:
    df: DataFrame of messages and categories 
    
    Returns:
    df: cleaned DataFrame of messages and categories
    
    """
    
    categories = df['categories'].str.split(';', expand=True) # split categories into separate category columns

    row = categories.iloc[0] # select the first row of the categories dataframe for column names

    category_colnames = row.apply(lambda x: x[:-2]) # extract a list of new column names for categories.

    categories.columns = category_colnames.to_list() # rename the columns of `categories`

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    categories.related.replace(2, 1, inplace=True) # replace 2 with 1 in related column

    df.drop('categories', axis=1, inplace=True) # drop the original categories column from `df`

    # concatenate the original dataframe with the new `categories` dataframe

    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True) # drop duplicates

    return df
    
def save_data(df, database_filepath):
    """
    Stores df in a SQLite database.
    
    Parameters:
    df: DataFrame
    database_filepath: SQLite database file
    
    Returns: There is no return value
    
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    """Loads data, cleans data, saves data to database"""
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
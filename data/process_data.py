# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads and merges 'messages' and 'categories' data files as dataframe
    
    Parameters:
        'messages_filepath': file path to csv file
        'categories_filepath': file path to csv file
        
    Returns:
        'df': pandas dataframe with merged data files
    """
    # load messages dataset
    messages = pd.read_csv('disaster_messages.csv')
    
    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """ Cleans dataframe with merged data files:
            - splits 'categories' column into 36 separate category columns
            - renames the 36 category columns with correct category names
            - keeps only column values in the 36 category columns and converts 
            columns to numeric
            - removes rows with value '2' in column 'related' from dataframe
            - removes duplicate rows from dataframe
    
    Parameters:
        'df': pandas dataframe with merged data files
        
    Returns:
        'df': cleaned pandas dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use row to extract a list of new column names for categories
    category_colnames = [x[:-2] for x in row]

    # rename the columns of categories
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from df
    df.drop(columns='categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    # drop rows with value 2 in column 'related'
    df.drop(df[df.related == 2].index, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """ Creates a sqlite engine and stores cleaned dataframe to database
    
    Parameters:
        'df': cleaned pandas dataframe
        'database_filename': file path to database
        
    Returns:
        None
    """
    # create sqlite engine and store dataframe to server
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)

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
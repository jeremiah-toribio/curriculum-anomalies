# easy file path reading
import os
# dataframes & arrays
import pandas as pd
import numpy as np
# visualizations
import seaborn as sns
import matplotlib.pyplot as plt
# clear warnings
import warnings
warnings.filterwarnings("ignore")
# request
import requests
# metrics
from sklearn import metrics


### Read and Prepare ###

def get_curriculum_data():
    '''

    Will be used to acquire and prepare codeup curriculum data.

    '''
    # Retrieving logs.csv
    if os.path.isfile('logs.csv'):
        # If file exists read in data
        print('File exists, pulling from system. (logs.csv)')
        logs = pd.read_csv('logs.csv')
    else:
        # Necessary data does not exist and should be retrieved before hand
        print('No file exists, please provide necessary data.')
        return
    
    # Retrieving data.txt
    if os.path.isfile('data.txt'):
        # If file exists read in data
        print('File exists, pulling from system. (data.txt)')
        df = pd.read_csv('data.txt', sep=" ", header=None)
    else:
        # Necessary data does not exist and should be retrieved before hand
        print('No file exists, please provide necessary data.')
        return

    # renaming columns to their actual names
    df = df.rename(columns={0:'date', 1:'time', 2:'path',3:'user_id',4:'cohort_id', 5:'ip'})

    # changing dtypes
        # cohort_id has a large number of missing values so this was the work around
    df['cohort_id'] = df['cohort_id'].fillna(-1)
    df['cohort_id'] = df['cohort_id'].astype('int64')
    df['cohort_id'] = df['cohort_id'].replace(np.nan, -1)

    # adding program_id
    df = df.merge(logs, left_on='cohort_id',right_on='id',how='left')
    
    # I wasted time doing this, leaving for aesthetics
            # remapping cohort id (tbd)
                # cohorts = logs[['id','name']]
                #     # respectively each name is associated with each id
                # cohort_names = cohorts['name'].unique().tolist()
                #     # removing -1 as it is not a match to a cohort name
                # ids = df.cohort_id.unique()[df.cohort_id.unique() > 0].tolist()
                # # appending (logs) cohort name to (df)
                # name_dict = {}
                # for key, val in zip(ids, cohort_names):
                #     name_dict[key] = val
                # df['name'] = df['cohort_id'].map(name_dict)
                #     # changing cohort_id back to nan

    

    #changing program_id to int but keeping nan
    df['program_id'] = df['program_id'].fillna(-1)
    df['program_id'] = df['program_id'].astype(int)
    df['program_id'] = df['program_id'].replace(np.nan, -1)
    df['program_id'] = df['program_id'].map({1:'Web Dev 1',2:'Fullstack',3:'Data Science',4:'Front End WD'})
    #mapping program id by respective program

    # giving staff it's own program id
    df.loc[(df['name'] == 'Staff') & (df['program_id'] == 'Fullstack'), 'program_id'] = 'Staff'
    
    # getting our datetime index
    df['datetime'] = df.date + df.time
    df['datetime'] = pd.to_datetime(df.datetime, format='%Y-%m-%d%H:%M:%S')
    # setting index
    df = df.set_index(df.datetime)

    # dropping extra columns
    df = df.drop(columns=['date','time','datetime',
                          'Unnamed: 0','slack','deleted_at',
                          'id'])

    
    return df


### Summarize & Describe ###
def summarize(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.

    returns:
        pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].dtype
            ]
        )

        check_columns = pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "dtype"],
    ).sort_values(by="Number of Unique Values")
   
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return check_columns

### All utilized in the summarize_df function
def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df) -> pd.DataFrame:
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')

def get_fences(df, col, k=1.5) -> tuple:
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound
def summarize_df(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('=======================\n=====   SHAPE   =====\n=======================')
    print(df.shape)
    print('========================\n=====   INFO   =====\n========================')
    print(df.info())
    print('========================\n=====   DESCRIBE   =====\n========================')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('==========================\n=====   DATA TYPES   =====\n==========================')

    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('O').describe().T.to_markdown())
    print('==========================\n=====   BY COLUMNS   =====\n==========================')
    print(missing_by_col(df).to_markdown())
    print('=======================\n=====   BY ROWS   =====\n=======================')
    print(missing_by_row(df).to_markdown())
    print('========================\n=====   OUTLIERS   =====\n========================')
    print(report_outliers(df, k=k))
    print('================================\n=====   THAT IS ALL, BYE   =====\n================================')


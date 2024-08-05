import numpy as np
import pandas as pd


def load_data(path, cluster):
    """Load data from a directory and cluster.

    Parameters
    ----------
    directory : str
        The directory where the data is stored.
    cluster : int
        The cluster to load.

    Returns
    -------
    data : pd.DataFrame
        The data for the given cluster.
    """
    data = pd.read_csv(f'{path}/{cluster}.csv')
    return data


AVA_0_df = pd.read_csv ('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_0_df.csv')
AVA_1_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_1_df.csv') 
AVA_2_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_2_df.csv')
AVA_3_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_3_df.csv') 
AVA_4_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_4_df.csv') 
AVA_5_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_5_df.csv') 
AVA_6_df = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/Clustering/clustered_bacterial_stim/AVA_6_df.csv')
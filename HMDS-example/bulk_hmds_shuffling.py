'This file takes a list of files and runs the HMDS bulk analysis on them'


import os 
import numpy as np
import pandas as pd
import cmdstanpy as stan
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

#stan model retrieval
path = '/Users/iuliarusu/Documents/Sharpee/HMDS-example/model/'
ltz_m = stan.CmdStanModel(stan_file=path+'lorentz.stan')


"pre-processing functions"
def distance_matrix(df):
    corr_matrix = df.corr()
    # distance_matrix = 1 - corr_matrix
    distance_matrix_squaredp = (1 - corr_matrix**2) * 2
    return distance_matrix_squaredp





"running model functions"
def run_ltz_m(distance_matrix_squaredp):
#returns poincare coordinates of N points in D dimensions
    dat = {'N': distance_matrix_squaredp.shape[0] , 'D': 3 , 'deltaij':distance_matrix_squaredp}
#run optimizer
    model = ltz_m.optimize(data=dat, iter=250000, algorithm='LBFGS', tol_rel_grad=1e2)
    return model

def build_result(model):
    hyp_emb = {'euc':model.euc, 'sig':model.sig, 'lambda':model.stan_variable('lambda')}
    return hyp_emb

"post-processing functions"

def d_lor(t1, t2, E1, E2):
    return np.arccosh(t1*t2 - np.dot(E1, E2))

#returns embedding distance matrix from optimization fit
def get_embed_dmat(fit):
    N = fit['euc'].shape[0]
    fit_ts = np.sqrt(1.0 + np.sum(np.square(fit['euc']), axis=1))

    fit_mat = np.zeros((N, N))

    for i in np.arange(N):
        for j in np.arange(i+1,N):
            fit_mat[i][j] = d_lor(fit_ts[i], fit_ts[j], fit['euc'][i], fit['euc'][j])
            fit_mat[j][i] = fit_mat[i][j]
            
    return fit_mat

#return poincare coordinates
def get_poin(fit):
    ts = np.sqrt(1.0 + np.sum(np.square(fit['euc']), axis=1))
    return (fit['euc'].T / (ts + 1)).T


def process_sim(fit):
    fit['emb_mat'] = get_embed_dmat(fit)/fit['lambda']
    fit['pcoords'] = get_poin(fit)
    fit['radii'] = 2.0*np.arctanh(np.sqrt(np.sum(np.square(fit['pcoords']), axis=1)))
    return fit

#fit model across dange of dimensions

def fit_many_dims(distance_matrix_squaredp, dims = 10):
    all_fits = []
    for d in np.arange(2, dims):
        dat={'N': distance_matrix_squaredp.shape[0], 'D':d, 'deltaij':distance_matrix_squaredp}
        #run optimizer
        model = ltz_m.optimize(data=dat, iter=250000, algorithm='LBFGS', tol_rel_grad=1e2)
        all_fits.append({'euc':model.euc, 'sig':model.sig, 'lambda':model.stan_variable('lambda'), 't':model.time})
    return all_fits



#return negative log likelihood of fit
def MDS_lkl(fit, dmat):
    lkl = 0;
    N = fit['sig'].shape[0]
    
    sigs = fit['sig']
    lam = fit['lambda']
    emb_mat = get_embed_dmat(fit)
    
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            seff = sigs[i]**2 + sigs[j]**2
            lkl += ((dmat[i][j] - emb_mat[i][j]/lam)**2 / (2.0*seff)) + 0.5*np.log(seff*2.0*np.pi)
    return lkl

#return bic of model
#input: optimization fit and distance matrix
def BIC(fit, dmat):
    N,D = fit['euc'].shape
    n = 0.5*N*(N-1)
    k = N*D + N + 1.0 - 0.5*D*(D-1)
    
    return k*np.log(n) + 2.0*MDS_lkl(fit, dmat)


def run_HMDS(df):
    distance_matrix_squaredp = distance_matrix(df)
    cell_counts = distance_matrix_squaredp.shape[0]

    all_fits = fit_many_dims(distance_matrix_squaredp)

    array_matrix = distance_matrix_squaredp.to_numpy()

    all_BIC = [BIC(fit, array_matrix) for fit in all_fits]
    #return index of minimum BIC

    min_BIC = (np.argmin(all_BIC)) + 2

    #extract lambda
    optimal_lambda = min_BIC - 2
    opt_lambda = all_fits[optimal_lambda]['lambda']

    #radii
    iteration = all_fits[min_BIC - 2]['t']
    radii = np.arccosh(iteration)
    
    # model = run_ltz_m(distance_matrix_squaredp)

    # hyp_emb = build_result(model)

    # fit = process_sim(hyp_emb)

    return min_BIC, cell_counts, opt_lambda, radii



#point to directory with files

directory = '/Users/iuliarusu/Documents/Sharpee/ProcAiryData/ProcAiryData/Yfull_op50_SF.npz'

#load data
# # Load the data
# Y_full = np.load('/Users/iuliarusu/Documents/Sharpee/ProcAiryData/Yfull_op50_SF.npz')
# w_0, w_1, w_2, w_3, w_4, w_5, w_6 = Y_full['arr_0'], Y_full['arr_1'], Y_full['arr_2'], Y_full['arr_3'], Y_full['arr_4'], Y_full['arr_5'], Y_full['arr_6']
# all_data = [w_0, w_1, w_2, w_3, w_4, w_5, w_6]





#Filtered UNclustered Data

filtered_unclustered = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/filtered_unclustred.csv')
all_data = [
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 0 ].iloc[:, : -1].T, 
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 1 ].iloc[:, : -1].T,
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 2 ].iloc[:, : -1].T,
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 3 ].iloc[:, : -1].T,
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 4 ].iloc[:, : -1].T,
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 5 ].iloc[:, : -1].T,
    filtered_unclustered[filtered_unclustered[ 'worm_id'] == 6 ].iloc[:, : -1].T
]

#filtered Clustered Data

# all_ws_clustered = pd.read_csv('/Users/iuliarusu/Documents/Sharpee/all_ws_clustered.csv')
# all_data = [
#     all_ws_clustered[all_ws_clustered['worm_id']==0].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==1].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==2].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==3].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==4].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==5].iloc[:, : -3].T,
#     all_ws_clustered[all_ws_clustered['worm_id']==6].iloc[:, : -3].T
# ]


# Settings
shuffling = False
batch_number = 1
cell_number = 50

# Initialize containers and counters
worm_counter = 0
exp_dataframes = {}
np.random.seed(42)  # Set numpy seed

# Shuffling function for columns
def shuffling_function(df, batch_num, cell_number):
    return df.sample(n=cell_number, axis=1, random_state=batch_num)


    
 # Main processing loop
for worm_data in all_data:

    # if worm_counter >= 2:
    #     break
    worm_counter += 1
    worm_df = pd.DataFrame(worm_data)  # Convert worm data to DataFrame
    print(f"worm_df shape: {worm_df.shape}")
    # Initialize a list to store batch results for each worm
    batch_results = []
    
    if shuffling:  # Run with shuffling enabled
        for batch in range(batch_number):
            # Shuffle and process data for each batch
            shuffled_worm_df = shuffling_function(worm_df, batch, cell_number)
            
            # Run the HMDS model and get all relevant results
            result, cell_counts, opt_lambda, radii = run_HMDS(shuffled_worm_df)
            
            # Compute radii statistics
            min_radii = np.min(radii)
            max_radii = np.max(radii)
            mean_radii = np.mean(radii)
            
            # Append results to batch results
            batch_results.append((
                f"Worm_{worm_counter}_Batch_{batch}",  # Worm ID + Batch
                result,  # BIC
                cell_counts,  # Cell Counts
                opt_lambda,  # Lambda
                min_radii,  # Min Radii
                max_radii,  # Max Radii
                mean_radii,  # Mean Radii
                ','.join(map(str, radii))  # All Radii as a comma-separated string
            ))
            
            print(f"Processed Worm {worm_counter}, Batch {batch}, Result: {result}")
            print(f"Lambda: {opt_lambda}, Radii: Min={min_radii}, Max={max_radii}, Mean={mean_radii}")
    else:  # Run without shuffling
        # Run the HMDS model and get all relevant results
        result, cell_counts, opt_lambda, radii = run_HMDS(worm_df)
        
        # Compute radii statistics
        min_radii = np.min(radii)
        max_radii = np.max(radii)
        mean_radii = np.mean(radii)
        
        # Append results to batch results (single batch with no shuffling)
        batch_results.append((
            f"Worm_{worm_counter}_NoShuffle",  # Worm ID without Batch
            result,  # BIC
            cell_counts,  # Cell Counts
            opt_lambda,  # Lambda
            min_radii,  # Min Radii
            max_radii,  # Max Radii
            mean_radii,  # Mean Radii
            ','.join(map(str, radii))  # All Radii as a comma-separated string
        ))
        
        print(f"Processed Worm {worm_counter}, No Shuffling, Result: {result}")
        print(f"Lambda: {opt_lambda}, Radii: Min={min_radii}, Max={max_radii}, Mean={mean_radii}")
    
    # Store results in the exp_dataframes dictionary for each worm
    exp_dataframes[f"Worm_{worm_counter}"] = batch_results

# Prepare summary data for export
data_for_export = [
    (
        batch_id,  # Worm ID + Batch
        bic_result,  # BIC
        cell_counts,  # Cell Counts
        opt_lambda,  # Lambda
        min_radii,  # Min Radii
        max_radii,  # Max Radii
        mean_radii,  # Mean Radii
        all_radii  # All Radii as a string
    )
    for batch_results in exp_dataframes.values()
    for batch_id, bic_result, cell_counts, opt_lambda, min_radii, max_radii, mean_radii, all_radii in batch_results
]

# Convert summary data to DataFrame and export as CSV
summary_df = pd.DataFrame(data_for_export, columns=['Worm ID', 'BIC', 'Cell Counts', 'Optimal Lambda', 'Min Radii', 'Max Radii', 'Mean Radii', 'All Radii'])
csv_file_path = '/Users/iuliarusu/Documents/Sharpee/Buffer_control_data/filtered_unclustered_full.csv'
summary_df.to_csv(csv_file_path, index=False)
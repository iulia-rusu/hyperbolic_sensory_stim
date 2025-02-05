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

# # Shuffling function to add columns up to a target count
# def shuffling_function(df, batch_num, target_column_count=130):
#     current_column_count = df.shape[1]
#     if current_column_count < target_column_count:
#         additional_columns_needed = target_column_count - current_column_count
#         columns_to_add = df.sample(n=additional_columns_needed, axis=1, replace=True, random_state=batch_num)
#         df = pd.concat([df, columns_to_add], axis=1)
#     return df

# Shuffling function to add columns up to a target count
def shuffling_function(df, batch_num, target_row_count=130):
    current_row_count = df.shape[0]
    if current_row_count < target_row_count:
        additional_rows_needed = target_row_count - current_row_count
        rows_to_add = df.sample(n=additional_rows_needed, axis=0, replace=True, random_state=batch_num)
        # Generate bare minimum Gaussian noise and only positive values
        noise = np.random.normal(0, 1, rows_to_add.shape)

        
        # Add the noise to the new columns
        rows_to_add = rows_to_add + noise
        df = pd.concat([df, rows_to_add], axis=0)
    return df

#point to directory with files

directory = '/Users/iuliarusu/Documents/Sharpee/Buffer_control_data/'
buffer_dataframes = {}
scaler = MinMaxScaler()
batch_number = 1
file_counter = 0  # Initialize loop cycle counter

for filename in os.listdir(directory):

    if filename.endswith('.csv'):
        file_counter += 1  # Track each cycle of the loop
        file_path = os.path.join(directory, filename)
        key = filename.split('.')[0]

        # Read and scale the CSV file
        df = pd.read_csv(file_path)
        print(f"File shape: {df.shape}")
        df = df.T
        
        print(f"Processing file: {filename}")
        print(f"Dataframe dimensions: {df.shape}")

       

        # Process each batch for the current file
        for batch in range(batch_number):
            # Apply shuffling to add columns up to 130 with batch-specific randomization
            df_shuffled = shuffling_function(df, batch_num=batch, target_row_count=130)
            print(f"Shuffled Data Shape: {df_shuffled.shape}")
            
             #add more rows 
            df_scaled = df.apply(lambda row: scaler.fit_transform(row.values.reshape(-1, 1)).flatten(), axis=1)
            # Convert back to a DataFrame
            df_scaled = pd.DataFrame(df_scaled.tolist(), index=df.index, columns=df.columns)
            print(f"Scaled dataframe shape:{df_shuffled.shape}")

            df_scaled = df_scaled.T
            print(f"Shape of Df, time x cells going into HMDS:{df_scaled.shape}")
            
            # Run the HMDS model on the processed data
            result, cell_counts, opt_lambda, radii = run_HMDS(df_scaled)
            print(f"Result: {result}, Cell Counts: {cell_counts}")
            # Store results with batch-specific keys
            buffer_dataframes[f"{key}_batch_{batch}_result"] = result
            buffer_dataframes[f"{key}_batch_{batch}_cell_counts"] = cell_counts
            buffer_dataframes[f"{key}_batch_{batch}_opt_lambda"] = opt_lambda
            buffer_dataframes[f"{key}_batch_{batch}_radii"] = radii

            # Print processing info
            print(f"Processed: {key} | Batch: {batch} | Loop cycle: {file_counter}")
            print(f"Minimum BIC: {result}")
            print(f"Matrix Shape: {cell_counts}")
            print(f"Lambda: {opt_lambda}")
            print(f"Radii: {np.mean(radii)}")




data_for_export = [
    (
        key.replace('_result', ''),  # Filename
        buffer_dataframes[key],  # BIC
        buffer_dataframes[key.replace('_result', '_cell_counts')],  # Cell Counts
        buffer_dataframes[key.replace('_result', '_opt_lambda')],  # Optimal Lambda
        np.min(buffer_dataframes[key.replace('_result', '_radii')]),  # Min Radii
        np.max(buffer_dataframes[key.replace('_result', '_radii')]),  # Max Radii
        np.mean(buffer_dataframes[key.replace('_result', '_radii')]),  # Mean Radii
        ','.join(map(str, buffer_dataframes[key.replace('_result', '_radii')]))  # All Radii
    )
    for key in buffer_dataframes.keys() if '_result' in key
]

summary_df = pd.DataFrame(data_for_export, columns=['Filename', 'BIC', 'Cell Counts', 'Optimal Lambda', 'Min Radii', 'Max Radii', 'Mean Radii', 'All Radii'])
csv_file_path = '/Users/iuliarusu/Documents/Sharpee/Buffer_control_data/buff_extra_cell_summary_fixed_radii.csv'
summary_df.to_csv(csv_file_path, index=False)
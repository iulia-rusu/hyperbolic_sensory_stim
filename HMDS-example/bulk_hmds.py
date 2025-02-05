'This file takes a list of files and runs the HMDS bulk analysis on them'


import os 
import numpy as np
import pandas as pd
import cmdstanpy as stan
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

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


    
    # model = run_ltz_m(distance_matrix_squaredp)

    # hyp_emb = build_result(model)

    # fit = process_sim(hyp_emb)

    return min_BIC, cell_counts



#point to directory with files

directory = '/Users/iuliarusu/Documents/Sharpee/Buffer_control_data/'

#master dictionary with key as file name of dictionary and 
buffer_dataframes = {}

file_counter = 0
#scaler for nomalizing processed csv dataframe
scaler = StandardScaler()

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        
        file_counter += 1 

        # Full path to the file
        file_path = os.path.join(directory, filename)
        
        # Extract a name to use as a key
        key = filename.split('.')[0]  # This takes the filename without the extension

        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Scale each column
        for column in df.columns:
            df[column] = scaler.fit_transform(df[[column]])
        
        # Store the scaled DataFrame in the dictionary
        buffer_dataframes[key] = df

        # Apply the processing function
        result, cell_counts = run_HMDS(df)

        # Store the result in the dictionary
        buffer_dataframes[key + '_result'] = result

        #store cell counts 
        buffer_dataframes[key + '_cell_counts'] = cell_counts

        print(f"Processed: {key}")
        print(f"Minimum BIC: {buffer_dataframes[key + '_result']}")
        print(f"Matrix Shape: {buffer_dataframes[key + '_cell_counts']}")


data_for_export = []


#export the summary data as a csv file
for key in buffer_dataframes.keys():
    if '_result' in key:

        base_key = key.replace('_result', '')

        bic_result = buffer_dataframes[key]
        cell_counts = buffer_dataframes[base_key + '_cell_counts']

        data_for_export.append((base_key, bic_result, cell_counts))

summary_df = pd.DataFrame(data_for_export, columns=['Filename', 'BIC', 'Cell Counts'])

csv_file_path = '/Users/iuliarusu/Documents/Sharpee/Buffer_control_data/unfiltered_unclustered_summary.csv'
summary_df.to_csv(csv_file_path, index=False)
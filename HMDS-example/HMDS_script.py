"Collection of computational and ploting functions for HMDS"

import numpy as np
import diptest  
import matplotlib.pyplot as plt
import cmdstanpy as stan
import scipy.stats as stats
import pickle
import seaborn as sns
%matplotlib inline

import pandas as pd

import plotly.express as px
from matplotlib import rcParams


#returns hyperbolic distance between vectors in poincare ball


path = '/Users/iuliarusu/Documents/Sharpee/HMDS-example/model/'
ltz_m = stan.CmdStanModel(stan_file=path+'lorentz.stan')



#data matrix processing, input 'arr_0' through 'arr_6'
def import_data_matrix(array):
    x = str(array)
    data_file = np.load('/Users/iuliarusu/Documents/Sharpee/ProcAiryData/Yfull_op50_SF.npz')
    arr = data_file[x]
    corr_matrix = (pd.DataFrame(arr)).corr()
    distance_matrix = 1 - corr_matrix
    distance_matrix_squaredp = (1 - corr_matrix**2) * 2
    return distance_matrix_squaredp




class PoincareEmbedding(self, ):

def poincare_dist(v1, v2):
    sq = np.sum(np.square(v1-v2))
    r1 = np.sum(np.square(v1))
    r2 = np.sum(np.square(v2))
    inv = 2.0*sq/((1.0-r1)*(1.0-r2))
    return np.arccosh(1.0 + inv)

#return NxN symmetric distance matrix from poincare coordinates
def get_dmat(p_coords):
    N = p_coords.shape[0]
    dists = np.zeros((N, N))
    
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            dists[i][j] = poincare_dist(p_coords[i], p_coords[j])
            dists[j][i] = dists[i][j]
    return dists

def run_optimizer(data):
    # dat={'N':100, 'D':5, 'deltaij':mat_dim}
    dat = {'N': 135 , 'D': 3 , 'deltaij':distance_matrix_squaredp}
#run optimizer
    model = ltz_m.optimize(data=dat, iter=250000, algorithm='LBFGS', tol_rel_grad=1e2)
    hyp_emb = {'euc':model.euc, 'sig':model.sig, 'lambda':model.stan_variable('lambda')}
    return hyp_emb






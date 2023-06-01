"""
Draft script to train a simple sklearn model 
and output the results that could be used to create 
a table similar to what is from the paper.

"""

import numpy as np
import pandas as pd
from pprint import pprint

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from astartes import train_val_test_split


# read in the data
CSV_PATH = '../barrier_prediction_with_RDB7/ccsdtf12_dz.csv'
df = pd.read_csv(CSV_PATH)
df

# helper function to featurize the data with 2048 morgan fingerprint
# https://github.com/chemprop/chemprop/blob/master/chemprop/features/features_generators.py
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048
def morgan_counts_features_generator(mol,
                                     radius= MORGAN_RADIUS,
                                     num_bits= MORGAN_NUM_BITS):
    """
    Generates a counts-based Morgan fingerprint for a molecule.
    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features

# create X and y
params = Chem.SmilesParserParams()
params.removeHs = False

X = np.zeros((len(df), 2048*2))
for i, row in df.iterrows():
    rsmi, psmi = row.rsmi, row.psmi
    
    rmol = Chem.MolFromSmiles(rsmi, params)
    r_morgan = morgan_counts_features_generator(rmol)
    
    pmol = Chem.MolFromSmiles(psmi, params)
    p_morgan = morgan_counts_features_generator(pmol)
    
    X[i, :] = np.concatenate((r_morgan,
                              p_morgan - r_morgan),
                             axis=0)

y = df.dE0.values

def produce_table(sklearn_model, X, y, samplers = ["random"], seed=0, hopts={},):
    final_dict = {}
    for sampler in samplers:
        error_dict = {'mae': {'train': [],
                              'val': [],
                              'test': [],
                             },
                      'rmse': {'train': [],
                               'val': [],
                               'test': [],
                              },
                      'R2': {'train': [],
                               'val': [],
                               'test': [],
                            },
                     }
        
        # obtain indices
        _,_,_, train_indices, val_indices, test_indices = train_val_test_split(X,
                                                                        train_size=0.85,
                                                                        val_size=0.05,
                                                                        test_size=0.1,
                                                                        sampler=sampler,
                                                                        random_state=seed,
                                                                        hopts=hopts,
                                                                        return_indices=True,
                                                                        ) 
        
        
        # create data splits
        X_train = X[train_indices]
        X_val = X[val_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]
        
        
        # fit the model to the training data
        sklearn_model.fit(X_train, y_train)
        
        # get predictions
        y_pred_train = sklearn_model.predict(X_train)
        y_pred_val = sklearn_model.predict(X_val)
        y_pred_test = sklearn_model.predict(X_test)
        
        
        # store MAEs
        train_mae = mean_absolute_error(y_train, y_pred_train)
        error_dict['mae']['train'].append(train_mae)

        val_mae = mean_absolute_error(y_val, y_pred_val)
        error_dict['mae']['val'].append(val_mae)

        test_mae = mean_absolute_error(y_test, y_pred_test)
        error_dict['mae']['test'].append(test_mae)
        
        
        # store RMSEs
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        error_dict['rmse']['train'].append(train_rmse)

        val_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        error_dict['rmse']['val'].append(val_rmse)

        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        error_dict['rmse']['test'].append(test_rmse)

        
        # store R2
        train_R2 = r2_score(y_train, y_pred_train)
        error_dict['R2']['train'].append(train_R2)

        val_R2 = r2_score(y_val, y_pred_val)
        error_dict['R2']['val'].append(val_R2)

        test_R2 = r2_score(y_test, y_pred_test)
        error_dict['R2']['test'].append(test_R2)
        
        final_dict[sampler] = error_dict
        
    return final_dict


# use default hyperparameters
sklearn_model = LinearSVR()

final_dict = sample_function(sklearn_model, X, y)
pprint(final_dict)


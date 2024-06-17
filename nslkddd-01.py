### This program creates a class that makes the NSL KDD dataset and its associated predictor matrices

# Load packages
from NSL_KDD_Data import NSLKDD
from inputData import InputData
import numpy as np
import pickle 



def main():
    print('\n Creating NSL KDD Dataset and associated design matrices \n\n\n')

    ### Create dataset object
    data = NSLKDD()
    nsl_kdd = data.makeNSLKDD()
	

	### Create input datasets
    inp = InputData(nsl_kdd, ['label', 'classType'])

    X_raw = inp.makeXdata()                   ### Creates dataset of X features
    X_OHE = inp.makeOHE()                     ### Creates dataset of X features with one-hot encoded categorical features
    X_GowMat = inp.gower_mtx()                ### Create Gower matrix


    # Store datasets in shelf file 
    nsl_kdd.to_pickle("./nsl_kdd.pkl")
    X_raw.to_pickle("./nsl_kdd_X_raw.pkl")    ## Stores dataset of X features
    X_OHE.to_pickle("./nsl_kdd_X_OHE.pkl")    ### Stores dataset of X features with one-hpt encoded categorical features
    np.save("nsl_kdd_X_GowMat", X_GowMat)     ### Stores Gower matrix


main() # Calls the main function

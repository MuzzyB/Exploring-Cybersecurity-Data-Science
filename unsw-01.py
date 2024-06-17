### This program makes the UNSW-NB15 dataframe and its associated predictor matrices
###  and saves the outputs in pickle files


# Load packages
from UNSWNB15_Data import UNSWNB15
from inputData import InputData
import numpy as np
import pickle 



def main():
    print('\n Creating UNSW-NB15 Datasets \n\n\n')

    ### Create dataset object
    data = UNSWNB15()
    unsw = data.makeUNSW()
	

	### Create input datasets
    inp = InputData(unsw, ['id', 'label', 'attack_cat'])

    X_raw = inp.makeXdata()                 ### Creates dataset of X features
    X_OHE = inp.makeOHE()                   ### Creates dataset of X features with one-hot encoded categorical features
    X_GowMat = inp.gower_mtx()              ### Create Gower matrix


    # Store datasets in shelf file         
    unsw.to_pickle("./unsw.pkl")
    X_raw.to_pickle("./unsw_X_raw.pkl")    ### Stores dataset of X features
    X_OHE.to_pickle("./unsw_X_OHE.pkl")    ### Stores dataset of X features with one-hpt encoded categorical features
    np.save("unsw_X_GowMat", X_GowMat)     ### Stores Gower matrix



main() # Calls the main function

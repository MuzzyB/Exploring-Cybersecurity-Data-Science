### This program produces the various dimensionality reduction outputs of the UNSW-NB15 dataset
### with a sample of 75,000 packets and saves the outputs in pickle files


# Load packages
from inputData import InputData
from DimRed import DimRed
import pandas as pd
import numpy as np
import pickle 


def main():

    ### Create dataset object
    unsw = pd.read_pickle("./unsw.pkl")
    print("\nUNSW-NB15 original dataset dimensions", unsw.shape)
    print("Class proportions:\n", unsw['attack_cat'].value_counts(normalize = True))

    ### Stratified Sampling
    # Required sample size
    #Prompt user to enter value of the sample size
    num = 75000
    # Total dataset length
    Num = unsw.shape[0]
    # Proportion selected 
    prop = num/Num
    unsw = unsw.groupby('attack_cat', group_keys=False).apply(lambda x: x.sample(frac = prop, random_state= 5701, ignore_index=True))
    print("\nUNSW NB15 dataset saved with shape", unsw.shape)
    print("Class proportions:\n", unsw['attack_cat'].value_counts(normalize = True))

	### Create and save input datasets
    inp = InputData(unsw, ['id', 'label', 'attack_cat'])    
    
    X_raw = inp.makeXdata()                             ### Creates dataset of X features
    X_raw.to_pickle("./unsw_X_raw75.pkl")               ### Stores dataset of X features
    print("\n X_raw dataset saved.\n ")

    X_OHE = inp.makeOHE()                               ### Creates dataset of X features with one-hot encoded categorical features
    X_OHE.to_pickle("./unsw_X_OHE75.pkl")               ### Stores dataset of X features with one-hpt encoded categorical features
    print("X_OHE dataset saved.")

    X_GowMat = inp.gower_mtx()                          ### Create Gower matrix
    np.save("unsw_X_GowMat75", X_GowMat)                ### Stores Gower matrix
    print("X_GowMat dataset saved.")

    
    
    ## DIMENSIONALITY REDUCTION
    
    ### Create dimensionality reduction object
    dmd = DimRed(unsw)

    ### Create and store dimension reduction outputs
    pca = dmd.makePCA(X_OHE)                        ### Creates principal component analysis  dataframe
    pca.to_pickle("./unsw_pca75.pkl")               ### Stores principal component analysis  dataframe
    print("PCA saved.")


    famd = dmd.makeFAMD(X_raw)                      ### Creates factor analysis for mixed data types dataframe
    famd.to_pickle("./unsw_famd75.pkl")             ### Stores factor analysis for mixed data types dataframe
    print("FAMD saved.")


    uMap = dmd.makeUmap(X_raw)                      ### Creates uniform manifold approximation and projection  dataframe
    uMap.to_pickle("./unsw_uMap75.pkl")             ### Stores uniform manifold approximation and projection  dataframe
    print("UMAP saved.")


    tsne = dmd.maketSNE(X_GowMat)                   ### Creates t-distributed stochastic neighbor embedding dataframe
    tsne.to_pickle("./unsw_tsne75.pkl")             ### Stores t-distributed stochastic neighbor embedding dataframe
    print("t-SNE saved.")


    isomap = dmd.makeIsomap(X_GowMat)               ### Creates isometric mapping dataframe
    isomap.to_pickle("./unsw_isomap75.pkl")         ### Stores isometric mapping dataframe
    print("ISOMAP saved.")

main() # Calls the main function

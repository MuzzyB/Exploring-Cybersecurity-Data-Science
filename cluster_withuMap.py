### This program produces the various clustering outputs of both the UNSW-NB15 dataset
### and NSL KDD dataset saves the outputs in pickle files


# Load packages
from Clustering import Clustering
import pandas as pd
import numpy as np
import pickle 



def main():

# Retrieve datasets in shelf file 
    nsl_kdd = pd.read_pickle("./nsl_kdd.pkl")
    print("\nNSL KDD dataset shape", nsl_kdd.shape)
    print("Class proportions:\n", nsl_kdd['classType'].value_counts(normalize = True))

    unsw = pd.read_pickle("./unsw.pkl")    
    print("\nUNSW-NB15 dataset shape", unsw.shape)
    print("Class proportions:\n", unsw['attack_cat'].value_counts(normalize = True))

    nsl_kdd_uMap = pd.read_pickle("./nsl_kdd_uMap.pkl") 
    print("\nNSL KDD UMAP shape", nsl_kdd_uMap.shape)
    unsw_uMap = pd.read_pickle("./unsw_uMap.pkl") 
    print("\nUNSW-NB15 UMAP shape", unsw_uMap.shape)


    ### Create clustering objects
    unsw_cl = Clustering(unsw)
    nsl_kdd_cl = Clustering(nsl_kdd)

    ### Perform clustering

    nsl_kdd_kmeans = nsl_kdd_cl.makeKmeans(nsl_kdd_uMap, 5)                 ### Creates k-Means dataframe
    nsl_kdd_kmeans.to_pickle("./nsl_kdd_kmeans.pkl")              
    print("k-means Clustering saved.")


    nsl_kdd_single = nsl_kdd_cl.makeSingle(nsl_kdd_uMap, 5,'euclidean')     ### Creates single dataframe 
    nsl_kdd_single.to_pickle("./nsl_kdd_single.pkl")              
    print("Single Clustering saved.")

                    
    nsl_kdd_avg = nsl_kdd_cl.makeAverage(nsl_kdd_uMap, 5,'euclidean')       ### Creates average dataframe
    nsl_kdd_avg.to_pickle("./nsl_kdd_avg.pkl")              
    print("Average Clustering saved.")

    nsl_kdd_cmpl = nsl_kdd_cl.makeComplete(nsl_kdd_uMap, 5,'euclidean')     ### Creates complete dataframe
    nsl_kdd_cmpl.to_pickle("./nsl_kdd_cmpl.pkl")              
    print("Spectral Clustering saved.")


    nsl_kdd_db = nsl_kdd_cl.makedbscan(nsl_kdd_uMap, 0.06, 250,'euclidean') ### Creates dbscan dataframe
    nsl_kdd_db.to_pickle("./nsl_kdd_db.pkl")              
    print("DBSCAN Clustering saved.")


    ### Creates hdbscan dataframe
    nsl_kdd_hdb = nsl_kdd_cl.makehdbscan(nsl_kdd_uMap, clust_size = 2500, minimum_samples = 1000, metric = 'euclidean')
    nsl_kdd_hdb.to_pickle("./nsl_kdd_hdb.pkl")              
    print("HDBSCAN Clustering saved.")

    ### Creates optics dataframe
    nsl_kdd_opt = nsl_kdd.cl.makeoptics(nsl_kdd_uMap, clust_size = 2500, minimum_samples = 1000, metric = 'minkowski')
    nsl_kdd_opt.to_pickle("./nsl_kdd_opt.pkl")              
    print("OPTICS Clustering saved.")


    unsw_kmeans = unsw_cl.makeKmeans(unsw_uMap, 10)                         ### Creates k-Means  dataframe
    unsw_kmeans.to_pickle("./unsw_kmeans.pkl")              
    print("k-means Clustering saved.")


    unsw_single = unsw_cl.makeSingle(unsw_uMap, 10,'euclidean')             ### Creates single dataframe
    unsw_single.to_pickle("./unsw_single.pkl")              
    print("unsw_single Clustering saved.")


    unsw_avg = unsw_cl.makeAverage(unsw_uMap, 10,'euclidean')               ### Creates average dataframe
    unsw_avg.to_pickle("./unsw_avg.pkl")              
    print("Average Clustering saved.")


    unsw_cmpl = unsw_cl.makeComplete(unsw_uMap, 10,'euclidean')             ### Creates complete dataframe
    unsw_cmpl.to_pickle("./unsw_cmpl.pkl")              
    print("Complete Clustering saved.")


    unsw_db = unsw_cl.makedbscan(unsw_uMap, 0.03, 200,'euclidean')          ### Creates dbscan dataframe
    unsw_db.to_pickle("./unsw_db.pkl")              
    print("DBSCAN Clustering saved.")


    ### Creates hdbscan dataframe
    unsw_hdb = unsw_cl.makehdbscan(nsl_kdd_uMap, clust_size = 2500, minimum_samples = 1000, metric = 'euclidean')
    unsw_hdb.to_pickle("./unsw_hdb.pkl")              
    print("HDBSCAN Clustering saved.")

                    
    ### Creates optics dataframe
    unsw_opt = unsw_cl.makeoptics(nsl_kdd_uMap, clust_size = 2500, minimum_samples = 1000, metric = 'minkowski')
    unsw_opt.to_pickle("./unsw_opt.pkl")              
    print("OPTICS Clustering saved.")


    nsl_kdd_affProp = nsl_kdd_cl.makeaffProp(nsl_kdd_uMap)                    ### Creates kaffinity propagation dataframe
    nsl_kdd_affProp.to_pickle("./nsl_kdd_affProp.pkl")              
    print("Affinity Propagation Clustering saved.")


    unsw_affProp = unsw_cl.makeaffProp(unsw_uMap)                           ### Creates affinity propagation dataframe
    unsw_affProp.to_pickle("./unsw_affProp.pkl")              
    print("Affinity Propagation Clustering saved.")


    nsl_kdd_bir = nsl_kdd_cl.makeBirch(nsl_kdd_uMap, 5)                     ### Creates birch dataframe
    nsl_kdd_bir.to_pickle("./nsl_kdd_bir_uMap75.pkl")              
    print("BIRCH Clustering saved.")

                    
    unsw_bir = unsw_cl.makeBirch(unsw_uMap, 10)                             ### Creates birch dataframe
    unsw_bir.to_pickle("./unsw_bir_uMap75.pkl")              
    print("BIRCH Clustering saved.")


    snn = nsl_kdd_cl.makeSNN(nsl_kdd_uMap, 2500, 0.06)                      ### Creates SNN dataframe
    snn.to_pickle("./nsl_kdd_snn_uMap75.pkl")              
    print("SNN saved.")

                    
    unsw_snn = unsw_cl.makeSNN(unsw_uMap, 2500, 0.06)                       ### Creates SNN dataframe
    unsw_snn.to_pickle("./unsw_snn_uMap75.pkl")              
    print("SNN saved.")


main() # Calls the main function

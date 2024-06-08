import numpy as np
import pandas as pd
from dask.distributed import Client

import joblib
import time

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes  # Assuming you are using singleodes for KPrototypes
from stepmix.stepmix import StepMix         # Gaussian Mixture Models Clustering
from snn import SNN
from stepmix.utils import get_mixed_descriptor
from sklearn import metrics




class Clustering:
    ###### CLUSTERING
    # Construct an object that will perform various clustering techniques on the data

    def __init__(self, data):
        # data is the input data
        self.data = data            # Defines the input data field 
        self.num = data.shape[0]  
    

    ## Cluster Evaluation
    def cluster_evaluation(name, estimator, labels, data, metric):
        if hasattr(estimator, 'labels_'):
            y_pred = estimator.labels_.astype(int)
        elif hasattr(estimator, 'predict'):
                    y_pred = estimator.predict(data) 
        else:
            y_pred = estimator   
        print('% s  %8.3f   %8.3f   %8.3f   \t %8.3f   %8.3f    %8.3f'
        % (name,
            metrics.silhouette_score(data, y_pred, metric=metric, sample_size = 300),
            metrics.calinski_harabasz_score(data, y_pred),
            metrics.davies_bouldin_score(data, y_pred),
            metrics.adjusted_rand_score(labels, y_pred),
            metrics.adjusted_mutual_info_score(labels,  y_pred),
            metrics.v_measure_score(labels, y_pred)))


    ### Prototype-Based Clustering        

    #### K-means Clustering
    def makeKmeans(self, X, num_clusters):
        try:
            # Perform k-means Clustering
            print('\n Start k-means Clustering')       
            time_start = time.time()

            k_means = cluster.KMeans(n_clusters = num_clusters)
            k_means.fit(X)
            labels = k_means.labels_

            print('k-means done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            km_result = pd.DataFrame(labels, index = X.index, columns=['Cluster ID Kmeans'])

            #Prints the count of each cluster group
            print(km_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return km_result
    

    #### K-prototypes Clustering for mixed data
    def makeKprototype(self, X, num_clusters, cat_cols):
        try:        
            client = Client(processes=False)             # create local cluster
            # Perform k-prototypes Clustering
            print('\n Start k-prototypes Clustering')
            time_start = time.time()

            data_1 = X
            kprot_data = data_1.copy()
            #Pre-processing
            for c in data_1.select_dtypes(exclude='object').columns:
                pt = PowerTransformer()
                kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

            categorical_columns = cat_cols  #make sure to specify correct indices like [1, 2, 3]

            #Actual clustering
            kprot = KPrototypes(n_clusters= num_clusters, init='Cao', n_jobs = -1)

            with joblib.parallel_backend('dask'):
                kprot_labels = kprot.fit_predict(kprot_data, categorical=categorical_columns)

            print('k-Prototype done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            kprot_result = pd.DataFrame(kprot_labels, index = X.index, columns=['Cluster ID kprot'])

            #Prints the count of each cluster group
            print(kprot_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return kprot_result
    


    #### Gaussian Mixture Models Clustering for Mixed Data
    def makeGMM(self, mixed_data, mixed_descriptor): 
        try:      
            # Perform Gaussian Mixture Model Clustering with mixed data types
            print('\n Start Gaussian Mixture Models Clustering')
            time_start = time.time()

            # Mixed-type mixture model
            gmm = StepMix(n_components=3, measurement=mixed_descriptor, verbose=0, random_state=123)

            # Fit model
            gmm.fit(mixed_data)

            #predictions from gmm
            labels = gmm.predict(mixed_data)

            print('GMM done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            gmm_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID GMM'])

            #Prints the count of each cluster group
            print(gmm_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return gmm_result 



    #### Gaussian Mixture Models Clustering for Numeric Data
    def makegmm(self, X, n_components): 
        try:      
            # Perform Gaussian Mixture Model Clustering with mixed data types
            print('\n Start Gaussian Mixture Models Clustering')
            time_start = time.time()

            # Mixture model
            gmm = GaussianMixture(n_components = n_components)

            # Fit model
            gmm.fit(X)

            #predictions from gmm
            labels = gmm.predict(X)

            print('GMM done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            gmm_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID GMM'])

            #Prints the count of each cluster group
            print(gmm_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return gmm_result
            
    



    #### BIRCH Clustering
    def makeBirch(self, X, n_clusters): 
        try:      
            # Perform Gaussian Mixture Model Clustering with mixed data types
            print('\n Start BIRCH Clustering')
            time_start = time.time()

            # Mixture model
            bir = cluster.Birch(threshold=0.5, n_clusters = n_clusters)

            # Fit model
            bir.fit(X)

            #predictions from gmm
            labels = bir.predict(X)

            print('BIRCH done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            bir_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID Birch'])

            #Prints the count of each cluster group
            print(bir_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return bir_result
            
    


    ### Graph-Based Clustering

    #### Agglomerative / Hierarchical Clustering
    ##### Single Link (MIN)
    def makeSingle(self, X, num_clusters, metric):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform k-means Clustering
            print('\n Start Single Link (MIN) Clustering')                
            time_start = time.time()

            single_clst = cluster.AgglomerativeClustering(n_clusters = num_clusters, metric=metric,  linkage='single')

            with joblib.parallel_backend('dask'):
                single_clst.fit(X)
            single_labels = single_clst.labels_

            print('Single Hierarchical done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            single_result = pd.DataFrame(single_labels, index = self.data.index, columns=['Cluster ID Single'])

            #Prints the count of each cluster group
            print(single_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return single_result
    

    ##### Average Link
    def makeAverage(self, X, num_clusters,metric):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform k-means Clustering
            print('\n Start Average Link (MIN) Clustering')       
            time_start = time.time()

            avg_clst = cluster.AgglomerativeClustering(n_clusters = num_clusters, metric=metric,  linkage='average')
            with joblib.parallel_backend('dask'):
                avg_clst.fit(X)
            avg_labels = avg_clst.labels_

            print('Average Hierarchical done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            avg_result = pd.DataFrame(avg_labels, index = self.data.index, columns=['Cluster ID avg'])

            #Prints the count of each cluster group
            print( avg_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return avg_result
    

    ##### Complete Link
    def makeComplete(self, X, num_clusters,metric):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform k-means Clustering
            print('\n Start Complete Link (MIN) Clustering')       
            time_start = time.time()

            cmp_clst = cluster.AgglomerativeClustering(n_clusters = num_clusters, metric=metric,  linkage='complete')
            with joblib.parallel_backend('dask'):
                cmp_clst.fit(X)
            cmp_labels = cmp_clst.labels_

            print('Complete Hierarchical done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            cmp_result = pd.DataFrame(cmp_labels, index = self.data.index, columns=['Cluster ID compl'])

            #Prints the count of each cluster group
            print(cmp_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return cmp_result
            
    

    #### Spectral Clustering [Elliptical and 2D Data]
    def makeSpeclut(self, X, num_clusters, num_neighbors,metric):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform Spectral Clustering
            print('\n Start Spectral Clustering')
            time_start = time.time()

            # training spectral clustering model
            spectral = cluster.SpectralClustering(n_clusters = num_clusters, random_state=1, affinity=metric, n_neighbors = num_neighbors, n_jobs= -1)
            with joblib.parallel_backend('dask'):
                spectral.fit(X)

            #predictions from spectral
            labels = spectral.labels_

            print('Spectral Clustering done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            spectral_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID SPC'])


            #Prints the count of each cluster group
            print(spectral_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return spectral_result
    


    ### Density-Based Clustering
    
    #### DBSCAN
    def makedbscan(self, X, epsilon, minimum_samples,metric):
        try:    
            client = Client(processes=False)             # create local cluster
            # Perform DBSCAN Clustering
            print('\n Start DBSCAN Clustering')
            time_start = time.time()

            # training optics clustering model
            db = cluster.DBSCAN(eps=epsilon, min_samples=minimum_samples, metric=metric, n_jobs = -1)

            with joblib.parallel_backend('dask'):
                db.fit(X)

            #predictions from optics
            labels = db.labels_

            print('DBSCAN Clustering done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            db_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID DBSCAN'])


            #Prints the count of each cluster group
            print(db_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return db_result
        
     

    #### HDBSCAN
    def makehdbscan(self, X, clust_size = 5, minimum_samples = None, metric ='euclidean'):
        try:    
            client = Client(processes=False)             # create local cluster
            # Perform HDBSCAN Clustering
            print('\n Start HDBSCAN Clustering')
            time_start = time.time()

            # training optics clustering model
            hdb = cluster.HDBSCAN(min_cluster_size=clust_size, min_samples=minimum_samples, metric=metric, n_jobs = -1)

            with joblib.parallel_backend('dask'):
                hdb.fit(X)

            #predictions from optics
            labels = hdb.labels_

            print('DBSCAN Clustering done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            hdb_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID HDBSCAN'])


            #Prints the count of each cluster group
            print(hdb_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return hdb_result

         

    #### OPTICS
    def makeoptics(self, X, clust_size, minimum_samples, metric):
        try:    
            client = Client(processes=False)             # create local cluster
            # Perform OPTICS Clustering
            print('\n Start OPTICS Clustering')
            time_start = time.time()

            # training optics clustering model
            opt = cluster.OPTICS(min_samples=minimum_samples, min_cluster_size = clust_size, metric=metric, n_jobs = -1)

            with joblib.parallel_backend('dask'):
                opt.fit(X)

            #predictions from optics
            labels = opt.labels_

            print('OPTICS Clustering done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            opt_result = pd.DataFrame(labels, index = self.data.index, columns=['Cluster ID OPTICS'])


            #Prints the count of each cluster group
            print(opt_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return opt_result   




    def makeSNN(self, X, num_neighbor, min_shared_neigh_prop):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform Shared Nearest Neighbors (SNN) Clustering
            print('\n Start SNN Clustering')
            time_start = time.time()

            # training snn clustering model predictions
            snn = SNN(neighbor_num=num_neighbor, min_shared_neighbor_proportion=min_shared_neigh_prop)   # Change neighbor_num to be < sample size
            with joblib.parallel_backend('dask'):
                snn_labels = snn.fit_predict(X)
            snn_result = pd.DataFrame(snn_labels, index = self.data.index, columns=['Cluster ID SNN'])

            print('SNN done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)

            #Prints the count of each cluster group
            print(snn_result.value_counts(sort=False))

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return snn_result
        



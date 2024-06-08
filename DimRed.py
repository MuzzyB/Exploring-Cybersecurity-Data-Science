import numpy as np
import pandas as pd
from dask.distributed import Client
import joblib

import time

from sklearn.preprocessing import StandardScaler

import umap
import prince
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap


class DimRed:        
    ## Dimension Reduction and Exploratory Data Analysis (EDA)
    # Construct an object that will perform various dimensionality reduction on the data

    def __init__(self, data):
        # data is the input data
        self.data = data            # Defines the input data field 
        self.num = data.shape[0]  


    ### PCA (Principal Component Analysis)
    def makePCA(self, X):
        try:
            # Perform t-SNE
            print('\n Start Principal Component Analysis')
            time_start = time.time()

            X = X.select_dtypes(exclude='object')
            pca = PCA(n_components=3)
            pca_embed = pca.fit_transform(X)   # Use gower matrix

            print('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)
            pca_result = pd.DataFrame(pca_embed, index = self.data.index, columns=['pca1', 'pca2', 'pca3'])

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return pca_result
    

    ### UMAP Embedding
    def umap_embed(self, df, n_components=3, n_jobs = -1, intersection=False):

        # Perform UMAP
        numerical = df.select_dtypes(exclude='object')
        # Scaling the data
        scaler = StandardScaler()
        numerical = scaler.fit_transform(numerical)
            
        ##preprocessing categorical
        categorical = df.select_dtypes(include=['object', 'category'])
        categorical = pd.get_dummies(categorical)

        #Embedding numerical & categorical
        fit1 = umap.UMAP(random_state=12, n_components=n_components, n_jobs=n_jobs).fit(numerical)        
        fit2 = umap.UMAP(metric='dice', n_neighbors=250, n_components=n_components,n_jobs=n_jobs).fit(categorical)


        # intersection will resemble the numerical embedding more.
        if intersection:
            embedding = fit1 * fit2

        # union will resemble the categorical embedding more.
        else:
            embedding = fit1 + fit2

        umap_embedding = embedding.embedding_

        return umap_embedding
    


    def makeUmap(self, X, n_components):
        try:
            client = Client(processes=False)             # create local cluster
            # Perform UMAP dimensionality reduction
            print('\n Start Uniform Manifold Approximation and Projection')
            time_start = time.time()

            with joblib.parallel_backend('dask'):
                umap_embed =  self.umap_embed(X, n_components=n_components, intersection=True)   # Intersection is True since we have only three categorical variables

            print('UMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)


            num_columns = umap_embed.shape[1]  # This gives the number of columns in tsne_embed
            column_names = [f'umap{i+1}' for i in range(num_columns)]  # Generating column names: tsne1, tsne2, ...
            umap_result = pd.DataFrame(umap_embed, index = self.data.index, columns = column_names)

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return umap_result
    
    

    ### t-SNE (t distributed Stochastic Neighbor Embedding)
    def maketSNE(self, X):
        try:
            # Perform t-SNE
            print('\n Start t distributed Stochastic Neighbor Embedding')
            time_start = time.time()

            tsne = TSNE(n_components=3, metric='precomputed', init = 'random', verbose=1, perplexity=40, n_iter=300)
            tsne_embed = tsne.fit_transform(X)   # Use gower matrix

            print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)
            tsne_result = pd.DataFrame(tsne_embed, index = self.data.index, columns=['tsne1', 'tsne2', 'tsne3'])

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return tsne_result
    


    ### Isometric Feature Mapping (ISOMAP)
    def makeIsomap(self, X):
        try:
            client = Client(processes=False)             # create local cluster
            ## Perform ISOMAP mapping
            print('\n Start Isometric Feature Mapping')
            time_start = time.time()

            iso = Isomap(n_neighbors=30, n_components=3, metric = 'precomputed', n_jobs= -1)
            with joblib.parallel_backend('dask'):
                iso_embed = iso.fit_transform(X)

            print('ISOMAP done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)
            iso_result = pd.DataFrame(iso_embed, index = self.data.index, columns=['iso1', 'iso2', 'iso3'])

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return iso_result



    ### MDS (Multidimensional Scaling)
    def makeMDS(self, X):
        try:
            client = Client(processes=False)             # create local cluster

            # Perform Multidimensional Scaling
            print('\n Start Multidimensional Scaling')
            time_start = time.time()

            mds = MDS(n_components=3, metric = False, dissimilarity ='precomputed', random_state=5701, n_jobs= -1)
            with joblib.parallel_backend('dask'):
                mds_embed = mds.fit_transform(X)   # Use Gower Matrix

            print('MDS done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)
            mds_result = pd.DataFrame(mds_embed, index = self.data.index, columns=['mds1', 'mds2', 'mds3'])

        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return mds_result
    


    ### Factor Analysis of Mixed Data (FAMD)
    def makeFAMD(self, X):
        try:
            ## Dimensional reduction using FAMD
            print('\n Start Factor Analysis of Mixed Data')
            time_start = time.time()

            famd = prince.FAMD(n_components=3, n_iter=3, copy=True, check_input=True, 
                                random_state=42, engine="sklearn", handle_unknown="error")
            famd = famd.fit(X)
            famd_result = famd.row_coordinates(X)

            print('FAMD done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)


        except MemoryError:
            print(MemoryError)
        except Exception as error:
                print(error)
                print(error.__doc__)
        else:
            return famd_result
        


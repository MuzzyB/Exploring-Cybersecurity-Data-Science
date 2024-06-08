import pandas as pd
from dask.distributed import Client

import joblib
import time
import gower


class InputData:
    try:
        # Construct an object that will prepare data into different input formats
        
        def __init__(self, data, dropVars):
            # data is the input data
            # dropVars is the array of features to be dropped eg ['label', 'classType']
            self.data = data            # Defines the input data field 
            self.dropVars = dropVars
            self.num = data.shape[0]

        # Create predictor matrix
        def makeXdata(self):
            Xdata = self.data.drop(self.dropVars, axis = 1, inplace = False)
            return Xdata
        
        # Process categorical data through one hot encoding
        def makeOHE(self):
            df = self.makeXdata()
            categorical_columns = df.select_dtypes(include=['object']).columns
            encoded_categorical = pd.get_dummies(df[categorical_columns])
            X_data_OHE = pd.concat([df.drop(columns=categorical_columns), pd.DataFrame(encoded_categorical)], axis=1)
            return X_data_OHE
        
        # Compute Gower matrix
        def gower_mtx(self):
            client = Client(processes=False)             # create local cluster
            print('\n Start Compute Gower matrix')
            time_start = time.time()

            df = self.makeXdata()
            with joblib.parallel_backend('dask'):
                gower_mtx = gower.gower_matrix(df)

            print('gower distance done! Time elapsed: {} seconds'.format(time.time()-time_start))
            print('Dataset size of: ', self.num)
            return gower_mtx
    except MemoryError:
        print(MemoryError)

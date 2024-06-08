import pandas as pd
import time


class UNSWNB15:
    # Construct an object that will prepare data into different input formats
    
    # Create predictor matrix
    def makeUNSW(self):
        #train = pd.read_csv(r'C:/Users/Owner/OneDrive/Documents\Muu/Coursera/Plan B Thesis/UNSW/UNSW_NB15_training-set.csv')
        #test = pd.read_csv(r'C:/Users/Owner/OneDrive/Documents\Muu/Coursera/Plan B Thesis/UNSW/UNSW_NB15_testing-set.csv')

        ## For reading data in Ubuntu
        train = pd.read_csv('UNSW_NB15_training-set.csv')
        test = pd.read_csv('UNSW_NB15_testing-set.csv')
        fulldata = pd.concat([train, test])

        ## Categorical data type
        fulldata['proto'] = fulldata['proto'].astype(object)
        fulldata['service'] = fulldata['service'].astype(object)
        fulldata['state'] = fulldata['state'].astype(object)

        return fulldata
    
    

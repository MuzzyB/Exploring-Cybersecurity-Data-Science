import pandas as pd


class NSLKDD:
    # Construct an object that will prepare data into different input formats
    
    # Create predictor matrix
    def makeNSLKDD(self):
        # train = pd.read_csv(r'C:/Users/Owner/OneDrive/Documents\Muu/Coursera/Plan B Thesis/NSL KDD/nsl_kdd_train.csv')
        # test = pd.read_csv(r'C:/Users/Owner/OneDrive/Documents\Muu/Coursera/Plan B Thesis/NSL KDD/nsl_kdd_test.csv')
        train = pd.read_csv('nsl_kdd_train.csv')
        test = pd.read_csv('nsl_kdd_test.csv')
        fulldata = pd.concat([train, test])

        # Define a mapping of old categories to new categories
        category_mapping = {
            'worm': 'dos',
            'land': 'dos',
            'smurf': 'dos',
            'udpstorm': 'dos',
            'teardrop': 'dos',
            'pod': 'dos',
            'mailbomb': 'dos',
            'neptune': 'dos',
            'processtable': 'dos',
            'apache2': 'dos',
            'back': 'dos',
            'ipsweep': 'probe',
            'nmap': 'probe',
            'satan': 'probe',
            'portsweep': 'probe',
            'mscan': 'probe',
            'saint': 'probe',
            'warezclient': 'r2l',
            'snmpgetattack': 'r2l',
            'warezmaster': 'r2l',
            'imap': 'r2l',
            'snmpguess': 'r2l',
            'named': 'r2l',
            'multihop': 'r2l',
            'phf': 'r2l',
            'spy': 'r2l',
            'sendmail': 'r2l',
            'ftp_write': 'r2l',
            'xsnoop': 'r2l',
            'xlock': 'r2l',
            'guess_passwd': 'r2l',
            'buffer_overflow': 'u2r',
            'sqlattack': 'u2r',
            'rootkit': 'u2r',
            'perl': 'u2r',
            'xterm': 'u2r',
            'loadmodule': 'u2r',
            'ps': 'u2r',
            'httptunnel': 'u2r',
            'normal': 'normal'
        }

        # Create a new column with the mapped categories
        fulldata['classType'] = fulldata['label'].map(category_mapping)


        ## Categorical data type
        fulldata['protocol_type'] = fulldata['protocol_type'].astype(object)
        fulldata['service'] = fulldata['service'].astype(object)
        fulldata['flag'] = fulldata['flag'].astype(object)
        ## Binary data type
        fulldata['land'] = fulldata['land'].astype(object)
        fulldata['logged_in'] = fulldata['logged_in'].astype(object)
        fulldata['root_shell'] = fulldata['root_shell'].astype(object)
        fulldata['su_attempted'] = fulldata['su_attempted'].astype(object)
        fulldata['is_host_login'] = fulldata['is_host_login'].astype(object)
        fulldata['is_guest_login'] = fulldata['is_guest_login'].astype(object)

        return fulldata
    
    

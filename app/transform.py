import pandas as pd
import numpy as np

class DataTransform:
    def transform(self):
        purchaseData = pd.read_csv('app/data/purchase_data.csv')
        purchaseData = purchaseData.drop(['InvoiceNo', 'InvoiceDate'], axis=1)
        purchaseData = purchaseData.dropna(
            subset=['StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID', 'Country'])

        customerIdArr = purchaseData['CustomerID'].unique().astype(np.int64)
        customerIds = pd.DataFrame(customerIdArr, columns=['CustomerId'])


        data_folder = 'app/data'

        userData = pd.read_csv(data_folder + '/user_data.csv')
        userData['login'] = userData['email']
        userData['activated'] = True
        userData['password_hash'] = '$2a$10$HdQfU7GJ8xTh7V23joLEe.qcySJz./z6bO0NKfJLNwbkfRYG9mXbu'
        userData['lang_key'] = 'en'
        userData['created_date'] = '2019-03-01 11:16:45.079'
        userData['created_by'] = 'system'
        userData['last_modified_by'] = 'system'

        customerData = customerIds.join(userData)
        columnNames = ['id', 'first_name', 'last_name', 'email', 'login',
                       'activated', 'password_hash', 'lang_key', 'created_date', 'created_by', 'last_modified_by']
        customerData.columns = columnNames

        divisor = 500

        def saveCsvInChunks(columnNames, data, path):
            currentMin = 0
            currentMax = divisor
            collectDf = pd.DataFrame(columns=columnNames)
            dataLength = data.shape[0]
            while (currentMax <= dataLength):
                collectDf = data.iloc[currentMin:currentMax]
                collectDf.to_csv(path + str(currentMax) + '.csv', sep=';', index=False)
                if (currentMax == dataLength):
                    break
                currentMin = currentMax + 1
                currentMax += divisor
                if (currentMax > dataLength):
                    currentMax = dataLength

        saveCsvInChunks(columnNames, customerData, data_folder + '/user_seed/user_seed_data')

        userAuthorities = customerIds.copy()
        userAuthorities['authority_name'] = 'ROLE_USER'
        userAuthorityColumnNames = ['user_id', 'authority_name']
        userAuthorities.columns = userAuthorityColumnNames

        saveCsvInChunks(userAuthorityColumnNames, userAuthorities, data_folder + '/authority_seed/authority_seed_data')


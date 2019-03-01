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

        userData = pd.read_csv('app/data/user_data.csv')
        userData['login'] = userData['email']
        userData['activated'] = True
        userData['password_hash'] = '$2a$10$HdQfU7GJ8xTh7V23joLEe.qcySJz./z6bO0NKfJLNwbkfRYG9mXbu'
        userData['lang_key'] = 'en'
        userData['created_by'] = 'system'
        userData['last_modified_by'] = 'system'

        customerData = customerIds.join(userData)
        customerData.columns = ['id', 'first_name', 'last_name', 'email', 'login',
                                'activated', 'password_hash', 'lang_key', 'created_by', 'last_modified_by']
        customerData.set_index('id')
        customerData.to_csv(
            'app/data/user_seed/user_seed_data.csv', sep=';', index=False)

        userAuthorities = customerIds.copy()
        userAuthorities['authority_name'] = 'ROLE_USER'
        userAuthorities.columns = ['user_id', 'authority_name']
        userAuthorities.set_index('user_id')
        userAuthorities.to_csv(
            'app/data/user_seed/authority_seed_data.csv', sep=';', index=False)


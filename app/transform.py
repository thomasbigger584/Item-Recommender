import pandas as pd
import numpy as np


class DataTransform:
    def transform(self):
        purchaseData = pd.read_csv('app/data/purchase_data.csv')
        purchaseData = purchaseData.drop(['InvoiceNo', 'InvoiceDate'], axis=1)
        purchaseData = purchaseData.dropna(subset=['StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID', 'Country'])

        

        print(purchaseData)

        # customersColumnList = list(purchaseData['CustomerID']).dropna

        # print(customersColumnList)

        # maxValue = purchaseData['CustomerID'].max()
        # print(maxValue)

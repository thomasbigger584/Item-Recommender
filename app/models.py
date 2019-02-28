from django.db import models
from django.utils import timezone

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

# Create your models here.


class LogMessage(models.Model):
    message = models.CharField(max_length=300)
    log_date = models.DateTimeField("date logged")

    def __str__(self):
        """Returns a string representation of a message."""
        date = timezone.localtime(self.log_date)
        return f"'{self.message}' logged on {date.strftime('%A, %d %B, %Y at %X')}"


# constants
data_folder = 'app/data'
model_folder = 'app/trained_models'
popularity = 'popularity'
cosine = 'cosine'
pearson = 'pearson'

split_ratio = 0.2


class ItemRecommender:
    def trainModels(self):
        def printLarge(data, rows=1000):
            pd.set_option('display.max_columns', 1000)
            pd.set_option('display.max_rows', rows)
            print(data)
            pd.reset_option('display.max_columns')
            pd.reset_option('display.max_rows')

        s = time.time()
        transactions = pd.read_csv(data_folder + '/trx_data.csv')

        transactions['products'] = transactions['products'].apply(
            lambda x: [int(i) for i in x.split('|')])

        customer_indexed_transactions = transactions.set_index('customerId')
        """
        shape (62483, 1)
        customerId                                       products
        0                                                    [20]
        1               [2, 2, 23, 68, 68, 111, 29, 86, 107, 152]
        2                      [111, 107, 29, 11, 11, 11, 33, 23]
        3                                              [164, 227]
        5                                                  [2, 2]
        6                                     [144, 144, 55, 266]
        7                                         [135, 206, 259]
        8                                          [79, 8, 8, 48]
        9                                        [102, 2, 2, 297]
        10                                     [84, 77, 290, 260]
        """

        products_indexed = customer_indexed_transactions['products']\
            .apply(pd.Series).reset_index()
        """
        shape (62483, 11)
               customerId      0      1      2      3      4      5      6      7      8      9
        0               0   20.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
        1               1    2.0    2.0   23.0   68.0   68.0  111.0   29.0   86.0  107.0  152.0
        2               2  111.0  107.0   29.0   11.0   11.0   11.0   33.0   23.0    NaN    NaN
        3               3  164.0  227.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
        4               5    2.0    2.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN    NaN
        5               6  144.0  144.0   55.0  266.0    NaN    NaN    NaN    NaN    NaN    NaN
        6               7  135.0  206.0  259.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN
        7               8   79.0    8.0    8.0   48.0    NaN    NaN    NaN    NaN    NaN    NaN
        8               9  102.0    2.0    2.0  297.0    NaN    NaN    NaN    NaN    NaN    NaN
        9              10   84.0   77.0  290.0  260.0    NaN    NaN    NaN    NaN    NaN    NaN
        """

        # -----------------------------------------------------------------------------------------
        # Our goal here is to break down each list of items in the products column into rows
        # and count the number of products bought by a user

        # Create data with user, item, and target field

        melted_dataset = pd.melt(products_indexed,
                                 id_vars=['customerId'],
                                 value_name='products')
        """
        shape (624830, 3)
                customerId variable  products
        0                0        0      20.0
        1                1        0       2.0
        2                2        0     111.0
        3                3        0     164.0
        4                5        0       2.0
        5                6        0     144.0
        6                7        0     135.0
        7                8        0      79.0
        8                9        0     102.0
        9               10        0      84.0
        ...
        624824       14302        9     282.0
        """

        melted_dataset_dropped_na = melted_dataset.dropna()
        """
        shape (211478, 3)
                customerId variable  products
        0                0        0      20.0
        1                1        0       2.0
        2                2        0     111.0
        3                3        0     164.0
        4                5        0       2.0
        5                6        0     144.0
        6                7        0     135.0
        7                8        0      79.0
        8                9        0     102.0
        9               10        0      84.0
        """

        # print(melted_dataset_dropped_na[melted_dataset_dropped_na['customerId'] == 1])

        dataset_dropped_var = melted_dataset_dropped_na.drop(
            ['variable'], axis=1)
        """
        shape (211478, 2)
                customerId  products
        0                0      20.0
        1                1       2.0
        2                2     111.0
        3                3     164.0
        4                5       2.0
        5                6     144.0
        6                7     135.0
        7                8      79.0
        8                9     102.0
        9               10      84.0
        """

        data_group_by = dataset_dropped_var\
            .groupby(['customerId', 'products'])
        """
            pandas.core.groupby.generic.DataFrameGroupBy object at 0x125b5b940
        """

        count_aggregate = data_group_by\
            .agg({'products': 'count'})
        """
        shape (133585, 1)
                            products
        customerId products
        0          1.0              2
                   13.0             1
                   19.0             3
                   20.0             1
                   31.0             2
                   52.0             1
                   69.0             2
                   93.0             3
                   136.0            2
                   157.0            1
                   198.0            1
        """

        count_aggregate_renamed_cols = count_aggregate\
            .rename(columns={'products': 'purchase_count'})\
            .reset_index()\
            .rename(columns={'products': 'productId'})

        count_aggregate_renamed_cols['productId'] =\
            count_aggregate_renamed_cols['productId'].astype(np.int64)
        """
        shape (133585, 3)
                customerId  productId  purchase_count
        0                0          1               2
        1                0         13               1
        2                0         19               3
        3                0         20               1
        4                0         31               2
        5                0         52               1
        6                0         69               2
        7                0         93               3
        8                0        136               2
        9                0        157               1
        10               0        198               1
        """

        data = count_aggregate_renamed_cols
        # -----------------------------------------------------------------------------------------

        def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
            if name == popularity:
                model = tc.popularity_recommender.create(train_data,
                                                         user_id=user_id,
                                                         item_id=item_id,
                                                         target=target)
            elif name == cosine:
                model = tc.item_similarity_recommender.create(train_data,
                                                              user_id=user_id,
                                                              item_id=item_id,
                                                              target=target,
                                                              similarity_type=cosine)
            elif name == pearson:
                model = tc.item_similarity_recommender.create(train_data,
                                                              user_id=user_id,
                                                              item_id=item_id,
                                                              target=target,
                                                              similarity_type=pearson)
            model.save(model_folder + '/' + name)
            return model

        # variables to define field names
        user_id = 'customerId'
        item_id = 'productId'
        target = 'purchase_count'
        users_to_recommend = list(transactions[user_id])
        n_rec = 10  # number of items to recommend
        n_display = n_rec * 3

        # The popularity model takes the most popular items for recommendation.
        # These items are products with the highest number of sells across customers.
        popularity_model = model(tc.SFrame(data), popularity, user_id, item_id, target,
                                 users_to_recommend, n_rec, n_display)

        # In collaborative filtering, we would recommend items based on how similar users purchase items.
        # For instance, if customer 1 and customer 2 bought similar items, e.g. 1 bought X, Y, Z and 2 bought X, Y, we would recommend an item Z to customer 2.
        cos_model = model(tc.SFrame(data), cosine, user_id, item_id, target,
                          users_to_recommend, n_rec, n_display)

        # # Similarity is the pearson coefficient between the two vectors.
        pear_model = model(tc.SFrame(data), pearson, user_id, item_id, target,
                           users_to_recommend, n_rec, n_display)

    def query(self):
        users_to_recommend = list([1])
        n_rec = 10

        def loadModel(name):
            return tc.load_model(model_folder + '/' + name)

        def recommend(model):
            return model.recommend(users=users_to_recommend, k=n_rec)

        def recommendationForName(name):
            model = loadModel(name)
            return recommend(model)

        def getRecommendation(name):
            recomm = recommendationForName(name).to_numpy()
            ranked_array = []
            for index in range(0, n_rec):
                ranked_item = recomm[index]
                ranked_array.append({
                    'productId': ranked_item[1],
                    'score': ranked_item[2],
                    'rank': ranked_item[3]
                })
            return ranked_array

        recommendations = {}
        recommendations['popularity'] = getRecommendation(popularity)
        recommendations['cosine'] = getRecommendation(cosine)
        recommendations['pearson'] = getRecommendation(pearson)

        return recommendations

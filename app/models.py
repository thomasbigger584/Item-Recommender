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

        # Dummy for marking whether a customer bought that item or not.
        # If one buys an item, then purchase_dummy are marked as 1
        data_dummy = data.copy()
        data_dummy['purchase_dummy'] = 1

        # Normalizing the purchase count, say by each user, would not work because customers may have different buying frequency don’t have the same taste.
        # However, we can normalize items by purchase frequency across all users.
        # we will normalize their purchase history, from 0–1 (with 1 being the most number of purchase for an item and 0 being 0 purchase count for that item

        df_matrix = pd.pivot_table(
            data, values='purchase_count', index='customerId', columns='productId')
        df_matrix_norm = (df_matrix-df_matrix.min()) / \
            (df_matrix.max()-df_matrix.min())
        d = df_matrix_norm.reset_index()
        d.index.names = ['scaled_purchase_freq']
        data_norm = pd.melt(
            d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()
        """
        shape (133585, 3)
                customerId productId  scaled_purchase_freq
        9                 9         0              0.133333
        25               25         0              0.133333
        32               33         0              0.133333
        35               36         0              0.133333
        43               44         0              0.133333
        55               56         0              0.133333
        59               60         0              0.000000
        71               72         0              0.333333
        75               76         0              0.133333
        90               91         0              0.000000
        97               98         0              0.333333
        99              100         0              0.133333
        """
        # -----------------------------------------------------------------------------------------

        # Split train and test set
        # Typically, we use a larger portion of the data for training and a smaller portion for testing.
        # We use 80:20 ratio for our train-test set size.
        # Our training portion will be used to develop a predictive model, while the other to evaluate the model’s performance.

        def split_data(data):
            # split ratio is 0.2
            train, test = train_test_split(data, test_size=split_ratio)
            train_data = tc.SFrame(train)
            test_data = tc.SFrame(test)
            return train_data, test_data

        train_data, test_data = split_data(data)
        train_data_dummy, test_data_dummy = split_data(data_dummy)
        train_data_norm, test_data_norm = split_data(data_norm)

        # Now that we have three datasets with purchase counts, purchase dummy, and scaled purchase counts

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
        n_display = 30

        # The popularity model takes the most popular items for recommendation.
        # These items are products with the highest number of sells across customers.
        target = 'purchase_count'
        popularity_model = model(train_data_dummy, popularity, user_id,
                                 item_id, target, users_to_recommend, n_rec, n_display)

        # ------ todo see what this looks like, continue ith the cmparisons of models saying we now have the 3 datasets
        # print(popularity_model)
        # return ''

        # In collaborative filtering, we would recommend items based on how similar users purchase items.
        # For instance, if customer 1 and customer 2 bought similar items, e.g. 1 bought X, Y, Z and 2 bought X, Y, we would recommend an item Z to customer 2.
        target = 'purchase_count'
        cos_model = model(train_data_dummy, cosine, user_id, item_id, target,
                          users_to_recommend, n_rec, n_display)

        # Similarity is the pearson coefficient between the two vectors.
        target = 'purchase_count'
        pear_model = model(train_data_dummy, pearson, user_id, item_id,
                           target, users_to_recommend, n_rec, n_display)
        print("Execution time:", round((time.time()-s)/60, 2), "minutes")

    def recommend(self):
        popularity_model = tc.load_model(model_folder + '/' + popularity)

        users_to_recommend = list(1)
        n_rec = 10
        popularity_recomm = popularity_model.recommend(
            users=users_to_recommend, k=n_rec)

        n_display = 30
        popularity_recomm.print_rows(n_display)

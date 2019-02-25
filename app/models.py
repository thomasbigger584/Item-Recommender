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


class ItemRecommender:
    def itemRecommender(self):
        customers = pd.read_csv('app/data/recommend_1.csv')
        transactions = pd.read_csv('app/data/trx_data.csv')

        s = time.time()

        transactions['products'] = transactions['products'].apply(
            lambda x: [int(i) for i in x.split('|')])

        data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(),
                       id_vars=['customerId'],
                       value_name='products') \
            .dropna().drop(['variable'], axis=1) \
            .groupby(['customerId', 'products']) \
            .agg({'products': 'count'}) \
            .rename(columns={'products': 'purchase_count'}) \
            .reset_index() \
            .rename(columns={'products': 'productId'})
        data['productId'] = data['productId'].astype(np.int64)

        def create_data_dummy(data):
            data_dummy = data.copy()
            data_dummy['purchase_dummy'] = 1
            return data_dummy

        data_dummy = create_data_dummy(data)

        def normalize_data(data):
            df_matrix = pd.pivot_table(
                data, values='purchase_count', index='customerId', columns='productId')
            df_matrix_norm = (df_matrix-df_matrix.min()) / \
                (df_matrix.max()-df_matrix.min())
            d = df_matrix_norm.reset_index()
            d.index.names = ['scaled_purchase_freq']
            return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()

        data_norm = normalize_data(data)

        def split_data(data):
            train, test = train_test_split(data, test_size=.2)
            train_data = tc.SFrame(train)
            test_data = tc.SFrame(test)
            return train_data, test_data

        train_data_dummy, test_data_dummy = split_data(data_dummy)
        train_data_norm, test_data_norm = split_data(data_norm)

        def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
            if name == 'popularity':
                model = tc.popularity_recommender.create(train_data,
                                                         user_id=user_id,
                                                         item_id=item_id,
                                                         target=target)
            elif name == 'cosine':
                model = tc.item_similarity_recommender.create(train_data,
                                                              user_id=user_id,
                                                              item_id=item_id,
                                                              target=target,
                                                              similarity_type='cosine')
            elif name == 'pearson':
                model = tc.item_similarity_recommender.create(train_data,
                                                              user_id=user_id,
                                                              item_id=item_id,
                                                              target=target,
                                                              similarity_type='pearson')
            base_folder = 'app/trained_models'
            popularity_model.save(base_folder + '/' + name)
            return model

        # variables to define field names
        user_id = 'customerId'
        item_id = 'productId'
        target = 'purchase_count'
        users_to_recommend = list(transactions[user_id])
        n_rec = 10  # number of items to recommend
        n_display = 30

        # The popularity model takes the most popular items for recommendation. These items are products with the highest number of sells across customers.
        name = 'popularity'
        target = 'purchase_count'
        popularity_model = model(train_data_dummy, name, user_id,
                                 item_id, target, users_to_recommend, n_rec, n_display)

        # Get recommendations for a list of users to recommend (from customers file)
        # Printed below is head / top 30 rows for first 3 customers with 10 recommendations each
        popularity_recomm = popularity_model.recommend(
            users=users_to_recommend, k=n_rec)
        # popularity_recomm.print_rows(n_display)

        # In collaborative filtering, we would recommend items based on how similar users purchase items.
        # For instance, if customer 1 and customer 2 bought similar items, e.g. 1 bought X, Y, Z and 2 bought X, Y, we would recommend an item Z to customer 2.
        name = 'cosine'
        target = 'purchase_count'
        cos = model(train_data_dummy, name, user_id, item_id, target,
                    users_to_recommend, n_rec, n_display)

        cos_recomm = popularity_model.recommend(users=users_to_recommend, k=n_rec)
        cos_recomm.print_rows(n_display)

        print("Execution time:", round((time.time()-s)/60, 2), "minutes")

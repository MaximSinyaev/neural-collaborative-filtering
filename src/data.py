import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from multiprocessing import Pool, Process


from pandarallel import pandarallel
from itertools import chain

RANDOM_SEED = 42
N_THREADS = 10

pandarallel.initialize(nb_workers=N_THREADS)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class UserItemDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, neg_user_tensor, neg_item_tensor, neg_sample_size):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.neg_user_tensor = neg_user_tensor
        self.neg_item_tensor = neg_item_tensor
        self.neg_sample_size =neg_sample_size
        
    def __getitem__(self, index):
        neg_range = (index * self.neg_sample_size, (index + 1) * self.neg_sample_size)
        return self.user_tensor[index], self.item_tensor[index],\
    self.neg_user_tensor[neg_range[0]: neg_range[1]], \
    self.neg_item_tensor[neg_range[0]: neg_range[1]]
        
    def __len__(self):
        return self.user_tensor.size(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

def save_negative_items_to_file(interact_status, user_pool, item_pool, separator, data_dir, verbose=False):
    for user_id in tqdm(user_pool, disable=(not verbose)):
        interacted_items = interact_status.loc[user_id]['interacted_items']
        negative_items = item_pool - interacted_items
        path = os.path.join(data_dir, f'{user_id}.txt')
        with open(path, 'w') as f:
            f.write(separator.join((str(item) for item in negative_items)))
    
class NegativeItemSet:
    """Class for negative items, to compute them on the fly, and not to keep in memory"""
    def __init__(self, ratings, item_pool, user_pool, num_neg_samples=99,
                 data_dir='data/negative_items', verbose=False, separator=','):
        self.ratings = ratings
        self.item_pool = item_pool
        self.user_pool = user_pool
        self.verbose = verbose
        self.data_path = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        self.separator = separator
        self.num_neg_samples = num_neg_samples
        self.interact_status = self.ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
#         self.count_negative_items(num_threads=20)
        
    def count_negative_items(self, num_threads=10):
        print('Sampling negative examples')
        for batch in np.array_split(list(self.user_pool), num_threads):
            interacted_status_batch = self.interact_status.iloc[batch]
            Process(target=save_negative_items_to_file, args=(interacted_status_batch, batch, self.item_pool, self.separator, self.data_path,)).start()
#         for user_id in tqdm(self.user_pool, disable=(not self.verbose)):
#             interacted_items = self.interact_status.iloc[user_id]['interacted_items']
#             negative_items = self.item_pool - interacted_items
#             path = os.path.join(self.data_dir, f'{user_id}.txt')
#             with open(path, 'w') as f:
#                 f.write(self.separator.join((str(item) for item in negative_items)))
                
        
        
    def __getitem__(self, index):
        interact_status = self.interact_status.iloc[index]
#         path = os.path.join(data_dir, f'{user_id}.txt')
#         with open(path, 'r') as f:
#             negative_items = f.read().split(self.separator)
#             interact_status.loc['negative_samples'] = random.sample(negative_items, self.num_neg_samples)
        interact_status.loc['negative_items'] = self.item_pool - interact_status['interacted_items']
#         interact_status.loc['negative_samples'] = random.sample(interact_status['negative_items'], self.num_neg_samples)
#         return interact_status[['userId', 'negative_items', 'negative_samples']]
        return interact_status[['userId', 'negative_items']]

#     def get_negative_samples


    
    

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, verbose=False, n_users_test_split=0.1, test_batch_size=64):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.verbose = verbose
        self.test_batch_size = 64
        self.n_users_test_split = n_users_test_split
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings)
        print('Negative sampled')
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings.loc[:, 'rating'] = ratings['rating'].mask(ratings['rating'] > 0, 1.)
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        self.n_users_test_split
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test_users = random.sample(self.user_pool, int(self.n_users_test_split * len(self.user_pool)))
        test = ratings.query('userId in @test_users')
        test = test[test['rank_latest'] == 1]
        train = ratings.drop(test.index)
#         train = ratings[ratings['rank_latest'] > 1]
#         assert train['userId'].nunique() == test['userId'].nunique()
        assert (train.shape[0] + test.shape[0]) == ratings.shape[0]
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings, num_neg_samples=99):
        """return all negative items & 100 sampled negative items"""
        return NegativeItemSet(ratings, item_pool=self.item_pool, user_pool=self.user_pool, verbose=self.verbose)
        # interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
        #     columns={'itemId': 'interacted_items'})
        # interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, num_neg_items))
        # interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, num_neg_samples))
        # return interact_status[['userId', 'negative_items', 'negative_samples']]

#     def data_generator(self, batch_size=256):
        
    
    def transform_to_array(self, row, num_neg):
        user_id = int(row.name)
        positive_num = len(row.itemId)
        res_len = (1 + num_neg) * positive_num
        user = [user_id] * res_len
#         negative = 
        negative_samples = random.sample(self.negatives[user_id]['negative_items'], num_neg * positive_num)
        items = [int(row.itemId[i // (1 + num_neg)]) if (i % (1 + num_neg) == 0) else negative_samples[i - (i // (1 + num_neg) + 1)] for i in range(res_len)]
        ratings = [float(row.rating[i // (1 + num_neg)]) if i % (1 + num_neg) == 0 else 0. for i in range(res_len)]
        return user, items, ratings
    
    def get_negative_samples(self, user_id, n_neg=400):
        negatives = self.negatives[user_id]['negative_items']
        return random.sample(negatives, n_neg)
    
    def instance_a_train_loader(self, num_negatives=4, batch_size=256):
        """
        
        instance train loader for one training epoch
        
        num_negative: None or int, default=None, if None, then number of negatives == len(interacted)
        """
        users, items, ratings = [], [], []
#         train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
#         train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        train_ratings_grouped = self.train_ratings.groupby('userId').agg({'itemId': list, 'rating': list})#.parallel_apply(self.transform_to_array, axis=1, num_neg=4)
#         [(users.extend(ar[0]), items.extend(ar[1]), ratings.extend(ar[2])) for ar in temp_ar]
        [(users.extend(ar[0]), items.extend(ar[1]), ratings.extend(ar[2])) for ar in train_ratings_grouped.parallel_apply(self.transform_to_array, axis=1, num_neg=num_negatives).values]
#         for i, row in tqdm(train_ratings_grouped.iterrows(), total=train_ratings_grouped.shape[0], disable=(not self.verbose)):
#             ar = self.transform_to_array(row, num_negatives)
#             users.extend(ar[0])
#             items.extend(ar[1])
#             ratings.extend(ar[2])
#             [(users.extend(ar[0]), items.extend(ar[1]), ratings.extend(ar[2])) for ar in temp_ar]
#             users.append(int(row.userId))
#             items.append(int(row.itemId))
#             ratings.append(float(row.rating))
#             negatives = self.negatives[users[-1]]
#             for j in range(num_negatives):
#                 assert row.userId == negatives['userId'], 'Все плохо'
#                 users.append(int(row.userId))
#                 items.append(int(negatives['negative_samples'][j]))
#                 ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self, n_neg=2000, batch_size=256):
        """create evaluate data"""
#         test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
#         test_ratings_grouped = self.test_ratings.groupby('userId').agg({'itemId': list, 'rating': list})
#         [(test_users.extend(ar[0]), test_items.extend(ar[1]), ratings.extend(ar[2])) for ar in train_ratings_grouped.parallel_apply(self.transform_to_array, axis=1, num_neg=num_negatives).values]
        test_users.extend(self.test_ratings.userId.astype('int').values)
        test_items.extend(self.test_ratings.itemId.astype('int').values)
        negative_users = np.repeat(test_users, n_neg)
        negative_items = [item for user in test_users for item in self.get_negative_samples(user, n_neg)]
        
#         for i, row in self.test_ratings.iterrows():
#             test_users.append(int(row.userId))
#             test_items.append(int(row.itemId))
#             negatives = self.negatives[users[-1]]
#             for j in range(len(negatives.negative_samples)):
#                 negative_users.append(int(row.userId))
#                 negative_items.append(int(negatives.negative_samples[j]))
        dataset = UserItemDataset(torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items), neg_sample_size=n_neg)
        return DataLoader(dataset, batch_size=self.test_batch_size, shuffle=True)

    def evaluate_data_generator(self, n_neg, batch_size):
        test_ratings_grouped = self.test_ratings.groupby('userId', as_index=False).agg({'itemId': list, 'rating': list})
        for idx in range(0, test_ratings_grouped.shape[0], batch_size):
            test_users, test_items, negative_users, negative_items = [], [], [], []
            test_users.extend([int(user) for user in test_ratings_grouped.userId.iloc[idx: idx + batch_size].values])
            test_items.extend([int(item) for i in test_ratings_grouped.itemId.iloc[idx: idx + batch_size].values for item in i])
            negative_users = np.repeat(test_users, n_neg)
            negative_items = [int(item) for user in test_users for item in self.get_negative_samples(user, n_neg)]
            yield [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

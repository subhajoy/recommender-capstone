import os,sys

src_dir = os.path.dirname( \
            os.path.dirname( \
                os.path.dirname( \
                    os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Abstract.RecommenderModelAbstract import RecommenderModelAbstract

import pandas as pd
from math import sqrt

class ItemItemCollaborativeFiltering(RecommenderModelAbstract):

    def __init__(self):
        super().__init__()
        self.train_raw = pd.DataFrame()
        self.test_raw = pd.DataFrame()
        self.user_list = list()
        self.item_list = list()
        self.dict_user_rating_listed_per_item = dict()
        self.dict_item_rating_listed_per_user = dict()
        self.model = dict()
        self.predicted_df = pd.DataFrame(columns = ['user', 'pred_item_rating'])

    # ------------------------------------------------------------------------ #
    # helper functions ------------------------------------------------------- #
    # ------------------------------------------------------------------------ #
    def cosine(self, dataA, dataB):
        if len(dataA) != len(dataB):
            print("Error: the length of two input lists are not same.")
            return -1
        AB = sum([dataA[i] * dataB[i] for i in range(len(dataA))])
        normA = sqrt(sum([dataA[i] ** 2 for i in range(len(dataA))]))
        normB = sqrt(sum([dataB[i] ** 2 for i in range(len(dataB))]))
        denominator = normA * normB
        if denominator == 0:
            return 0
        return AB / denominator

    def getNearestNeighbors(self, target, d, nNeighbors = None):
        similarities = []
        for i in self.item_list:
            if i==target:
                continue
            dataA = sorted(d[target]['user_rating'], key=lambda x:x[0])
            dataB = sorted(d[i]['user_rating'], key=lambda x:x[0])
            common_users = list(set([x[0] for x in dataA]).intersection([x[0] for x in dataB]))
            if len(common_users)==0:
                similarities.append((0.0, 0))
                continue
            dataA = [x[1] for x in dataA if x[0] in common_users]
            dataB = [x[1] for x in dataB if x[0] in common_users]
            similarities.append((self.cosine(dataA, dataB),i))
        similarities.sort(reverse = True)
        if nNeighbors != None:
            similarities = similarities[0:nNeighbors]
        return similarities

    def buildModel(self, nNeighbors = 20):
        print("Model builder is running...")
        model = {}
        for item in self.item_list:
            model.setdefault(item, {})
            correlations = self.getNearestNeighbors(target=item, \
                            d=self.dict_user_rating_listed_per_item, nNeighbors=nNeighbors)
            for correlation, neighbor in correlations:
                model[item][neighbor] = correlation
        # Row normalization
        for c in model:
            COLSUM = sum([model[c][r] for r in model[c]])
            if COLSUM > 0:
                for r in model[c]:
                    model[c][r] /= COLSUM
        print("\tComplete!")
        return model

    def Recommendation(self, user, d_u, model):
        item_u, rating_u = zip(*d_u[user]['item_rating'])
        predictedScores = []
        for candidate in self.item_list:
            score_num = 0
            score_den = 0
            if candidate in item_u:
                continue
            correlations = model[candidate]
            for i in range(len(item_u)):
                score_num += rating_u[i]*correlations.get(item_u[i],0)
            #     score_den += correlations.get(item_u[i],0)
            # if(score_den!=0):
            #     score = score_num/score_den
            # else:
            #     score = 0
            predictedScores.append((candidate, score_num))
        predictedScores.sort(key = lambda x: x[1], reverse = True)
        return predictedScores
    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #

    def fetchData(self, train, test):
        self.train = train.copy()
        self.test = test.copy()

        self.user_list = pd.unique(self.train['user']).tolist()
        self.item_list = pd.unique(self.train['item']).tolist()

        df_user_rating_paired = pd.DataFrame({'item':self.train['item'], \
                           'user_rating':[i for i in zip(self.train['user'], self.train['rating'])]})
        ratings_grouped_by_items = df_user_rating_paired.groupby(df_user_rating_paired['item'], \
                            group_keys=False)
        df_user_rating_listed_per_item = pd.DataFrame(ratings_grouped_by_items['user_rating'].apply(list))

        df_item_rating_paired = pd.DataFrame({'user':self.train['user'], \
                           'item_rating':[i for i in zip(self.train['item'],self.train['rating'])]})
        ratings_grouped_by_users = df_item_rating_paired.groupby(df_item_rating_paired['user'], \
                            group_keys=False)
        df_item_rating_listed_per_user = pd.DataFrame(ratings_grouped_by_users['item_rating'].apply(list))

        self.dict_user_rating_listed_per_item = df_user_rating_listed_per_item.to_dict('index')
        self.dict_item_rating_listed_per_user = df_item_rating_listed_per_user.to_dict('index')

    def featureEngineering(self):
        pass

    def dataSampling(self):
        pass

    def dataFormatConversion(self):
        pass

    def modelFitting(self, nNeighbors):
        self.model = self.buildModel(nNeighbors)

    def doPredictions(self):
        for user in self.user_list:
            l = self.Recommendation(user, \
                    d_u=self.dict_item_rating_listed_per_user, model=self.model)
            self.predicted_df = self.predicted_df.append({'user':user, \
                    'pred_item_rating':l}, ignore_index=True)

    def getPredictions(self, topN=None):
        output_df = self.predicted_df.copy()
        if(topN!=None):
            output_df['pred_item_rating'] = output_df['pred_item_rating'].apply(lambda x: x[:topN])
        return output_df

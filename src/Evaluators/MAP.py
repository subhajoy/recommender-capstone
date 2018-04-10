import os,sys

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Abstract.EvaluatorAbstract import EvaluatorAbstract
import pandas as pd
from ml_metrics.average_precision import apk

class MAP(EvaluatorAbstract):

    def __init__(self):
        super().__init__()

        self.K = 5

        self.test = pd.DataFrame()
        self.train = pd.DataFrame()
        self.actuals = pd.DataFrame()
        self.predicted = pd.DataFrame()
        self.df_prepared = pd.DataFrame(columns=['user', 'actuals', 'predicted'])

    def fetchData(self, train, test, predicted):
        self.train = train
        self.test = test
        self.predicted = predicted.copy()

    def prepareData(self):
        item_list_in_train = self.train['item'].tolist()
        self.actuals = self.test.copy()
        self.actuals = self.actuals[self.actuals['item'].isin(item_list_in_train)]

        df_item_rating_paired = pd.DataFrame({'user':self.actuals['user'], \
                           'actuals':[i for i in zip(self.actuals['item'],self.actuals['rating'])]})
        ratings_grouped_by_users = df_item_rating_paired.groupby(df_item_rating_paired['user'], \
                            group_keys=False)
        self.actuals = pd.DataFrame(ratings_grouped_by_users['actuals'].apply(list))

        self.predicted = self.predicted.rename(columns={'user':'user', 'pred_item_rating':'predicted'}) \
                                        .set_index('user')

        self.df_prepared = self.predicted.join(self.actuals, how='inner')
        self.df_prepared['actuals'] = self.df_prepared['actuals'] \
                                                .apply(func=lambda x: [i for i,r in x])
        self.df_prepared['predicted'] = self.df_prepared['predicted'] \
                                                .apply(func=lambda x: [i for i,r in x])

    def runScoring(self, K=5):
        self.K = K

        self.df_prepared['apk'] = self.df_prepared.apply( \
                                        func=lambda row: apk(row['actuals'], row['predicted'], self.K), axis=1)

    def showResults(self):
        print("MAP@{}: {}".format(self.K, self.df_prepared['apk'].mean()))

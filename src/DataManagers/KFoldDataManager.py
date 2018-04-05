import os,sys

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Abstract.DataManagerAbstract import DataManagerAbstract
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class KFoldDataManager(DataManagerAbstract):

    def __init__(self, folds=5):
        super().__init__()
        
        self.data_dir = ''
        self.ratings_csv = ''
        self.items_json = ''

        self.folds = folds
        self.train_test_ind_list = list()

        self.df_ratings_raw = pd.DataFrame()
        self.df_items_raw = pd.DataFrame()

        self.sep = ','

    def loadPath(self):
        self.data_dir = os.path.join(os.path.dirname(src_dir),'data')
        self.raw_data_dir = os.path.join(self.data_dir,'raw')
        self.ratings_csv = os.path.join(self.raw_data_dir,'ratings.csv')
        self.items_json = os.path.join(self.raw_data_dir,'items.json')

    def loadData(self):
        self.df_ratings_raw = pd.read_csv(self.ratings_csv, sep=self.sep, header=0)
        self.df_items_raw = pd.read_json(self.items_json, lines=True)

    def trainTestSplit(self):
        skf = StratifiedKFold(n_splits=self.folds)
        for train_ind, test_ind in skf.split(self.df_ratings_raw, self.df_ratings_raw['user']):
            self.train_test_ind_list.append((train_ind, test_ind))

    def getData(self, fold_index):
        if fold_index >= self.folds:
            print("Input positional parameter (fold_index) not greater than {}(number of folds)".format(self.folds))
        train_ind, test_ind = self.train_test_ind_list[fold_index]
        return (self.df_ratings_raw.iloc[train_ind], self.df_ratings_raw.iloc[test_ind], self.df_items_raw)

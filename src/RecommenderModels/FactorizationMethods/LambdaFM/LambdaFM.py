import os,sys

src_dir = os.path.dirname( \
            os.path.dirname( \
                os.path.dirname( \
                    os.path.dirname( \
                        os.path.abspath(__file__)))))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from Abstract.RecommenderModelAbstract import RecommenderModelAbstract

import numpy as np
import pandas as pd
import subprocess

class LambdaFM(RecommenderModelAbstract):

    def __init__(self):
        super().__init__()
        self.vocab = dict()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.items = None
        self.data_dir = os.path.join(os.path.dirname(src_dir), 'data')
        self.base_dir = os.path.join(self.data_dir, 'lambdaFM')
        self.create_dir(self.base_dir)
        self.train_data = os.path.join(self.base_dir, 'train.dat')
        self.test_data = os.path.join(self.base_dir, 'test.dat')
        self.model_file = os.path.join(self.base_dir, 'model')
        self.predicted_file = os.path.join(self.base_dir, 'predicted.sco')

    # ------------------------------------------------------------------------ #
    # helper functions ------------------------------------------------------- #
    # ------------------------------------------------------------------------ #
    def create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def gen_lambdafm_data(self, df, out_fname, qid_name):
        fp_out = open(out_fname,'w')
        count = 0
        columns = df.columns.tolist()
        columns.remove('rating')

        df.sort_values(by=qid_name, axis=0, inplace=True, ascending=True)
        for i, row in df.iterrows():
            count+=1
            print_str = '{} qid:{}'.format(int(row['rating']), int(row[qid_name]))
            for col in columns:
                x = row[col]
                if((type(x)==float) or (type(x)==np.float64)):
                    x = int(x)
                print_str = print_str + ' {}_{}:1'.format(col, x)
            if (count==1):
                fp_out.write(print_str)
            else:
                fp_out.write('\n'+print_str)
        fp_out.close()

    def negative_sampling(self, train_samples, population_list, total_samples, test_samples=None):
        if(test_samples!=None):
            exiled_list = train_samples + test_samples
            sample_size = total_samples - len(test_samples)
            to_update_list = test_samples
        else:
            exiled_list = train_samples
            negative_sample_size = len(train_samples)
            to_update_list = train_samples
        population_list_updated = [i for i in population_list if i not in exiled_list]
        negative_list = np.random.choice(population_list_updated, sample_size).tolist()
        list_updated = to_update_list + negative_list
        return list_updated
    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # sampling functions ----------------------------------------------------- #
    # ------------------------------------------------------------------------ #
    def sampling_func_1(self, args):
        positive_rating_cutoff = args[0]
        total_samples = args[1]
        item_list_in_train = self.train['item'].tolist()
        self.test = self.test[self.test['item'].isin(item_list_in_train)]
        self.test = self.test[self.test['rating']>positive_rating_cutoff]
        train_item_grouped_by_users = self.train.groupby(self.train['user'], group_keys=False)
        train_item_listed_per_user = train_item_grouped_by_users.apply(lambda tdf: pd.Series(dict([[vv,tdf[vv].tolist()] for vv in tdf if vv not in ['user']])))
        test_item_grouped_by_users = self.test.groupby(self.test['user'], group_keys=False)
        test_item_listed_per_user = test_item_grouped_by_users.apply(lambda tdf: pd.Series(dict([[vv,tdf[vv].tolist()] for vv in tdf if vv not in ['user']])))
        tmp_df = train_item_listed_per_user.join(test_item_listed_per_user, \
                                lsuffix='_train', rsuffix='_test', how='inner') \
                                .reset_index()
        tmp_df['item_test'] = tmp_df.apply(lambda row: \
                                self.negative_sampling(train_samples=row['item_train'], \
                                test_samples=row['item_test'], population_list=item_list_in_train, \
                                total_samples=total_samples), axis=1)
        tmp_df = tmp_df[['user', 'item_test']]
        rows = list()
        # TODO: Extend this part to hold for multiple grouped columns
        for i, row in tmp_df.iterrows():
            for item in row['item_test']:
                rows.append([row['user'], item])
        self.test = pd.DataFrame(rows, columns=['user', 'item'])
        self.test.insert(0,'rating',[0]*self.test.shape[0])
    # ------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------ #

    def fetchData(self, train, test, items=None):
        self.train = train.copy()
        self.test = test.copy()
        if (items!=None):
            self.items = items.copy()

    def featureEngineering(self):
        pass

    def dataSampling(self, sampling_func, *args):
        sampling_func(args)

    def dataFormatConversion(self):
        self.gen_lambdafm_data(self.train, self.train_data, qid_name='user')
        self.gen_lambdafm_data(self.test, self.test_data, qid_name='user')

    def modelFitting(self, d=None):
        # lib_dir = os.path.join(os.path.abspath(__file__), 'library')
        subprocess.call('make', cwd='library/')


    def doPredictions(self):
        pass

    def getPredictions(self, topN=None):
        pass

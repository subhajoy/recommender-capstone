import os
import pandas as pd

# ------------------------------------------------------------------------------

base_dir = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
data_dir = os.path.join(base_dir,'data')
raw_data_dir = os.path.join(data_dir,'raw')
script_dir = os.path.join(base_dir,'scripts')
splitted_data_dir = os.path.join(data_dir,'splitted')

# ------------------------------------------------------------------------------

def bin_splitter(row, K):
    length = row['len']
    output = list()
    bin_endpts = [i for i in range(0,length,int(length/K))]
    if (length%K==0):
        bin_endpts.append(length)
    for i in range(K):
        bin_ = row['item_rating'][bin_endpts[i]:bin_endpts[i+1]]
        output.append(bin_)
    for i in range(bin_endpts[K],length):
        output[i-bin_endpts[K]].append(row['item_rating'][i])
    return tuple(output)

def test_length_match(row, K):
    total = row['len']
    total_from_bins = 0
    for i in range(K):
        total_from_bins += len(row['bin_{}'.format(str(i))])
    try:
        assert total == total_from_bins
    except AssertionError:
        print('Error with user {}: Given length: {} Total length from bins: {}' \
              .format(row['user'], total, total_from_bins))

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def concat_lists(row, test_bin, K):
    output_list = list()
    for i in range(K):
        if i==test_bin:
            continue
        output_list.extend(row['bin_{}'.format(str(i))])
    return output_list

# https://stackoverflow.com/questions/38428796/how-to-do-lateral-view-explode-in-pandas
def explode_and_unzip(df):
    rows = list()
    for i, row in df.iterrows():
        for tup in row['item_rating']:
            rows.append([row['user'], tup])
    out_df = pd.DataFrame(rows, columns=df.columns)
    out_df['item'], out_df['rating'] = zip(*out_df['item_rating'])
    out_df = out_df.drop('item_rating', axis=1)
    return out_df

# ------------------------------------------------------------------------------
def create_split(K):
    create_dir(splitted_data_dir)

    df_ratings_raw = pd.read_csv(os.path.join(raw_data_dir,'ratings.csv'), sep=',', header=0)
    df_ratings = pd.DataFrame({'user':df_ratings_raw['user'], \
                               'item_rating':[i for i in zip(df_ratings_raw['item'],df_ratings_raw['rating'])]})
    ratings_grp_by_users = df_ratings.groupby(df_ratings['user'], group_keys=False)
    df_ratings_listed_per_user = pd.DataFrame(ratings_grp_by_users['item_rating'].apply(list)) \
                                .reset_index()
    # faster ways to apply the list is mentioned in this link:
    # https://stackoverflow.com/questions/22219004/grouping-rows-in-list-in-pandas-groupby/22221675
    df_ratings_listed_per_user['len'] = df_ratings_listed_per_user.apply(lambda row: len(row['item_rating']), axis=1)

    bins = [i for i in zip(*df_ratings_listed_per_user.apply(func=bin_splitter, K=K, axis=1))]

    df_ratings_listed_per_user_binned = df_ratings_listed_per_user[['user','len']]

    for i in range(K):
        col_name = 'bin_{}'.format(str(i))
        df_ratings_listed_per_user_binned[col_name] = bins[i]

    df_ratings_listed_per_user_binned.apply(func=test_length_match, K=K, axis=1)

    for i in range(K):
        bin_dir = os.path.join(splitted_data_dir,str(i))
        create_dir(bin_dir)
        df_train_dense = pd.DataFrame(df_ratings_listed_per_user_binned['user']).copy()
        df_train_dense['item_rating'] = df_ratings_listed_per_user_binned.apply(concat_lists, K=K, test_bin=i, axis=1)
        df_test_dense =  pd.DataFrame(df_ratings_listed_per_user_binned[['user','bin_{}'.format(str(i))]]).copy()
        df_test_dense = df_test_dense.rename(columns={'user':'user', 'bin_{}'.format(str(i)):'item_rating'})

        df_train = explode_and_unzip(df_train_dense)
        df_test = explode_and_unzip(df_test_dense)

        try:
            assert ((df_train.shape[0]+df_test.shape[0])==df_ratings_raw.shape[0])
        except AssertionError:
            print('Error!\n\nDataFrame:\t#rows:\ndf_train\t{}\ndf_test\t{}\ndf_ratings_raw\t{}\n' \
                                        .format(df_train.shape[0],df_test.shape[0],df_ratings_raw.shape[0]))

        df_train.to_csv(os.path.join(bin_dir,'train.csv'), index=False, sep=',')
        df_test.to_csv(os.path.join(bin_dir,'test.csv'), index=False, sep=',')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create a K-fold train-test split")
    parser.add_argument('-K', action='store', default=5, \
                        help='Define K, the number of folds, since 5 is the minimum number of ratings per user, \
                                it is recommended to keep K less than 5 (default: 5)')
    args = parser.parse_args()
    create_split(int(args.K))

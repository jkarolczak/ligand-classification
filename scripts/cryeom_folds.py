from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.model_selection._split import StratifiedGroupKFold


def sanity_check(train_csvs, test_csvs, cryeom):
    all_instances_test = []
    for train, test in zip(train_csvs, test_csvs):
        print(f'Train size: {len(train)}, test size: {len(test)}')
        all_instances_test.extend(list(test['blob_map_filename']))
        instances_overlap = pd.merge(train[['blob_map_filename']], test[['blob_map_filename']], on='blob_map_filename')
        print(f'Instance overlap: {len(instances_overlap)}')
        for name in ['group', 'ligand']:
            cryoem_unique = cryeom[name].unique()
            train_groups = train[name].unique()
            test_groups = test[name].unique()
            overlap = np.intersect1d(train_groups, test_groups)
            in_test_not_in_train = np.setdiff1d(test_groups, train_groups)
            print(f'{name} overlap: {len(overlap)}, {len(cryoem_unique)}')
            if name == 'ligand':
                print(f'Classes in test but not in train: {len(in_test_not_in_train)} {in_test_not_in_train}')
        print('------------------------------------------')
    all_instances_test = pd.DataFrame(all_instances_test, columns=['blob_map_filename'])
    instances_overlap = pd.merge(cryoem[['blob_map_filename']], all_instances_test[['blob_map_filename']], on='blob_map_filename')
    print(f'Test overlap: {len(instances_overlap)}, {len(cryoem)}')

class CustomGreedyKFold:
    def __init__(self, k, shuffle, random_state):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def greedy_kfold(self, df):
        if self.shuffle:
            df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        folds = [[] for i in range(self.k)]
        folds_instances = []
        X = df['blob_map_filename'].to_numpy()
        y = df['ligand'].to_numpy()
        groups = df['group'].to_numpy()
        groups_class = df[['ligand', 'group']]
        groups_class = groups_class.groupby(['ligand', 'group']).size().to_frame('count').reset_index(drop=False)

        for c in np.unique(y):
            bin_sizes = [np.sum(y == c) // self.k for _ in range(self.k)]
            for i in range(np.sum(y == c) % self.k):
                bin_sizes[i] += 1
            bins = [[] for _ in range(self.k)]
            group_class = groups_class[groups_class['ligand'] == c]
            groups_desc = group_class.sort_values(by='count', ascending=False)[['group', 'count']]
            for i in range(len(groups_desc)):
                current_size = groups_desc.iloc[i]['count']
                current_group = groups_desc.iloc[i]['group']
                instances_idx = np.flatnonzero((y == c) & (groups == current_group))
                remaining = current_size
                while remaining > 0:
                    max_bin_size_id = np.argmax(bin_sizes)
                    max_bin_size = bin_sizes[max_bin_size_id]
                    if max_bin_size >= remaining:
                        bins[max_bin_size_id].extend(instances_idx)
                        bin_sizes[max_bin_size_id] -= remaining
                        remaining = 0
                    else:
                        bins[max_bin_size_id].extend(instances_idx[:max_bin_size])
                        instances_idx = instances_idx[max_bin_size:]
                        bin_sizes[max_bin_size_id] -= max_bin_size
                        remaining -= max_bin_size
            for i in range(len(bins)):
                folds[i].extend(bins[i])
        for f in folds:
            fold = df.iloc[f]
            folds_instances.append(fold)
        return folds_instances

    def combine_folds(self, folds):
        train_csvs, test_csvs = [], []
        for i in range(len(folds)):
            test_csvs.append(folds[i])
            rest = list(range(len(folds)))
            rest.remove(i)
            rest_folds = [folds[j] for j in rest]
            rest_folds = pd.concat(rest_folds)
            train_csvs.append(rest_folds)
        return train_csvs, test_csvs

def kfold(strategy, df, k=3, remove_too_small=False):
    print(len(df))
    if remove_too_small:
        groups_class = df[['ligand', 'group']].groupby(['ligand', 'group']).size().to_frame('count').reset_index(drop=False)
        too_small = groups_class[groups_class['count'] < k]
        for i in range(len(too_small)):
            c = too_small['ligand'].iloc[i]
            group = too_small['group'].iloc[i]
            df = df[(df['ligand'] != c) | (df['group'] != group)]
    print(len(df))
    if strategy == 'sklearn':
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=23)

        X = df['blob_map_filename'].to_numpy()
        y = df['ligand'].to_numpy()
        groups = df['group'].to_numpy()

        train_csvs, test_csvs = [], []
        for i, (train_idx, test_idx) in enumerate(sgkf.split(X=X, y=y, groups=groups)):
            train_csv = df.iloc[train_idx]
            test_csv = df.iloc[test_idx]
            train_csvs.append(train_csv)
            test_csvs.append(test_csv)
    else:
        kf = CustomGreedyKFold(k=k, shuffle=True, random_state=23)
        folds_instances = kf.greedy_kfold(df)
        train_csvs, test_csvs = kf.combine_folds(folds_instances)
    sanity_check(train_csvs, test_csvs, df)
    return train_csvs, test_csvs

def save_kfolds(dfs, main_path):
    for i, df in enumerate(dfs):
        df[['blob_map_filename']].to_csv(f'{main_path}/fold{i}.csv', index=False)

if __name__ == '__main__':
    path_cryoem = 'cryoem_q0.6_blob_labels.csv'
    path_ligands = 'cmb_data.csv'
    strategy = 'custom'
    k = 3
    remove_too_small = False
    folds_save_path = 'cryoem_custom'

    cryoem = pd.read_csv(path_cryoem, sep=',')
    ligand_mapping = pd.read_csv(path_ligands, sep=',')
    cryoem = cryoem[cryoem['ligand'].isin(ligand_mapping['ligand'])]
    cryoem['group'] = cryoem['blob_map_filename'].apply(lambda x: x.split('_')[0])

    print(cryoem.head())
    print(f"Number of classes: {len(cryoem['ligand'].unique())}")
    print('---------------------------')

    train_csvs, test_csvs = kfold(strategy, cryoem, k=k, remove_too_small=remove_too_small)
    save_kfolds(test_csvs, folds_save_path)

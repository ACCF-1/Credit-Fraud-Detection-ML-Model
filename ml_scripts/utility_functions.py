import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sklearn.impute as imp
from sklearn.preprocessing import KBinsDiscretizer
import networkx
from networkx.algorithms.components.connected import connected_components
from IPython.display import display

import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'ml_scripts'))
import config as cfg

import inspect

'''general utility functions'''

'''------------------------------------------EDA functions------------------------------------------'''
def prt_stats(dataset, col_name):
    print(np.mean(dataset[col_name]))
    print(np.median(dataset[col_name]))
    print(dataset[col_name].skew())
    print(dataset[col_name].kurtosis(), '\n')

def cat_visual(dataset, col_name):
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    sns.countplot(data=dataset, y=col_name)
    plt.subplot(1,2,2)
    dataset[col_name].values_counts(normalize=True).plot.bar(rot=25)
    plt.ylabel(col_name)    
    plt.xlabel('% ' + 'distribution per category')
    plt.tight_layout()
    plt.show()

def num_visual(dataset, col_name):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.kdeplot(dataset[col_name], color='g', shade=True)
    plt.subplot(1,2,2)
    sns.boxplot(dataset[col_name])
    plt.show()
    prt_stats(dataset, col_name)

def chk_missing(col): #FIXME
    if col.isin(['', None, pd.NaT, np.nan]) == True:
        pass

def imbalance_visual(dataset, tgt_feat=cfg.target, imbalance_tolerance:int=5):
    cnts = dataset[tgt_feat].value_counts()
    cnts.index = cnts.index.astype(str)
    one_class_cnt = cnts.loc['1']
    zero_class_cnt = cnts.loc['0']
    imbalance_ratio = round(zero_class_cnt/one_class_cnt, 4)
    print('major class: ', zero_class_cnt, '\n',
          'minority class: ', one_class_cnt, '\n',
          'imbalance ratio: ', imbalance_ratio, '\n')
    plt.bar(cnts.index, cnts)
    if imbalance_ratio > imbalance_tolerance:
        print('The imbalance level of the dataset requires special care.')
    plt.show()

def data_overview(raw_data_df:pd.DataFrame, tgt_col:str, phase:str='training', print_output:bool=True):
    if phase == 'training':
        if len(raw_data_df.columns.values) > 100:
            print('Please be aware of the curse of dimensionality (No. of features > 100)\n')

        data_overview = {
            'col names': raw_data_df.columns.values,
            'shape': raw_data_df.shape,
            'top rows': raw_data_df.head(),
            'bot rows': raw_data_df.tail(),
            'type': raw_data_df.dtypes.to_frame('Data Type'),
            'info': raw_data_df.describe(include='all').T,
            'class_weight': len(raw_data_df[tgt_col])/min(raw_data_df[tgt_col].value_counts()),
            'class_ratio': raw_data_df[tgt_col].value_counts(normalize=True).to_frame('Percentage'),
            'class_distribution': raw_data_df[tgt_col].value_counts().to_frame('Counts'),
        }

        for key, val in data_overview.items():
            print(f'\n{key}')
            display(val)

        imbalance_visual(raw_data_df)

        print('duplicated values:\n', raw_data_df.duplicated().any(), '\n')
        print('unique values:\n', )
        display(raw_data_df.nunique().to_frame('Unique Values Count'))

        if raw_data_df.isna().sum().any() == True:
            print('null values:\n')
            display(raw_data_df.isna().sum()[raw_data_df.isna().sum() > 0].to_frame('Missing Values'))
        else:
            print('There is no null value in this dataset\n')
        return data_overview

def detect_outlier(dataset:pd.DataFrame, tgt_col, feats:list, threshold:int =2, saves_path:str=None):
    if saves_path == None:
        saves_path == os.path.dirname(os.getcwd()) if 'ml_scripts' in os.getcwd() else os.getcwd()
    df = dataset.copy()
    df[cfg.target] = tgt_col
    fig, axs = plt.subplots(len(feats), 1, figsize=(8, 4*len(feats)))
    for idx, feat in enumerate(feats):
        if np.issubdtype(df[feat].dtypes, np.number) == True:
            sns.boxplot(x=df[feat], color='darkblue', ax=axs[idx])
            axs[idx].set_title('Boxplot - ' + df[feat].name)
    fig.tight_layout()
    fig.show()
    fig.savefig(saves_path + '/' + 'outlier chk - boxplot' + '.pdf')

def feat_vs_tgt_chk(df:pd.DataFrame, tgt_col, feats:list, threshold:float, status:str, graph_type:str, saves_path:str):
    if saves_path == None:
        saves_path == os.path.dirname(os.getcwd()) if 'ml_scripts' in os.getcwd() else os.getcwd()
    dataset = df.copy()
    dataset[cfg.target] = tgt_col.copy()
    fig, axs = plt.subplots(len(feats), 1, figsize=(8, 4*len(feats)))
    for idx, feat in enumerate(feats):
        graph_data = dataset[[feat, cfg.target]].groupby([feat]).mean().sort_values(by=feat, ascending=False).reset_index()
        graph_data['Abnormal'] = graph_data[cfg.target] > threshold
        try:
            if graph_type == 'barplot':
                sns.barplot(data=graph_data, x=feat, y=cfg.target, width=0.7, hue='Abnormal',
                            palette={True:'red', False:'navy'}, ax=axs[idx])
                axs[idx].set_title(feat)
                axs[idx].set_xticklabels(graph_data[feat].sort_values(ascending=True), rotation=90)
            elif graph_type == 'pointplot':
                sns.pointplot(data=graph_data, x=feat, y=cfg.target, palette='deep', ax=axs[idx])
                axs[idx].set_title(feat)
                axs[idx].set_xticklabels(graph_data[feat].sort_valuues(ascending=True), rotation=90)
        except:
            print(dataset[[feat, cfg.target]].groupby([feat]).mean().sort_values(by=feat, ascending=False))
            return
    fig.tight_layout()
    fig.show()
    fig.savefig(saves_path + '/' + inspect.stack()[0][3] + ' - ' + status + ', ' + graph_type + '.pdf')

def non_norm_identifier(dataset:pd.DataFrame) ->list:
    non_norm_cols = []
    other_cols = set()
    binary_cat_cols = []
    try:
        for col in dataset.columns:
            if dataset[col].min() < 0 or dataset[col].max() > 1:
                non_norm_cols.append(col)
    except:
        other_cols.add(col)

    try:
        for col in dataset.columns:
            if all(dataset[col].unique()==[0,1])==True or all(dataset[col].unique()==0) or all(dataset[col].unique()==1):
                non_norm_cols.append(col)
    except:
        other_cols.add(col)
    return non_norm_cols, binary_cat_cols, list(other_cols)



'''------------------------------------------ETL functions------------------------------------------'''
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pyodbc

def replace_null_like_values(df):
    # Define placeholders to treat as null
    null_like_values = ['NA', 'N/A', 'null', 'NULL', '-', '', ' ', '--', 'na', 'NaN', 'missing', 'Missing']
    
    # First, replace non-string nulls (None, NaN) - already handled by pandas
    df = df.copy()
    
    # Replace string-based placeholders in object columns only
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace(null_like_values, np.nan)
    
    return df

def connect_database(
  data_pool_server,
  data_pool_database,
  driver="Driver={SQL Server Native Client 11.0}",
  connection="Trusted_Connection=yes"      
):
    return pyodbc.connect(f'{driver};{data_pool_server};{data_pool_database};{connection};')

class CustomColumnTransformer(ColumnTransformer):
    def transform(self, X):
        # Get the transformed data
        transformed_data = super.transform(X)
        # Get the feature names
        feature_names = self.get_feature_names_out()
        # Create a Dataframe with the transformed data and feature names
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        return transformed_df

    def fit_transform(self, X, y=None):
        # Get the transformed data
        transformed_data = super.fit_transform(X, y)
        # Get the feature names
        feature_names = self.get_feature_names_out()
        # Create a Dataframe with the transformed data and feature names
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        return transformed_df

class CustomFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self._columns = None
        self.func = func
        #super().__init__(func=func, validate=validate)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = self.func(X)
        self._columns = df.columns.values
        return df
    
    def fit_transform(self, X: np.ndarray | pd.DataFrame | spmatrix, y: None | np.ndarray | pd.DataFrame | spmatrix):
        df = self.func(X)
        self._columns = df.columns.values
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self._columns
        #return super().get_feature_names_out(input_features)

def split_num_and_cat(dataset:pd.DataFrame, to_return:int):
    non_norm_cols, binary_cat_cols, _ = non_norm_identifier(dataset)

    cat_in_non_norm_cols = [col for col in non_norm_cols if "|" not in col and dataset[col].dtype == int]
    num_in_non_norm_cols = [col for col in non_norm_cols if col not in cat_in_non_norm_cols]

    cat_cols = binary_cat_cols + cat_in_non_norm_cols
    num_cols = [col for col in dataset.columns if col not in cat_cols]

    if to_return == 1:
        return cat_in_non_norm_cols
    elif to_return == 2:
        return num_in_non_norm_cols
    elif to_return == 3:
        return cat_cols
    elif to_return == 4:
        return [col for col in num_cols if col not in num_in_non_norm_cols]
    elif to_return == 5:
        return binary_cat_cols
    elif to_return == 6:
        return non_norm_cols

def get_idx_for_col_trans(feat_names, col_type:str):
    feat_idx_mapping = []
    for i, feat_name in enumerate(feat_names):
        if col_type == 'num':
            if 'scaling__' in feat_name or 'age trans__' in feat_name:
                feat_idx_mapping.append(i)
        elif col_type == 'cat':
            if 'cat trans__' in feat_name or 'remainder__' in feat_name:
                feat_idx_mapping.append(i)
    return feat_idx_mapping

def get_feature_correlation(
    df:pd.DataFrame, 
    top_n:int=None, 
    corr_method:str='spearman', 
    remove_duplicates:bool=True, 
    remove_self_correlations:bool=True
)->pd.DataFrame:
    '''
    Compute the feature correlation and sort feature pairs based on their correlation

    :param df: Dataframe with predictor variables
    :param top_n: Top N feature pairs to be reported (if None, return all pairs)
    :param corr_method: Correlation computation method
    :param remove_duplicates: whether duplicate features must be removed
    :param remove_self_correlations: whether self correlation will be removed

    :return: DataFrame
    '''
    corr_matrix_abs = df.corr(method=corr_method).abs
    corr_matrix_abs_us = corr_matrix_abs.unstack()
    sorted_corr_feats = corr_matrix_abs_us.sort_values(kind='quicksort', ascending=False).reset_index()

    if remove_self_correlations:
        sorted_corr_feats = sorted_corr_feats[(sorted_corr_feats.level_0 != sorted_corr_feats.level_1)]

    if remove_duplicates:
        sorted_corr_feats = sorted_corr_feats.iloc[:-2:2]

    sorted_corr_feats.columns = ['Feature 1', 'Feature 2', 'Correlation (abs)']
    sorted_corr_feats = sorted_corr_feats[~sorted_corr_feats.apply(frozenset, axis=1).duplicated()]
    if top_n:
        return sorted_corr_feats[:top_n]

    return sorted_corr_feats

def cal_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

from sklearn.base import BaseEstimator, TransformerMixin

class CorrFilterTransformer(BaseEstimator, TransformerMixin):
    '''
    Transformer that can only be applied on after split dataset
    '''
    def __init__(self):
        self._columns = None
        self._corr_threshold = 0.85
        self._saves_path = os.path.dirname(os.getcwd() if 'ml_scripts' in os.getcwd() else os.getcwd())

    @property
    def columns_(self):
        if self._columns is None:
            raise Exception('CorrFilterTransformer has not been fitted yet')
        return self._columns

    def fit(self, X, y=None):
        corr_df = get_feature_correlation(X)
        corr_df.to_csv(os.path.join(self._saves_path, 
                                    'saves', 
                                    cfg.target + ' - feature correlation.csv'), 
                        index=False)

        # array of high corr features
        high_corr_feats_arr = corr_df[corr_df['Correlation (abs)'] > self._corr_threshold][['Feature 1', 'Feature 2']].values.flatten()

        # df that True if feature is high corr
        without_high_corr_feats_df = corr_df[corr_df['Correlation (abs)'] <= self._corr_threshold][['Feature 1', 'Feature 2']].isin(high_corr_feats_arr)

        # array without high corr features
        without_high_corr_feats_arr = corr_df[~without_high_corr_feats_df].values.flatten()
        without_high_corr_feats_arr = without_high_corr_feats_arr[~(pd.isnull(without_high_corr_feats_arr))]
        # remove duplicate
        without_high_corr_feats_arr = list(set(without_high_corr_feats_arr))

        # get single feature from a group of corr features
        high_corr_feats_grp = self.to_graph(corr_df[corr_df['Correlation (abs)'] > self._corr_threshold][['Feature 1', 'Feature 2']].values)
        high_corr_feats_grp = list(connected_components(high_corr_feats_grp))
        high_corr_cleansed = [list(ele)[0] for ele in high_corr_feats_grp]
        pd.DataFrame(high_corr_feats_grp).to_csv(os.path.join(self._saves_path,
                                                              'saves',
                                                              cfg.target + ' - high feature correlation.csv'),
                                                index=False)
        self._columns = high_corr_cleansed + without_high_corr_feats_arr
        return self

    def transform(self, X, y=None):
        X = X[self._columns]
        return X

    def fit_transform(self, X:pd.DataFrame, y=None, **fit_params):
        self.fit(X)
        X = X[self._columns]
        return X

    def to_graph(self, arr):
        G = networkx.Graph()
        for part in arr:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges
            G.add_edges_from(self.to_edges(part))
        return G

    @staticmethod
    def to_edges(arr):
        '''
        treat 'arr as a Graph and return it's edges
        '''
        it = iter(arr)
        last = next(it)

        for current in it:
            yield last, current
            last = current

    def get_feature_names_out(self, input_features=None):
        return self._columns    
    
class VarFilterTransformer(BaseEstimator, TransformerMixin): #FIXME inherit from VarianceThreshold instead?
    '''
    Transformer that can only be applied on dataset after split
    '''
    def fit(self, X, y=None):
        self.threshold = 0.001
        self.var_transformer = VarianceThreshold(threshold=self.threshold)
        self.var_transformer = self.var_transformer.fit(X)
        self._columns = self.var_transformer.get_feature_names_out
        return self

    def transform(self, X, y=None):
        # Get the transformed data
        transformed_data = self.var_transformer.transform(X)
        # Get the feature data
        feature_names = self.get_feature_names_out()
        # Create a Dataframe with the transformed data and feature names
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        return transformed_df

    def fit_transform(self, X, y=None):
        self.threshold = 0.001
        self.var_transformer = VarianceThreshold(threshold=self.threshold)
        self.var_transformer = self.var_transformer.fit(X)
        self._columns = self.var_transformer.get_feature_names_out

        # Get the transformed data
        transformed_data = self.var_transformer.transform(X)
        # Get the feature data
        feature_names = self.get_feature_names_out()
        # Create a Dataframe with the transformed data and feature names
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        return transformed_df
        
    def get_feature_names_out(self):
        return self._columns

class NullRatioFilterTransformer(BaseEstimator, TransformerMixin):
    '''
    Transformer that can only be applied on dataset after split
    '''
    def __init__(self):
        self._columns = None
    
    @property
    def columns(self):
        if self._columns is None:
            raise Exception('NullRatioFilterTransformer has not been fitted yet')
    
    def fit(self, X, y=None):
        df = X.dropna(thresh=int(cfg.null_threshold * len(X)), axis='columns')
        self._columns = df.columns.values
        return self

    def transform(self, X, y=None):
        X = X[self._columns]
        return self

    def fit_transform(self, X, y=None):
        df = X.dropna(thresh=int(cfg.null_threshold * len(X)), axis='columns')
        self._columns = df.columns.values
        X = X[self._columns]
        return X

    def get_feature_names_out(self):
        return self._columns

class BinTransformer(BaseEstimator, TransformerMixin):
    '''
    Transformer that can only be applied on after split dataset
    '''
    def __init__(self, bin_cols_info:dict):
        self.bin_cols_info = bin_cols_info
        self.bin_2 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform', subsample=None)
        self.bin_3 = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None)
        self.bin_5 = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)
        self._columns = None
        self.bin_edge = {}

    def fit(self, X, y=None):
        self.bin_2.fit(X[self.bin_cols_info['bin_2']])
        self.bin_3.fit(X[self.bin_cols_info['bin_3']])
        self.bin_5.fit(X[self.bin_cols_info['bin_5']])
        self._columns = X.columns.values #FIXME
        self.bin_edge = {'bin 2': self.bin_2.bin_edges_,
                         'bin 3': self.bin_3.bin_edges_,
                         'bin 5': self.bin_5.bin_edges_,}
        return self

    def transform(self, X, y=None):
        transformed_df = X.copy()
        transformed_df[self.bin_cols_info['bin_2']] = self.bin_2.transform(transformed_df[self.bin_cols_info['bin_2']])
        transformed_df[self.bin_cols_info['bin_3']] = self.bin_3.transform(transformed_df[self.bin_cols_info['bin_3']])
        transformed_df[self.bin_cols_info['bin_5']] = self.bin_5.transform(transformed_df[self.bin_cols_info['bin_5']])
        self._columns = transformed_df.columns.values
        return transformed_df
    
    def fit_transform(self, X, y=None):
        self.bin_2.fit(X[self.bin_cols_info['bin_2']])
        self.bin_3.fit(X[self.bin_cols_info['bin_3']])
        self.bin_5.fit(X[self.bin_cols_info['bin_5']])
        transformed_df = X.copy()
        transformed_df[self.bin_cols_info['bin_2']] = self.bin_2.transform(transformed_df[self.bin_cols_info['bin_2']])
        transformed_df[self.bin_cols_info['bin_3']] = self.bin_3.transform(transformed_df[self.bin_cols_info['bin_3']])
        transformed_df[self.bin_cols_info['bin_5']] = self.bin_5.transform(transformed_df[self.bin_cols_info['bin_5']])
        self._columns = transformed_df.columns.values
        return transformed_df

    def get_feature_names_out(self, input_features=None):
        return self._columns


class ImputeTransformer(BaseEstimator, TransformerMixin):
    '''
    Transformer that can only be spplied on after split dataset
    '''
    def __init__(self, strategy:str):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'KNN':
            impute = imp.KNNImputer(n_neighbors=2)
        else:
            impute = imp.SimpleImputer(strategy=self.strategy)
        self.impute = impute.fit(X)
        self._columns = self.impute.get_feature_names_out()
        return self
    
    def transform(self, X, y=None):
        transformed_data = self.impute.transform(X)
        features_names = self.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=features_names)
        return transformed_df
    
    def fit_transform(self, X, y=None):
        if self.strategy == 'KNN':
            impute = imp.KNNImputer(n_neighbors=2)
        else:
            impute = imp.SimpleImputer(strategy=self.strategy)
        self.impute = impute.fit(X)
        self._columns = self.impute.get_feature_names_out()

        transformed_data = self.impute.transform(X)
        features_names = self.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_data, columns=features_names)
        return transformed_df
    
    def get_feature_names_out(self, input_features=None):
        return self._columns


'''------------------------------------------ML functions------------------------------------------'''
# Result evaluation functions
def param_combinations(CV_model, mdl_param_grid:dict, param_to_chk:list, saves_path=None):
    print("Fitting process tracker")
    means = CV_model.cv_results_['mean_test_score']
    stds = CV_model.cv_results_['std_test_score']
    params = CV_model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with %r' % (mean, stdev, param))

    mapped_param_to_chk = []
    for idx, to_chk in enumerate(param_to_chk):
        for key in mdl_param_grid.keys():
            if to_chk in key:
                mapped_param_to_chk.append(key)
    '''            
    if saves_path == None:
        saves_path == os.path.dirname(os.getcwd()) if 'ml_scripts' in os.getcwd() else os.getcwd()
    scores = np.array(means).reshape(len(mdl_param_grid[param_to_chk[0]]), len(mdl_param_grid[param_to_chk[1]]))
    for i, val in enumerate(mdl_param_grid[param_to_chk[0]]):
        plt.plot(mdl_param_grid[param_to_chk[1]], scores[i], label=param_to_chk[0] + ': ' + str(val))
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('F2 score')
    plt.savefig(saves_path + '/' + param_to_chk[0] + '_vs_' + param_to_chk[1] + '.png')'''

# Plot no skill and model precision-recall curves
from sklearn.metrics import precision_recall_curve
def plot_pr_curve(test_tgt, model_name:str, naive_probs, model_probs, saves_path=None):
    # determine save path
    if saves_path == None:
        saves_path == os.path.dirname(os.getcwd()) if 'ml_scripts' in os.getcwd() else os.getcwd()

    # plot model precision-recall curve
    bl_precision, bl_recall, _ = precision_recall_curve(test_tgt, naive_probs)
    plt.plot(bl_recall, bl_precision, marker="_", label='No Skill')

    # calculate the no skill line as the proportion of the positive class
    mdl_precision, mdl_recall, _ = precision_recall_curve(test_tgt, model_probs)
    plt.plot(mdl_recall, mdl_precision, marker='.', label=model_name)

    # show the plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(saves_path, 'results', 'PR AUC analysis.png') )
    plt.show()
    

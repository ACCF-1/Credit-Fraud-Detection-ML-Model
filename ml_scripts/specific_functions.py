# In[0] Libraries
'''import necessary libraries'''
# data
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# ML
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.pipeline as skl_pipe
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer

# params
import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
import config as cfg

# others
import utility_functions as uf


# In[1] FunctionTransformer: Convert data type
'''FunctionTransformer: Convert data type'''
def convert_data_type(
        dataset: pd.DataFrame,
        data_schema: pd.DataFrame,
):
    # Define mapping of schema data types to pandas types
    dtype_mapping = {
        "string": "string",
        "int": "Int64",
        "float": "float64",
        "boolean": "boolean",
        "datetime": "datetime64[ns]"
    }
    schema_dict = data_schema.set_index("column_name")["data_type"].to_dict()

    # Convert data types
    for column, dtype in schema_dict.items():
        if dtype in dtype_mapping:
            dataset[column] = dataset[column].astype(dtype_mapping[dtype])
        else:
            print(f"Warning: Unknown data type '{dtype}' for column '{column}'")
    return dataset

data_type_prep = FunctionTransformer(convert_data_type, kw_args={"data_schema": None})  # Replace None with actual data schema DataFrame if available   


# In[2] FunctionTransformer: Reduce dimensionality with domain knowledge
'''FunctionTransformer: Reduce dimensionality with domain knowledge'''
def drop_pointless_feats(dataset, irrelevant_cols:list):
    '''
    Transfomrer that can be applied on whole or after split dataset
    '''
    feats_to_keep = [col for col in dataset if col not in irrelevant_cols]
    dataset = dataset[feats_to_keep]
    return dataset

domain_drop_prep = FunctionTransformer(drop_pointless_feats)


# In[3] ColumnTransformer: Binning, scaling, and encoding
'''ColumnTransformer: Normalization & Standardization - binning, scaling, and encoding'''
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler


def get_bin_scale_encode_col_transformer(cols_to_bin: dict):
    """ Create a ColumnTransformer that applies different transformations to numerical and categorical features.
    Args:
        cols_to_bin (dict): Dictionary with keys as binning types and values as lists of column names to be binned.
    Returns:
        col_trans: ColumnTransformer object with the specified transformations.
    """
    trans_to_cat_feats = [inner for outer in list(cols_to_bin.values()) for inner in outer]

    # pipelines
    num_pipe = skl_pipe.Pipeline(steps=[('impute', 'pending'),
                                        ('data type convert', data_type_prep),
                                        ('standardize', StandardScaler()),
                                        ('normalize', MinMaxScaler())])
    cat_pipe = skl_pipe.Pipeline(steps=[('impute', 'pending'),
                                        ('data type convert', data_type_prep),
                                        ('encoding', OneHotEncoder(handle_unknown='ignore', drop='if_binary')),])
    to_cat_pipe = skl_pipe.Pipeline(steps=[('bin custom', uf.BinTransformer(cols_to_bin)),
                                        ('cat trans', cat_pipe)])

    col_trans = uf.CustomColumnTransformer(transformers=[('num trans', num_pipe, lambda X: get_cols_for_cleansing(X.columns, 'num')),
                                                        ('cat trans', cat_pipe, lambda X: get_cols_for_cleansing(X.columns, 'cat')),
                                                        ('other trans', to_cat_pipe, trans_to_cat_feats)],
                                                        remainder='passthrough')
    return col_trans

cols_to_bin = pd.read_csv(
    os.path.join(os.path.dirname(os.getcwd()), 'configurations', 'columns_to_bin.csv')
).groupby('bin')['column'].apply(list).to_dict()

col_trans = get_bin_scale_encode_col_transformer(cols_to_bin)


# In[4] Statistic filtering - correlation & variance
'''Statistic filtering - correlation & variance'''
from scipy.stats import chi2_contingency

def num_corr_filtering(df, threshold=0.9):
    df = pd.df.corr().abs()
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [col for col in upper.columns if any (upper[col] > threshold)]
    df_filtered = df.drop(to_drop, axis=1)
    return df_filtered

num_corr_filter = uf.CustomFunctionTransformer(func=num_corr_filtering)

def drop_high_corr_cat_feats(df, threshold=0.05):
    df = pd.DataFrame(data=df)
    # get list of categorical columns
        # categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [col for col in df.columns]
    columns_to_remove = []
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            # create contingency table for feature pair
            contingency_table = pd.crosstab(df[cat_cols[i]], df[cat_cols[j]])
            # Calculate chi-square stats, p-value, dof, and expected freq
            chi2, p, _, expected = chi2_contingency(contingency_table)
            # check if p-value is below the threshold
            if p < threshold:
                # determine features to be removed based on chi-square value
                if chi2 >= expected.mean():
                    columns_to_remove.append(cat_cols[i])
                else:
                    columns_to_remove.append(cat_cols[j])
    # remove highly correlated categorical features from the dataframe
    df = df.drop(columns_to_remove, axis=1)
    return df

cat_corr_filter = uf.CustomFunctionTransformer(func=drop_high_corr_cat_feats)


# In[5] Num Cat splitter for cleansing
'''Split numerical and categorical variables for cleansing'''
def get_cols_for_cleansing(feat_names, col_type:str, exception=None, trans_to_cat_feats:dict={}, return_idx=False):
    feat_mapping = []
    trans_to_cat_feats = [inner for outer in list(cols_to_bin.values()) for inner in outer]
    mapped_trans_to_cat_feats = [feat for feat in feat_names if any(picked_feat in feat for picked_feat in trans_to_cat_feats)]
    feats_without_trans_to_cat_feats = feat_names[~feat_names.isin(mapped_trans_to_cat_feats)]
    num_exception = exception

    if col_type == 'convert to cat':
        return trans_to_cat_feats

    for i, feat_name in enumerate(feats_without_trans_to_cat_feats):
        if col_type == 'num':
            if "|" in feat_name:
                feat_mapping.append(i) if return_idx == True else feat_mapping.append(feat_name)
        elif col_type == 'cat' and feat_name != num_exception:
            if "|" not in feat_name:
                feat_mapping.append(i) if return_idx == True else feat_mapping.append(feat_name)
    return feat_mapping


# In[6] Num Cat splitter for selector
'''Split numerical and categorical variables for selector'''
def get_cols_for_selector(feat_names, col_type:str, must_keep_feats:list=[], return_idx=False):
    feat_mapping = []
    mapped_must_keep_feats = [feat for feat in feat_names if any(picked_feat in feat for picked_feat in must_keep_feats)]
    feats_without_must_keep_feats = feat_names[~feat_names.isin(mapped_must_keep_feats)]

    for i, feat_name in enumerate(feats_without_must_keep_feats):
        if col_type == 'num':
            if "num trans__" in feat_name:
                feat_mapping.append(i) if return_idx == True else feat_mapping.append(feat_name)
        elif col_type == 'cat':
            if "cat trans__" in feat_name or 'remainder__' in feat_name or 'to cat trans__' in feat_name:
                feat_mapping.append(i) if return_idx == True else feat_mapping.append(feat_name)
    return feat_mapping


#In[0] Libraries & Setup
'''Setup'''
import pandas as pd
import numpy as np

import sys
import os
import importlib
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'ml_scripts'))

import utility_functions as uf
import specific_functions as sf
import IDA_n_cleansing
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def convert_datatypes(
    df: pd.DataFrame,
    data_schema: pd.DataFrame
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
            df[column] = df[column].astype(dtype_mapping[dtype])
        else:
            print(f"Warning: Unknown data type '{dtype}' for column '{column}'")
    return df

# In[1] Univariate Analysis Functions

def analyze_fraud_patterns(df: pd.DataFrame, tgt_col: str) -> pd.DataFrame:
    """Analyze and display patterns in fraud transactions."""
    fraud_df = df[df[tgt_col] == 1]
    print("\n=== Fraud Transaction Patterns ===")
    print("Common characteristics of fraud transactions:")
    display(fraud_df.describe(include='all').T)
    return fraud_df

def plot_amount_analysis(df: pd.DataFrame, tgt_col: str):
    """Plot transaction amount analysis."""
    print("\n=== Transaction Amount Analysis ===")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(x=tgt_col, y='amount', data=df)
    plt.title('Transaction Amount by Fraud Status')
    plt.subplot(1,2,2)
    sns.histplot(df['amount'], bins=50, kde=True)
    plt.title('Transaction Amount Distribution')
    plt.tight_layout()
    plt.show()

def plot_filtered_amount_analysis(df: pd.DataFrame, tgt_col: str, threshold: float = 2000):
    """Plot transaction amount analysis for amounts below threshold."""
    filtered_df = df[df['amount'] < threshold]
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(x=tgt_col, y='amount', data=filtered_df)
    plt.title(f'Transaction Amount (<${threshold}) by Fraud Status')
    plt.subplot(1,2,2)
    sns.histplot(filtered_df['amount'], bins=50, kde=True)
    plt.title(f'Transaction Amount Distribution (<${threshold})')
    plt.tight_layout()
    plt.show()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to dataframe."""
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['transaction_hour'] = df['transaction_date'].dt.hour
    return df

def plot_time_analysis(df: pd.DataFrame, tgt_col: str):
    """Plot transaction time analysis."""
    print("\n=== Transaction Time Analysis ===")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.countplot(x='transaction_hour', data=df)
    plt.title('Transaction Volume by Hour of Day')
    plt.subplot(1,2,2)
    sns.countplot(x='transaction_hour', data=df[df[tgt_col] == 1])
    plt.title('Fraud Transactions by Hour of Day')
    plt.tight_layout()
    plt.show()

def percentage_plot(ax, data, col):
    """Helper function to create percentage plots."""
    total = len(data)
    counts = data[col].value_counts()
    percentages = (counts / total * 100).sort_values(ascending=True)
    
    sns.barplot(x=percentages.values, y=percentages.index, ax=ax, color='skyblue')
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.5, p.get_y() + p.get_height()/2., 
                f'{width:.1f}%', 
                ha='left', va='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, percentages.max() * 1.15)

def analyze_merchants(df: pd.DataFrame, tgt_col: str):
    """Analyze merchant-related patterns."""
    print("\n=== Merchant Analysis ===")
    top_merchants = df['merchant_id'].value_counts().head(20)
    display(top_merchants.to_frame('Transaction Count'))

    plt.figure(figsize=(12,6))
    sns.countplot(y='merchant_id', data=df, order=df['merchant_id'].value_counts().index[:20])
    plt.title('Top 20 Merchants by Transaction Volume')
    plt.show()

    fraud_by_merchant = df.groupby('merchant_id')[tgt_col].mean().sort_values(ascending=False)
    display(fraud_by_merchant.head(10).to_frame('Fraud Rate'))

def analyze_customers(df: pd.DataFrame, tgt_col: str):
    """Analyze customer-related patterns."""
    print("\n=== Customer Analysis ===")
    customer_transactions = df['customer_id'].value_counts()
    display(customer_transactions.describe().to_frame('Transaction Count Stats'))
    display(customer_transactions.head(10).to_frame('Transaction Count'))

    fraud_customers = df[df[tgt_col] == 1]['customer_id'].value_counts()
    display(fraud_customers.to_frame('Fraud Transactions'))

def analyze_mccs(df: pd.DataFrame, tgt_col: str):
    """Analyze merchant category codes."""
    print("\n=== MCCs Analysis ===")
    mcc_counts = df['MCCs'].value_counts()
    display(mcc_counts.to_frame('Transaction Count'))

    plt.figure(figsize=(10,6))
    sns.countplot(y='MCCs', data=df, order=df['MCCs'].value_counts().index)
    plt.title('Transaction Count by MCC')
    plt.show()

    fraud_by_mcc = df.groupby('MCCs')[tgt_col].mean().sort_values(ascending=False)
    display(fraud_by_mcc.to_frame('Fraud Rate'))

def analyze_ip_addresses(df: pd.DataFrame, tgt_col: str):
    """Analyze IP address patterns."""
    print("\n=== IP Address Analysis ===")
    df['ip_first_octet'] = df['IP_address'].str.split('.').str[0]

    plt.figure(figsize=(12,6))
    sns.countplot(y='ip_first_octet', data=df, order=df['ip_first_octet'].value_counts().index[:20])
    plt.title('Top IP Address First Octets')
    plt.show()

    fraud_by_ip = df.groupby('ip_first_octet')[tgt_col].mean().sort_values(ascending=False)
    display(fraud_by_ip.head(20).to_frame('Fraud Rate'))

def plot_univariate_analysis(df: pd.DataFrame, col: str):
    """Plot univariate analysis for a single column."""
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(y=col, data=df)
    plt.title(f'Boxplot of {col}')
    plt.subplot(1,2,2)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# In[2] Multivariate Analysis Functions

def plot_categorical_analysis(df: pd.DataFrame, tgt_col: str, categorical_cols: list):
    """Plot analysis of categorical variables."""
    print("\n=== Categorical Variables Analysis ===")
    for col in categorical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        percentage_plot(ax1, df, col)
        ax1.set_title(f'{col} Distribution (Overall)')
        ax1.set_xlabel('Percentage')
        
        percentage_plot(ax2, df[df[tgt_col] == 1], col)
        ax2.set_title(f'{col} Distribution (Fraud Cases)')
        ax2.set_xlabel('Percentage')
        
        plt.tight_layout()
        plt.show()

def analyze_card_presence(df: pd.DataFrame, tgt_col: str):
    """Analyze card presence patterns."""
    print("\n=== Card Present Analysis ===")
    card_present_fraud = pd.crosstab(df['card_present_or_not'], df[tgt_col], normalize='index') * 100
    display(card_present_fraud)

def analyze_correlations(df: pd.DataFrame):
    """Analyze numerical correlations."""
    print("\n=== Correlation Analysis ===")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def analyze_customer_fraud_patterns(fraud_df: pd.DataFrame):
    """Analyze customer fraud patterns."""
    customer_fraud_patterns = fraud_df.groupby('customer_id').agg({
        'transaction_date': ['count', lambda x: (x.max() - x.min()).total_seconds()/3600],
        'amount': ['mean', 'sum']
    })
    customer_fraud_patterns.columns = ['transaction_count', 'time_span_hours', 'mean_amount', 'total_amount']
    display(customer_fraud_patterns.sort_values('transaction_count', ascending=False).head(10))

def analyze_store_card_merchants(df: pd.DataFrame, tgt_col: str):
    """Analyze store card merchant patterns."""
    print("\n=== Store Card Merchant Analysis ===")
    store_card_analysis = df.groupby('store_card_merchant_id').agg({
        tgt_col: 'mean',
        'customer_id': 'count'
    }).sort_values(tgt_col, ascending=False)
    display(store_card_analysis.head(10))

def analyze_merchant_age(df: pd.DataFrame, tgt_col: str):
    """Analyze fraud patterns by merchant age."""
    df['merchant_age'] = (pd.to_datetime(df['transaction_date']) - 
                          pd.to_datetime(df['merchant_established_date'])).dt.days / 365
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(df['merchant_age'], bins=30, kde=True)
    plt.title('Distribution of Merchant Age (Years)')
    plt.subplot(1,2,2)
    sns.boxplot(x=tgt_col, y='merchant_age', data=df)
    plt.title('Merchant Age vs. Fraud')
    plt.tight_layout()
    plt.show()

def analyze_customer_behavior(df: pd.DataFrame, tgt_col: str):
    """Analyze customer transaction behavior."""
    customer_stats = df.groupby('customer_id').agg({
        'amount': ['mean', 'std', 'count'],
        tgt_col: 'sum'
    }).sort_values(('amount', 'count'), ascending=False)
    display(customer_stats.head(10))

def analyze_geographic_patterns(df: pd.DataFrame, tgt_col: str):
    """Analyze fraud by geographic regions (using IP)."""
    df['ip_country'] = df['IP_address'].str.split('.').str[0]  # Simplified example
    
    plt.figure(figsize=(12,6))
    fraud_by_country = df[df[tgt_col]==1]['ip_country'].value_counts().head(10)
    sns.barplot(x=fraud_by_country.values, y=fraud_by_country.index)
    plt.title('Top 10 Countries by Fraud Count (IP-based)')
    plt.show()
    

# In[3] Temporal analysis

def plot_daily_transactions(df: pd.DataFrame, tgt_col: str):
    """Plot daily transaction patterns."""
    df['transaction_day'] = df['transaction_date'].dt.day
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.countplot(x='transaction_day', data=df)
    plt.title('Daily Transaction Volume')
    plt.subplot(1,2,2)
    sns.countplot(x='transaction_day', data=df[df[tgt_col]==1])
    plt.title('Daily Fraud Transactions')
    plt.tight_layout()
    plt.show()

def plot_transaction_trends(df: pd.DataFrame, tgt_col: str):
    """Plot transaction trends over time."""
    df['date'] = df['transaction_date'].dt.date
    daily_counts = df.groupby('date').size()
    daily_fraud = df[df[tgt_col]==1].groupby('date').size()
    
    plt.figure(figsize=(12,6))
    plt.plot(daily_counts.index, daily_counts.values, label='Total Transactions')
    plt.plot(daily_fraud.index, daily_fraud.values, label='Fraud Transactions', color='red')
    plt.title('Transaction Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def analyze_transaction_clustering(df: pd.DataFrame, tgt_col: str, time_window_minutes=30):
    """Analyze rapid transactions (potential fraud clusters)."""
    df = df.sort_values('transaction_date')
    df['time_diff'] = df.groupby('customer_id')['transaction_date'].diff().dt.total_seconds() / 60
    df['is_rapid'] = df['time_diff'] <= time_window_minutes
    rapid_stats = df.groupby(tgt_col)['is_rapid'].value_counts(normalize=True).unstack()
    rapid_stats = rapid_stats * 100
    print(f"Rapid transactions (within {time_window_minutes} mins):")
    display(rapid_stats.style.format("{:.1f}%"))


# In[4] reference
# time distribution
def plot_transaction_distributions(df):
    """
    Plots the distribution of transaction amounts and transaction times.

    Parameters:
        df (pd.DataFrame): The DataFrame containing transaction data. 
                           It should have 'amount' and 'date_in_sec' columns.
    """
    # Set up the figure
    plt.figure(figsize=(14, 6))

    # Plot the distribution of transaction amounts
    plt.subplot(1, 2, 1)
    sns.histplot(df['amount'], kde=True, bins=50, color='blue')
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')

    # Plot the distribution of transaction times
    time_val = df[df['No._of_day']<10]['date_in_sec'].sample(n=5000)
    plt.subplot(1, 2, 2)
    sns.histplot(time_val/86400, kde=True, bins=100, color='green')
    plt.title('Distribution of Transaction Times')
    plt.xlabel('Transaction Time (in seconds)')
    plt.ylabel('Frequency')

    # Show the plots
    plt.tight_layout()
    plt.show()


#review below two functions
def compute_transaction_statistics(
    transactions_df: pd.DataFrame,
    time_days_col: str,
    customer_id_col: str,
    fraud_col: str,
    scale_factor: int = 50
): #FIXME to optimize
    """
    Computes and returns statistics for transactions, including:
    - Number of transactions per day
    - Number of fraudulent transactions per day
    - Number of unique fraudulent cards per day

    Parameters:
        transactions_df (pd.DataFrame): The DataFrame containing transaction data.
        time_days_col (str): The column name representing transaction time in days.
        customer_id_col (str): The column name representing customer IDs.
        fraud_col (str): The column name indicating whether a transaction is fraudulent.
        scale_factor (int): The factor to scale the number of transactions per day. Default is 50.

    Returns:
        pd.DataFrame: A DataFrame containing the computed statistics.
    """
    # Number of transactions per day
    nb_tx_per_day = transactions_df.groupby([time_days_col])[customer_id_col].count()
    # Number of fraudulent transactions per day
    nb_fraud_per_day = transactions_df.groupby([time_days_col])[fraud_col].sum()
    # Number of unique fraudulent cards per day
    nb_fraudcard_per_day = transactions_df[transactions_df[fraud_col] > 0].groupby([time_days_col])[customer_id_col].nunique()

    # Align all series to the same index
    aligned_index = nb_tx_per_day.index.union(nb_fraud_per_day.index).union(nb_fraudcard_per_day.index)
    nb_tx_per_day = nb_tx_per_day.reindex(aligned_index, fill_value=0)
    nb_fraud_per_day = nb_fraud_per_day.reindex(aligned_index, fill_value=0)
    nb_fraudcard_per_day = nb_fraudcard_per_day.reindex(aligned_index, fill_value=0)

    # Prepare data for plotting
    n_days = len(aligned_index)
    tx_stats = pd.DataFrame({
        "value": pd.concat([nb_tx_per_day / scale_factor, nb_fraud_per_day, nb_fraudcard_per_day]),
        "stat_type": (
            ["nb_tx_per_day"] * n_days +
            ["nb_fraud_per_day"] * n_days +
            ["nb_fraudcard_per_day"] * n_days
        )
    })
    tx_stats = tx_stats.reset_index()

    return tx_stats

def plot_fraud_and_transaction_statistics(
    tx_stats, 
    time_days_col, 
    value_col, 
    stat_type_col, 
    hue_order=None, 
    scale_factor=50
): #FIXME to optimize
    """
    Plots the total transactions, number of fraudulent transactions, and number of compromised cards per day.

    Parameters:
        tx_stats (pd.DataFrame): The DataFrame containing transaction statistics.
        time_days_col (str): The column name representing transaction time in days.
        value_col (str): The column name representing the values to plot.
        stat_type_col (str): The column name representing the type of statistic (e.g., 'nb_tx_per_day').
        hue_order (list): The order of the hue categories for the plot. Default is None.
        scale_factor (int): The scaling factor for transactions per day. Default is 50.
    """
    sns.set(style='darkgrid')
    sns.set(font_scale=1.4)

    # Create the figure
    fraud_and_transactions_stats_fig = plt.gcf()
    fraud_and_transactions_stats_fig.set_size_inches(15, 8)

    # Plot the data
    sns_plot = sns.lineplot(
        x=time_days_col,
        y=value_col,
        data=tx_stats,
        hue=stat_type_col,
        hue_order=hue_order,
        legend=False
    )

    # Set plot title and labels
    sns_plot.set_title(
        'Total transactions, and number of fraudulent transactions \n'
        'and number of compromised cards per day',
        fontsize=20
    )
    sns_plot.set(
        xlabel="Number of days since beginning of data generation",
        ylabel="Number"
    )
    max_value = tx_stats[value_col].max()
    sns_plot.set_ylim([0, max_value*1.1])

    # Add legend
    labels_legend = [
        f"# transactions per day (/ {scale_factor})",
        "# fraudulent txs per day",
        "# fraudulent cards per day"
    ]
    sns_plot.legend(loc='upper left', labels=labels_legend, bbox_to_anchor=(1.05, 1), fontsize=15)

    # Show the plot
    plt.tight_layout()
    plt.show()


# In[9] main function to perform EDA

def perform_eda(df: pd.DataFrame, tgt_col: str, categorical_cols: list = None):
    """
    Perform exploratory data analysis (EDA) on the credit card fraud dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame containing credit card transaction data.
        tgt_col (str): The target column indicating fraud.
    """
    input_df = df.copy()

    # Perform analysis
    fraud_df = analyze_fraud_patterns(input_df, tgt_col)
    plot_amount_analysis(input_df, tgt_col)
    #plot_filtered_amount_analysis(input_df, tgt_col)
    input_df = add_time_features(input_df)
    plot_time_analysis(input_df, tgt_col)
    plot_categorical_analysis(input_df, tgt_col, categorical_cols)
    analyze_merchants(input_df, tgt_col)
    analyze_customers(input_df, tgt_col)
    analyze_card_presence(input_df, tgt_col)
    analyze_mccs(input_df, tgt_col)
    analyze_ip_addresses(input_df, tgt_col)
    analyze_geographic_patterns(input_df, tgt_col)
    analyze_correlations(input_df)
    analyze_customer_fraud_patterns(fraud_df)
    analyze_store_card_merchants(input_df, tgt_col)


# In[0] Load Data
if __name__ == '__main__':
    # Load the dataset
    cf_raw_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            'interim',
            'credit_fraud_data_cleaned.csv'
        ),
    )

    data_schema = pd.read_excel(
        os.path.join(os.path.dirname(os.getcwd()),
        'configurations',
        'data_schema.xlsx'),
        sheet_name='schema'
    )
    tgt_col = 'is_fraud'
    cf_raw_df = convert_datatypes(cf_raw_df, data_schema)


    # Perform EDA
    perform_eda(cf_raw_df, tgt_col, ['merchant_category', 'type_of_credit_card_used', 'card_present_or_not'])

    # In[2] cleansing and validation
    
    '''plot_transaction_distributions(cf_raw_df)
    tx_stats = compute_transaction_statistics(cf_raw_df, 'No._of_day', 'customer_id', 'is_fraud')
    plot_fraud_and_transaction_statistics(tx_stats, 'No._of_day', 'value', 'stat_type')'''

# %%

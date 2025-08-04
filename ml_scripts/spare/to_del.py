# In[0] Libraries
'''Import necessary libraries'''
# Standard library imports
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Union

# Third-party imports
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as imb_pipe
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn import (
    calibration, discriminant_analysis, ensemble, feature_selection, 
    gaussian_process, linear_model, metrics, model_selection, 
    naive_bayes, neighbors, svm, tree
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline as skl_pipe
from xgboost import XGBClassifier

# Local imports
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'ml_scripts'))
import config as cfg
import specific_functions as sf
import utility_functions as uf

# Disable chained assignment warning
pd.options.mode.chained_assignment = None



preprocessor = skl_pipe([
    ('TypeConvert', sf.data_type_prep),
    ('DomainDrop', sf.domain_drop_prep),
])

irrelevant_cols = [
    'transaction_date',
    'customer_id', 
    'amount',
    'date_in_sec', 
    'No._of_day', 
    'merchant_id', 
    'pos_id', 
    'merchant_established_date', 
    'MCCs',
    'merchant_category',
    'IP_address',
    'type_of_credit_card_used',
    'store_card_merchant_id',
    'IP_address_country_code', #FIXME
    'country_code_date',
    'type_of_credit_card_used_with_mer_id',
    'type_of_credit_card_used',
] #FIXME

preprocessor.set_params(
    TypeConvert__kw_args={
        "data_schema": pd.read_excel(
            os.path.join(os.path.dirname(os.getcwd()) if __name__ == "__main__" else os.getcwd(), 'configurations', 'data_schema.xlsx'),
            sheet_name='schema'
        )},
    DomainDrop__kw_args={"irrelevant_cols": irrelevant_cols}
)

def model_prediction(self, export_csv: bool = True) -> Union[np.ndarray, pd.DataFrame]:
    if self._trained_model is None:
        raise Exception("Please either fit a new model or get an existing model first")
    
    if self._phase == 'deploy':
        mdl_predictions = self._trained_model.predict(self._data_for_mdl)
        preds_to_export = pd.DataFrame({
            'id': self._data_for_mdl['id'].copy(),
            self._target: mdl_predictions
        })
    else:
        if self._test_X is None:
            raise RuntimeError("Cannot proceed because not yet split the data")
        mdl_predictions = self._trained_model.predict(self._test_X)


'''def bin_custom(dataset):
    bins = np.array([0, 15, 25])
    dataset = np.digitize(dataset, bins)
    return dataset'''

'''reference

distribution_amount_times_fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = cf_raw_df[cf_raw_df['No._of_day']<10]['amount'].sample(n=10000).values
time_val = cf_raw_df[cf_raw_df['No._of_day']<10]['date_in_sec'].sample(n=10000).values

sns.distplot(amount_val, ax=ax[0], color='r', hist = True, kde = False)
ax[0].set_title('Distribution of transaction amounts', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])
ax[0].set(xlabel = "Amount", ylabel="Number of transactions")

# We divide the time variables by 86400 to transform seconds to days in the plot
sns.distplot(time_val/86400, ax=ax[1], color='b', bins = 100, hist = True, kde = False)
ax[1].set_title('Distribution of transaction times', fontsize=14)
ax[1].set_xlim([min(time_val/86400), max(time_val/86400)])
ax[1].set_xticks(range(10))
ax[1].set(xlabel = "Time (days)", ylabel="Number of transactions")
'''


# %%

class EDAAnalyzer:
    """Enhanced class to perform comprehensive exploratory data analysis on credit card fraud dataset."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with the dataframe to analyze."""
        self.df = df.copy()
        self._setup_logging()
        self._preprocess_data()
        
    def _setup_logging(self):
        """Configure logging for the EDA process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _preprocess_data(self):
        """Perform initial data preprocessing."""
        try:
            # Convert dates and extract features
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
            self.df['transaction_hour'] = self.df['transaction_date'].dt.hour
            self.df['transaction_day'] = self.df['transaction_date'].dt.day
            self.df['transaction_dayofweek'] = self.df['transaction_date'].dt.dayofweek
            self.df['transaction_month'] = self.df['transaction_date'].dt.month
            
            # Convert merchant established date to age
            self.df['merchant_established_date'] = pd.to_datetime(self.df['merchant_established_date'], errors='coerce')
            self.df['merchant_age'] = (self.df['transaction_date'] - self.df['merchant_established_date']).dt.days / 365
            
            # Extract potential country from IP (simplified)
            self.df['ip_prefix'] = self.df['IP_address'].str.split('.').str[0]
            
            self.logger.info("Data preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            raise
            
    def analyze_basic_stats(self) -> Dict:
        """Analyze basic statistics of the dataset."""
        stats = {
            'num_transactions': len(self.df),
            'num_fraud': self.df['is_fraud'].sum(),
            'fraud_percentage': self.df['is_fraud'].mean() * 100,
            'date_range': {
                'min': str(self.df['transaction_date'].min()),
                'max': str(self.df['transaction_date'].max())
            },
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns}
        }
        return stats
    
    def analyze_transaction_amounts(self) -> Dict:
        """Analyze transaction amount statistics with more detailed breakdown."""
        amount_stats = {
            'overall': self.df['amount'].describe().to_dict(),
            'by_fraud_status': self.df.groupby('is_fraud')['amount'].describe().to_dict(),
            'amount_bins': {
                'fraud': pd.cut(self.df[self.df['is_fraud']==1]['amount'], 
                               bins=[0, 10, 50, 100, 500, 1000, 5000, float('inf')],
                               right=False).value_counts().sort_index().to_dict(),
                'non_fraud': pd.cut(self.df[self.df['is_fraud']==0]['amount'], 
                                   bins=[0, 10, 50, 100, 500, 1000, 5000, float('inf')],
                                   right=False).value_counts().sort_index().to_dict()
            },
            'top_amount_fraud': self.df[self.df['is_fraud']==1].nlargest(5, 'amount')[['amount', 'merchant_category', 'type_of_credit_card_used']].to_dict('records')
        }
        return amount_stats
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in transactions and fraud."""
        fraud_df = self.df[self.df['is_fraud'] == 1]
        
        return {
            'by_hour': {
                'all': self.df['transaction_hour'].value_counts().sort_index().to_dict(),
                'fraud': fraud_df['transaction_hour'].value_counts().sort_index().to_dict(),
                'fraud_rate': self.df.groupby('transaction_hour')['is_fraud'].mean().to_dict()
            },
            'by_day': {
                'all': self.df['transaction_day'].value_counts().sort_index().to_dict(),
                'fraud': fraud_df['transaction_day'].value_counts().sort_index().to_dict(),
                'fraud_rate': self.df.groupby('transaction_day')['is_fraud'].mean().to_dict()
            },
            'by_dayofweek': {
                'all': self.df['transaction_dayofweek'].value_counts().sort_index().to_dict(),
                'fraud': fraud_df['transaction_dayofweek'].value_counts().sort_index().to_dict(),
                'fraud_rate': self.df.groupby('transaction_dayofweek')['is_fraud'].mean().to_dict()
            }
        }
    
    def analyze_card_merchant_info(self) -> Dict:
        """Analyze card and merchant information with fraud rates."""
        card_stats = self.df.groupby('type_of_credit_card_used').agg(
            count=('is_fraud', 'count'),
            fraud_count=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', 'mean')
        ).sort_values('fraud_rate', ascending=False).to_dict('index')
        
        merchant_stats = self.df.groupby('merchant_category').agg(
            count=('is_fraud', 'count'),
            fraud_count=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', 'mean')
        ).sort_values('fraud_rate', ascending=False).to_dict('index')
        
        mcc_stats = self.df.groupby('MCCs').agg(
            count=('is_fraud', 'count'),
            fraud_count=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', 'mean')
        ).sort_values('fraud_rate', ascending=False).head(10).to_dict('index')
        
        fraud_df = self.df[self.df['is_fraud'] == 1]
        
        return {
            'card_types': card_stats,
            'card_presence': self.df.groupby('card_present_or_not').agg(
                count=('is_fraud', 'count'),
                fraud_count=('is_fraud', 'sum'),
                fraud_rate=('is_fraud', 'mean')
            ).to_dict('index'),
            'merchant_categories': merchant_stats,
            'merchant_category_codes': mcc_stats,
            'merchant_age_stats': {
                'fraud': fraud_df['merchant_age'].describe().to_dict(),
                'non_fraud': self.df[self.df['is_fraud']==0]['merchant_age'].describe().to_dict()
            }
        }
    
    def analyze_customer_behavior(self) -> Dict:
        """Analyze customer transaction patterns with more metrics."""
        customer_stats = self.df.groupby('customer_id').agg(
            transaction_count=('amount', 'count'),
            total_amount=('amount', 'sum'),
            fraud_count=('is_fraud', 'sum'),
            avg_amount=('amount', 'mean')
        )
        
        fraud_customers = customer_stats[customer_stats['fraud_count'] > 0]
        
        return {
            'customer_summary': customer_stats.describe().to_dict(),
            'top_fraud_customers': fraud_customers.nlargest(5, 'fraud_count').to_dict('index'),
            'repeat_fraud_customers': len(fraud_customers[fraud_customers['fraud_count'] > 1]),
            'fraud_clustering': {
                'transactions_before_fraud': self._analyze_transactions_before_fraud()
            }
        }
    
    def _analyze_transactions_before_fraud(self) -> Dict:
        """Analyze transaction patterns before first fraud for each customer."""
        fraud_customers = self.df[self.df['is_fraud'] == 1]['customer_id'].unique()
        results = []
        
        for cust in fraud_customers:
            cust_df = self.df[self.df['customer_id'] == cust].sort_values('transaction_date')
            first_fraud_idx = cust_df['is_fraud'].idxmax()
            before_fraud = cust_df.loc[:first_fraud_idx]
            
            if len(before_fraud) > 1:  # At least one transaction before fraud
                results.append({
                    'customer_id': cust,
                    'transactions_before_fraud': len(before_fraud) - 1,
                    'time_to_first_fraud': (before_fraud.iloc[-1]['transaction_date'] - before_fraud.iloc[0]['transaction_date']).total_seconds() / 3600,
                    'avg_amount_before_fraud': before_fraud.iloc[:-1]['amount'].mean()
                })
        
        if results:
            summary_df = pd.DataFrame(results)
            return summary_df.describe().to_dict()
        return {}
    
    def analyze_merchant_behavior(self) -> Dict:
        """Analyze merchant transaction patterns with fraud rates."""
        merchant_stats = self.df.groupby('merchant_id').agg(
            transaction_count=('amount', 'count'),
            total_amount=('amount', 'sum'),
            fraud_count=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', 'mean'),
            avg_amount=('amount', 'mean')
        ).sort_values('fraud_count', ascending=False)
        
        fraud_merchants = merchant_stats[merchant_stats['fraud_count'] > 0]
        
        return {
            'merchant_summary': merchant_stats.describe().to_dict(),
            'top_fraud_merchants': fraud_merchants.head(10).to_dict('index'),
            'merchant_fraud_correlation': {
                'transaction_count_vs_fraud': merchant_stats[['transaction_count', 'fraud_count']].corr().iloc[0,1],
                'avg_amount_vs_fraud': merchant_stats[['avg_amount', 'fraud_rate']].corr().iloc[0,1]
            }
        }
    
    def analyze_geographic_patterns(self) -> Dict:
        """Analyze geographic patterns using IP addresses."""
        fraud_df = self.df[self.df['is_fraud'] == 1]
        
        return {
            'ip_prefix_distribution': {
                'all': self.df['ip_prefix'].value_counts().head(10).to_dict(),
                'fraud': fraud_df['ip_prefix'].value_counts().head(10).to_dict()
            },
            'ip_fraud_rates': self.df.groupby('ip_prefix')['is_fraud'].mean().nlargest(10).to_dict()
        }
    
    def analyze_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for numerical features."""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        return self.df[numerical_cols].corr()
    
    def generate_visualizations(self, interactive: bool = False) -> None:
        """Generate comprehensive EDA visualizations with option for interactive plots."""
        if interactive:
            self._generate_interactive_visualizations()
        else:
            self._generate_static_visualizations()
    
    def _generate_static_visualizations(self) -> None:
        """Generate static matplotlib visualizations."""
        plt.figure(figsize=(20, 24))
        
        # Transaction amount distribution
        plt.subplot(4, 3, 1)
        sns.histplot(self.df['amount'], bins=50, kde=True)
        plt.title('Transaction Amount Distribution')
        plt.xlim(0, self.df['amount'].quantile(0.99))
        
        # Fraud by hour
        plt.subplot(4, 3, 2)
        hourly_fraud_rate = self.df.groupby('transaction_hour')['is_fraud'].mean()
        sns.lineplot(x=hourly_fraud_rate.index, y=hourly_fraud_rate.values)
        plt.title('Fraud Rate by Hour of Day')
        plt.ylabel('Fraud Rate')
        plt.xlabel('Hour of Day')
        
        # Fraud by card type
        plt.subplot(4, 3, 3)
        fraud_by_card = self.df.groupby('type_of_credit_card_used')['is_fraud'].mean().sort_values(ascending=False)
        fraud_by_card.plot(kind='bar')
        plt.title('Fraud Percentage by Card Type')
        plt.ylabel('Fraud Percentage')
        
        # Fraud by merchant category
        plt.subplot(4, 3, 4)
        fraud_by_merchant = self.df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False).head(10)
        fraud_by_merchant.plot(kind='bar')
        plt.title('Top 10 Fraud-Prone Merchant Categories')
        plt.ylabel('Fraud Percentage')
        
        # Card presence and fraud
        plt.subplot(4, 3, 5)
        sns.barplot(x='card_present_or_not', y='is_fraud', data=self.df)
        plt.title('Fraud Rate by Card Presence')
        
        # Amount vs Fraud
        plt.subplot(4, 3, 6)
        sns.boxplot(x='is_fraud', y='amount', data=self.df)
        plt.ylim(0, self.df['amount'].quantile(0.99))
        plt.title('Transaction Amount by Fraud Status')
        
        # Merchant age vs fraud
        plt.subplot(4, 3, 7)
        sns.boxplot(x='is_fraud', y='merchant_age', data=self.df)
        plt.title('Merchant Age by Fraud Status')
        plt.ylabel('Merchant Age (years)')
        
        # IP prefix analysis
        plt.subplot(4, 3, 8)
        top_fraud_ips = self.df.groupby('ip_prefix')['is_fraud'].mean().nlargest(10)
        top_fraud_ips.plot(kind='bar')
        plt.title('Top 10 Fraud-Prone IP Prefixes')
        plt.ylabel('Fraud Rate')
        
        # Customer transaction patterns
        plt.subplot(4, 3, 9)
        customer_fraud_counts = self.df.groupby('customer_id')['is_fraud'].sum()
        customer_fraud_counts[customer_fraud_counts > 0].value_counts().plot(kind='bar')
        plt.title('Customers by Number of Fraudulent Transactions')
        plt.xlabel('Number of Fraudulent Transactions')
        plt.ylabel('Count of Customers')
        
        # MCC analysis
        plt.subplot(4, 3, 10)
        fraud_by_mcc = self.df.groupby('MCCs')['is_fraud'].mean().nlargest(10)
        fraud_by_mcc.plot(kind='bar')
        plt.title('Top 10 Fraud-Prone MCC Codes')
        plt.ylabel('Fraud Rate')
        
        # Time between transactions for fraud customers
        plt.subplot(4, 3, 11)
        fraud_customers = self.df[self.df['is_fraud'] == 1]['customer_id'].unique()
        time_diffs = []
        for cust in fraud_customers[:100]:  # Sample for performance
            cust_trans = self.df[self.df['customer_id'] == cust].sort_values('transaction_date')
            if len(cust_trans) > 1:
                diffs = cust_trans['transaction_date'].diff().dt.total_seconds().div(60).dropna()
                time_diffs.extend(diffs)
        sns.histplot(time_diffs, bins=50)
        plt.title('Time Between Transactions (Fraud Customers)')
        plt.xlabel('Minutes between transactions')
        plt.xlim(0, 1440)  # Limit to 24 hours
        
        plt.tight_layout()
        plt.show()
    
    def _generate_interactive_visualizations(self) -> None:
        """Generate interactive Plotly visualizations."""
        # Create subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Transaction Amount Distribution',
                'Fraud Rate by Hour of Day',
                'Fraud Percentage by Card Type',
                'Top 10 Fraud-Prone Merchant Categories',
                'Fraud Rate by Card Presence',
                'Transaction Amount by Fraud Status',
                'Merchant Age by Fraud Status',
                'Top 10 Fraud-Prone IP Prefixes',
                'Customers by Fraudulent Transactions',
                'Top 10 Fraud-Prone MCC Codes',
                'Time Between Transactions (Fraud)'
            )
        )
        
        # Plot 1: Transaction amount distribution
        fig.add_trace(
            go.Histogram(x=self.df['amount'], nbinsx=50, name='Amount Distribution'),
            row=1, col=1
        )
        
        # Plot 2: Fraud rate by hour
        hourly_fraud = self.df.groupby('transaction_hour')['is_fraud'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=hourly_fraud['transaction_hour'], y=hourly_fraud['is_fraud'], 
                      mode='lines', name='Fraud Rate'),
            row=1, col=2
        )
        
        # Plot 3: Fraud by card type
        card_fraud = self.df.groupby('type_of_credit_card_used')['is_fraud'].mean().sort_values(ascending=False).reset_index()
        fig.add_trace(
            go.Bar(x=card_fraud['type_of_credit_card_used'], y=card_fraud['is_fraud'], name='Card Type'),
            row=1, col=3
        )
        
        # Plot 4: Top fraud merchant categories
        merchant_fraud = self.df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=False).head(10).reset_index()
        fig.add_trace(
            go.Bar(x=merchant_fraud['merchant_category'], y=merchant_fraud['is_fraud'], name='Merchant Category'),
            row=2, col=1
        )
        
        # Plot 5: Card presence fraud rate
        presence_fraud = self.df.groupby('card_present_or_not')['is_fraud'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=presence_fraud['card_present_or_not'], y=presence_fraud['is_fraud'], name='Card Presence'),
            row=2, col=2
        )
        
        # Plot 6: Amount by fraud status
        fig.add_trace(
            go.Box(x=self.df['is_fraud'], y=self.df['amount'], name='Amount by Fraud'),
            row=2, col=3
        )
        
        # Plot 7: Merchant age by fraud
        fig.add_trace(
            go.Box(x=self.df['is_fraud'], y=self.df['merchant_age'], name='Merchant Age'),
            row=3, col=1
        )
        
        # Plot 8: Top fraud IP prefixes
        ip_fraud = self.df.groupby('ip_prefix')['is_fraud'].mean().nlargest(10).reset_index()
        fig.add_trace(
            go.Bar(x=ip_fraud['ip_prefix'], y=ip_fraud['is_fraud'], name='IP Prefix'),
            row=3, col=2
        )
        
        # Plot 9: Customers by fraud counts
        cust_fraud = self.df.groupby('customer_id')['is_fraud'].sum()
        cust_fraud = cust_fraud[cust_fraud > 0].value_counts().reset_index()
        fig.add_trace(
            go.Bar(x=cust_fraud['index'], y=cust_fraud['is_fraud'], name='Fraud Counts'),
            row=3, col=3
        )
        
        # Plot 10: Top fraud MCC codes
        mcc_fraud = self.df.groupby('MCCs')['is_fraud'].mean().nlargest(10).reset_index()
        fig.add_trace(
            go.Bar(x=mcc_fraud['MCCs'].astype(str), y=mcc_fraud['is_fraud'], name='MCC Codes'),
            row=4, col=1
        )
        
        # Plot 11: Time between transactions for fraud customers
        fraud_customers = self.df[self.df['is_fraud'] == 1]['customer_id'].unique()
        time_diffs = []
        for cust in fraud_customers[:100]:  # Sample for performance
            cust_trans = self.df[self.df['customer_id'] == cust].sort_values('transaction_date')
            if len(cust_trans) > 1:
                diffs = cust_trans['transaction_date'].diff().dt.total_seconds().div(60).dropna()
                time_diffs.extend(diffs)
        fig.add_trace(
            go.Histogram(x=time_diffs, nbinsx=50, name='Time Between Transactions'),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title_text="Comprehensive Fraud Analysis Dashboard",
            showlegend=False
        )
        
        fig.show()
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive EDA report with all analyses."""
        try:
            report = {
                'basic_stats': self.analyze_basic_stats(),
                'amount_analysis': self.analyze_transaction_amounts(),
                'temporal_patterns': self.analyze_temporal_patterns(),
                'card_merchant_analysis': self.analyze_card_merchant_info(),
                'customer_behavior': self.analyze_customer_behavior(),
                'merchant_behavior': self.analyze_merchant_behavior(),
                'geographic_patterns': self.analyze_geographic_patterns(),
                'correlation_matrix': self.analyze_correlation_matrix().to_dict()
            }
            self.logger.info("Comprehensive EDA report generated successfully")
            return report
        except Exception as e:
            self.logger.error(f"Error generating EDA report: {str(e)}")
            raise
    
    def perform_full_eda(self, interactive_viz: bool = False) -> Tuple[Dict, None]:
        """Perform complete EDA and return report with visualizations.
        
        Args:
            interactive_viz: Whether to use interactive Plotly visualizations (default: False)
        
        Returns:
            Tuple containing the report dictionary and None (for consistency)
        """
        report = self.generate_report()
        self.generate_visualizations(interactive=interactive_viz)
        return report, None


def perform_eda(df: pd.DataFrame) -> Dict:
    """
    Perform exploratory data analysis on the credit card fraud dataset.
    
    Args:
        df (pd.DataFrame): The raw DataFrame containing credit card transaction data.
        
    Returns:
        Dict: A comprehensive report of the EDA findings.
    """
    analyzer = EDAAnalyzer(df)
    return analyzer.perform_full_eda()
          
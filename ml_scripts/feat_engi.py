
# In[0] setup
import numpy as np
import pandas as pd
from datetime import datetime

import geoip2.database
import requests
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import pytz
import os


# In[1] IP behaviour feature transform: get_ip_country_code & timezone

def add_ip_country_code(df):
    ip_address = df['IP_address']
    # Path to the GeoLite2 database file
    GEOIP_DATABASE_PATH = 'GeoLite2-Country.mmdb'

    # Initialize the GeoIP2 reader
    reader = geoip2.database.Reader(GEOIP_DATABASE_PATH)

    # Define the function to get the country code
    def ip_to_country_code(val):
        try:
            response = reader.country(val)
            return response.country.iso_code  # Return the 2-letter country code
        except Exception as e:
            return None
        
    country_code = ip_address.apply(ip_to_country_code)
    reader.close()
    df['IP_address_country_code'] = country_code
    return df

def add_ip_country_date(
        df, 
        date_col='transaction_date',
        country_code_col='IP_address_country_code',
    ):
    df[date_col] = pd.to_datetime(df[date_col])

    # Localize the UK timestamp to UTC (or Europe/London if UK time is needed)
    try:
        df[date_col] = df[date_col].dt.tz_localize('UTC')  # Use 'Europe/London' for UK time
    except TypeError as e:
        if "Already tz-aware, use tz_convert to convert" in str(e):
            df[date_col] = df[date_col].dt.tz_convert('UTC')

    # Function to get timezone from country code
    def get_timezone(country_code):
        try:
            # Get the first timezone for the country code
            return pytz.country_timezones(country_code)[0]
        except (KeyError, IndexError):
            # Default to UTC if country code is not found
            return 'UTC'
        except AttributeError as e:
            if "'NoneType' object has no attribute 'upper'" in str(e):
                return np.nan
        except ValueError as e:
            if "NaTType does not support astimezone" in str(e):
                return np.nan

    # Apply the function to get the timezone for each country code
    df['timezone'] = df[country_code_col].apply(get_timezone)

    # Convert UK timestamp to the respective timezone
    df['country_code_date'] = df.apply(
        lambda row: row[date_col].astimezone(pytz.timezone(row['timezone'])) 
        if pd.notna(row[date_col]) and pd.notna(row['timezone'])
        else np.nan,
        axis=1
    )

    # Drop the intermediate timezone column if not needed
    df.drop(columns=['timezone'], inplace=True)
    return df

def add_total_seconds_since_midnight(
        df, 
        date_col='transaction_date',
):
    # Apply the function to calculate seconds since midnight
    df[f'{date_col}_sec_since_midnight'] = df[date_col].apply(lambda x: -999 if pd.isna(x) else x.hour * 3600 + x.minute * 60 + x.second)
    return df


# In[] IP behaviour feature transform: add_distinct_ip_window

def add_distinct_ip_window(
        df, 
        date_col='transaction_date', 
        id_col='customer_id',
        ip_col='IP_address',
        window=[1,5,12,28]
):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    
    for window_interval in window:
        distinct_ip_counts = []
        # Process each customer separately
        for customer_id, group in df.groupby(id_col):
            ip_set = set()
            timestamps = group[date_col].tolist()
            ip_addresses = group[ip_col].tolist()
            
            start_idx = 0
            count_list = []
            
            for end_idx in range(len(group)):
                # Remove outdated IPs from the set
                while timestamps[start_idx] < timestamps[end_idx] - pd.Timedelta(days=window_interval):
                    ip_set.discard(ip_addresses[start_idx])
                    start_idx += 1
                
                # Add the current IP
                ip_set.add(ip_addresses[end_idx])
                no_null_like_ip_set = {ip for ip in ip_set if pd.notna(ip) and str(ip).strip()}
                count_list.append(len(no_null_like_ip_set)) #add count of unique IP
            
            distinct_ip_counts.extend(count_list)
        
        df[f'distinct_ip_{window_interval}d'] = distinct_ip_counts
    df = df.sort_values([date_col])
    return df

def add_ip_used_by_many(
        df,
        id_count_threshold:list=None,
):
    id_count_threshold = [int(df.customer_id.nunique()*0.01)] if id_count_threshold is None else id_count_threshold
    # Count unique customer IDs per IP
    ip_customer_counts = df.groupby('IP_address')['customer_id'].nunique()

    # Flag IPs used by multiple customer IDs (count > 1)
    for threshold in id_count_threshold:
        df[f'IP_used_by_{threshold}_customer_id_or_more'] = df['IP_address'].map(lambda ip: ip_customer_counts[ip] >= threshold, na_action='ignore').astype(bool).astype('Int64')
        df[f'IP_used_by_{threshold}_customer_id_or_more'] = df[f'IP_used_by_{threshold}_customer_id_or_more'].fillna(0).astype(int)
    return df

def add_time_since_last_ip_per_customer(
    df, 
    customer_col='customer_id', 
    ip_col='IP_address', 
    timestamp_col='transaction_date',
    fill_na_with_zero=True
): #FIXME
    """
    Adds a column showing seconds since the same IP last appeared **for each customer**.
    
    Args:
        df: Input DataFrame
        customer_col: Column name for customer IDs (default: 'customer_id')
        ip_col: Column name for IP addresses (default: 'IP_address')
        timestamp_col: Column name for timestamps (default: 'timestamp')
    
    Returns:
        DataFrame with new column 'seconds_since_last_ip'
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by customer and timestamp (chronological order per customer)
    df = df.sort_values([customer_col, timestamp_col]).reset_index(drop=True)
    
    # Group by customer + IP, then shift timestamps
    df['prev_timestamp'] = df.groupby([customer_col, ip_col])[timestamp_col].shift(1)
    
    # Calculate time difference in seconds
    df['seconds_since_last_ip'] = (
        (df[timestamp_col] - df['prev_timestamp']).dt.total_seconds()
    )

    # Optionally replace NaN with 0 (for first occurrences)
    if fill_na_with_zero:
        df['seconds_since_last_ip'] = df['seconds_since_last_ip'].fillna(0)

    # Drop temporary column
    df = df.drop(columns=['prev_timestamp'])
    df = df.sort_values('transaction_date')
    
    return df

# In[] IP behaviour feature transform: timestamp decomposition
import holidays

def add_is_weekend(
        df, 
        date_col='transaction_date', 
        output_col='is_weekend'
):
    transaction_date = pd.to_datetime(
        df[date_col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x),
        errors='coerce'
    )
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    df[output_col] = transaction_date.dt.weekday >= 5
    # Binary value: 0 if weekday, 1 if weekend
    df[output_col] = df[output_col].astype(int)
    return df

def add_is_each_weekday(
        df, 
        date_col='transaction_date',
        local_or_IP_country='local'
):
    transaction_date = pd.to_datetime(
        df[date_col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x),
        errors='coerce'
    )
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i in range(7):
        is_weekday = transaction_date.dt.weekday == i
        # Binary value: 0 if weekday, 1 if weekend
        df[f'{local_or_IP_country}_is_{weekday[i]}'] = is_weekday.astype(int)
    return df

def add_is_UK_bank_holiday(
        df, 
        country_code='GB',
        date_col='transaction_date'
):
    # Create a UK holidays object
    holidays_of_country = holidays.country_holidays(country_code)

    # Function to check if a date is a UK bank holiday
    df[f'is_{country_code}_bank_holiday'] = df[date_col].dt.date.map(lambda x: x in holidays_of_country)
    df[f'is_{country_code}_bank_holiday'] = df[f'is_{country_code}_bank_holiday'].astype(int)
    return df

def add_bank_holiday_by_country_code(
        df, 
        date_col='transaction_date'
):
    df[date_col] = pd.to_datetime(
        df[date_col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x),
        errors='coerce'
    )

    def is_bank_holiday(row):
        #country_code = row['IP_address_country_code'].upper()
        date = row[date_col]
        
        try:
            country_holidays = holidays.country_holidays(row['IP_address_country_code'].upper())
            return 1 if date in country_holidays else 0
        except NotImplementedError:
            return 'Country code not supported'
        except (KeyError, TypeError, AttributeError):
            return np.nan  # Return False if country code is not supported

    df['bank_holiday_of_IP_country'] = df.apply(is_bank_holiday, axis=1)
    return df

def decompose_date_components(df, date_col='transaction_date'):
    df['transaction_date_day'] = df[date_col].dt.day
    df['transaction_date_month'] = df[date_col].dt.month
    #df['transaction_date_year'] = df[date_col].dt.year
    return df

'''- date_isWorkday
- date_isBankHoliday_anywhere
- date_isBankHoliday_inCountryOfCurrentIP
- date_isBankHoliday_inCountryOfPreviousIP
- date_isBankHoliday_inCountryOfAnyRecentIP
- time_inCountryOfCurrentIP'''


# In[] IP behaviour feature transform: add_time_features

def is_night(transaction_date):
    # Get the hour of the transaction
    is_night = transaction_date.dt.hour <= 6
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = is_night.astype(int)
    
    return is_night

def add_recent_ip_time_ago_by_customer(df, date_col='transaction_date'):
    df['customer_id_sec_since_last_trade'] = df.groupby('customer_id')['transaction_date'].diff().dt.total_seconds()
    df['customer_id_sec_since_last_trade'] = df['customer_id_sec_since_last_trade'].fillna(0).astype(int)
    return df

def add_distinct_ip_window(
        df, 
        date_col='transaction_date', 
        id_col='customer_id',
        ip_col='IP_address',
        window=[1,5,12,28,60]
):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    
    for window_interval in window:
        distinct_ip_counts = []
        # Process each customer separately
        for customer_id, group in df.groupby(id_col):
            ip_set = set()
            timestamps = group[date_col].tolist()
            ip_addresses = group[ip_col].tolist()
            
            start_idx = 0
            count_list = []
            
            for end_idx in range(len(group)):
                # Remove outdated IPs from the set
                while timestamps[start_idx] < timestamps[end_idx] - pd.Timedelta(days=window_interval):
                    ip_set.discard(ip_addresses[start_idx])
                    start_idx += 1
                
                # Add the current IP
                ip_set.add(ip_addresses[end_idx])
                no_null_like_ip_set = {ip for ip in ip_set if pd.notna(ip) and str(ip).strip()}
                count_list.append(len(no_null_like_ip_set))
            
            distinct_ip_counts.extend(count_list)
        
        df[f'customer_id_distinct_ip_{window_interval}d'] = distinct_ip_counts
        df = df.sort_values([date_col])
    return df


# In[] seasonality feature transform: add_date_features

def break_date_components(df, date_col='transaction_date'):
    df['transaction_date_day'], df['transaction_date_month'] = df[date_col].dt.day, df[date_col].dt.month
    return df


# In[] transaction related feature transform: add_days_of_established

def add_days_of_established(df, date_col:str='merchant_established_date'):
    df['established_period_in_day'] = (datetime.today() - df[date_col]).dt.days
    return df

def add_ma_trade_info_by_id(
        df, 
        date_col:str='transaction_date', 
        amt_col:str='amount',
        id_col:str='customer_id',
        window:list=[1,5,12,28,60],
        trade_info:list=['amount']
):
    '''Partial Windows'''
    for val in trade_info:
        if val not in ['amount', 'count']:
            raise Exception
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    
    for info in trade_info:
        for window_interval in window:
            all_id_ma_output = []
            # Process each customer separately
            for id, group in df.groupby(id_col):
                vals_to_cal_ma = []
                timestamps = group[date_col].tolist()
                amount = group[amt_col].tolist()
                
                start_idx = 0
                ma_output_list = []
                
                for end_idx in range(len(group)):
                    # Remove outdated value
                    while timestamps[start_idx] < timestamps[end_idx] - pd.Timedelta(days=window_interval):
                        vals_to_cal_ma.remove(amount[start_idx]) #FIXME cnt_list.remove(timestamps[start_idx])?
                        start_idx += 1
                    
                    # Add the current value
                    vals_to_cal_ma.append(amount[end_idx])
                    if info == 'amount':
                        ma_output_list.append(np.nanmean(vals_to_cal_ma)) #simple average
                    elif info == 'count':
                        ma_output_list.append(len(vals_to_cal_ma))
                
                all_id_ma_output.extend(ma_output_list)
            
            df[f'{id_col}_ma_{info}_{window_interval}d'] = all_id_ma_output

    df = df.sort_values([date_col])
    return df

# In[] Postcode feature transform: transform postcode to London or not

def postcode_to_LN_or_not(dataset):
    '''
    Transformer that can be applied on whole or after split dataset
    '''
    newly_engi_feats = set()
    ref_col = 'postcode_latest'
    inner_lon = ['E','EC','N','NW','SE','SW','W','WC']
    outer_lon = ['BR','CR','DA','EN','HA','IG','KT','RM','SM','TN','TW','UB','WD']
    brim_man_lon = inner_lon + outer_lon + ['B','M']
    dataset['prefix'] = dataset[ref_col].str.extract(r'^([A-Z]+)', expand=False)
    dataset['in_london'] = dataset['prefix'].mask(dataset['prefix'].isin(inner_lon + outer_lon), 1)
    dataset['in_london'] = dataset['in_london'].mask(~dataset['in_london'].isin([1]), 0)
    dataset['in_london'] = dataset['in_london'].astype(int)
    newly_engi_feats.add('in_london')
    dataset = dataset.drop([ref_col, 'prefix'], axis=1)
    return dataset


# In[8] IP behaviour feature transform: encode_cat_col
def onehot_encode_column(df, column_name, drop_first=False, prefix=None, dtype=int):
    """
    One-hot encode a specified column in a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the column to encode
    - column_name: name of the column to one-hot encode
    - drop_first: whether to drop the first category (to avoid multicollinearity)
    - prefix: prefix to use for the new columns (defaults to the original column name)
    - dtype: data type for the new columns (default is int)
    
    Returns:
    - DataFrame with the original column replaced by one-hot encoded columns
    """
    # Make a copy of the original DataFrame to avoid modifying it
    df = df.copy()
    
    # Use pandas get_dummies function to perform one-hot encoding
    dummies = pd.get_dummies(
        df[column_name],
        prefix=prefix if prefix else column_name,
        drop_first=drop_first,
        dtype=dtype
    )
    
    # Drop the original column and concatenate the new dummy columns
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    
    return df

def label_encode_column(df, column_name, inplace=False):
    """
    Label encode a specified column in a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the column to encode
    - column_name: name of the column to label encode
    - inplace: if True, modifies the DataFrame in place; otherwise returns a new DataFrame
    
    Returns:
    - DataFrame with the specified column label encoded (if inplace=False)
    - None (if inplace=True)
    """
    le = LabelEncoder()
    
    if not inplace:
        df = df.copy()
    
    df[column_name] = le.fit_transform(df[column_name])
    
    if not inplace:
        return df

# In[7] binning UK vs non_UK country code

def bin_uk_nonuk_country_code(df, country_code_col='IP_address_country_code', output_col='is_UK'):
    """
    Bin the country codes into UK and non-UK categories.
    
    Parameters:
    - df: pandas DataFrame
    - country_code_col: name of the column containing country codes
    - output_col: name of the new output column
    
    Returns:
    - DataFrame with the new binary column added
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Create a binary column based on the country code
    df[output_col] = df[country_code_col].apply(lambda x: 1 if x == 'GB' else (0 if x is not None else np.nan))
    
    return df


# In[8] feature combination

def combine_two_cols_with_null_check(
        df, 
        output_col='type_of_credit_card_used_with_mer_id', 
        str_col='type_of_credit_card_used', 
        id_col='store_card_merchant_id', 
        id_format=str, 
        separator='_',
        special_str_value="Store Credit Card"
):
    """
    Combine a string column and an integer column into a single string column,
    using the string value only when the integer value is null.
    
    Parameters:
    - df: pandas DataFrame
    - str_col: name of the string column
    - id_col: name of the integer column
    - output_col: name of the new output column
    - int_format: function to format the integer (default str)
    - separator: string to separate values when combining (default '_')
    
    Returns:
    - DataFrame with the new combined column added
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize the output column with string values
    df[output_col] = df[str_col].astype(str)
    
    # Where integer is not null, use the formatted integer value
    mask = df[id_col].notna()
    df.loc[mask, output_col] = df.loc[mask, str_col].apply(id_format).astype(str) #FIXME
    
    # Where both columns have values, combine them with separator
    both_mask = df[str_col].notna() & df[id_col].notna() & (df[str_col] == special_str_value)
    df.loc[both_mask, output_col] = (df.loc[both_mask, str_col].astype(str) + 
                                    separator + 
                                    df.loc[both_mask, id_col].apply(id_format).astype(str))
    
    return df

# In[9] add merchant card flags

def add_merchant_card_flags(cf_raw_df):
    """
    Adds merchant card related flag columns to the dataframe.
    
    Parameters:
    cf_raw_df (pd.DataFrame): Input dataframe containing credit card transaction data
    
    Returns:
    pd.DataFrame: Dataframe with added columns:
        - is_merchant_card: 1 if store credit card, else 0
        - is_merchant_mismatch: 1 if merchant_id matches store_card_merchant_id, else 0
        - is_merchant_card_and_mismatch: 1 if both above conditions are true, else 0
    """
    cf_raw_df = cf_raw_df.copy()
    
    cf_raw_df['is_merchant_card'] = (cf_raw_df['type_of_credit_card_used'] == 'Store Credit Card').astype(int)
    cf_raw_df['is_merchant_mismatch'] = (cf_raw_df['merchant_id'] == cf_raw_df['store_card_merchant_id']).astype(int)
    cf_raw_df['is_merchant_card_and_mismatch'] = (
        (cf_raw_df['is_merchant_card'] == 1) & 
        (cf_raw_df['is_merchant_mismatch'] == 1)
    ).astype(int)
    
    return cf_raw_df


# In[0]
%time
if __name__ == '__main__':
    cf_raw_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            'raw',
            'credit_fraud_data.csv'
        ),
    ) #_cleaned
    tgt = 'is_fraud'

    cf_raw_df['transaction_date'] = pd.to_datetime(cf_raw_df['transaction_date'], dayfirst=True, errors='coerce')
    for col in cf_raw_df.columns:
        cf_raw_df[col] = pd.to_datetime(cf_raw_df[col]) if '_date' in col else cf_raw_df[col]

    cf_raw_df = add_ip_country_code(cf_raw_df)
    cf_raw_df = add_ip_country_date(cf_raw_df)
    cf_raw_df = add_total_seconds_since_midnight(cf_raw_df)

    cf_raw_df = add_distinct_ip_window(cf_raw_df)
    cf_raw_df = add_ip_used_by_many(cf_raw_df)

    cf_raw_df = add_is_weekend(cf_raw_df)
    cf_raw_df = add_is_each_weekday(cf_raw_df)
    cf_raw_df = add_is_UK_bank_holiday(cf_raw_df)  # Add UK bank holiday feature
    #cf_raw_df = add_bank_holiday_by_country_code(cf_raw_df, date_col='country_code_date')  # Add bank holiday feature based on IP country code

    cf_raw_df = add_recent_ip_time_ago_by_customer(cf_raw_df) # Add time since last trade feature
    cf_raw_df = add_distinct_ip_window(cf_raw_df)  # Add distinct IP addresses in the last n days feature   
    cf_raw_df = add_days_of_established(cf_raw_df)
    cf_raw_df = add_ma_trade_info_by_id(cf_raw_df, trade_info=['amount', 'count'])
    #cf_raw_df['encoded_IP_address'] = encode_cat_col(cf_raw_df['IP_address'])

    #cf_raw_df = combine_two_cols_with_null_check(cf_raw_df)
    cf_raw_df = add_merchant_card_flags(cf_raw_df)

    cf_raw_df = add_total_seconds_since_midnight(cf_raw_df, date_col='country_code_date')
    cf_raw_df = decompose_date_components(cf_raw_df, date_col='transaction_date')

    cf_raw_df = onehot_encode_column(cf_raw_df, 'merchant_category')
    #cf_raw_df = onehot_encode_column(cf_raw_df, 'type_of_credit_card_used_with_mer_id') #FIXME

    cf_raw_df = label_encode_column(cf_raw_df, 'card_present_or_not')
    
    cf_raw_df = add_time_since_last_ip_per_customer(cf_raw_df)

    cf_raw_df = cf_raw_df.sort_values('transaction_date')

    cf_raw_df.to_csv(
        os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            'processed',
            'credit_fraud_data_transformed.csv'
        ),
        index=False
    )


# %%

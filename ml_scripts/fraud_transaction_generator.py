'''fraud_transaction_generator'''

'''to do
add fraud - fake merchant logic
credit card class, pos class
['amount']<1

roadmap:
unittest
add doc string
optimize iterrows too slow
'''

# In[0] setup
'''Setup libraries and params'''

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

import geoip2.database
import ipaddress
import socket
import struct


# In[1] Classes
'''Classes'''

class Utility():
    _24HRS_IN_SEC = 86400
    _ALL_COUNTRY_CODE = [
            'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR',
            'AS', 'AT', 'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE',
            'BF', 'BG', 'BH', 'BI', 'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ',
            'BR', 'BS', 'BT', 'BV', 'BW', 'BY', 'BZ', 'CA', 'CC', 'CD',
            'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN', 'CO', 'CR',
            'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM',
            'DO', 'DZ', 'EC', 'EE', 'EG', 'EH', 'ER', 'ES', 'ET', 'FI',
            'FJ', 'FK', 'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 'GF',
            'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GS',
            'GT', 'GU', 'GW', 'GY', 'HK', 'HM', 'HN', 'HR', 'HT', 'HU',
            'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR', 'IS', 'IT',
            'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN',
            'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK',
            'LR', 'LS', 'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME',
            'MF', 'MG', 'MH', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ',
            'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA',
            'NC', 'NE', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP', 'NR', 'NU',
            'NZ', 'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL', 'PM',
            'PN', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS',
            'RU', 'RW', 'SA', 'SB', 'SC', 'SD', 'SE', 'SG', 'SH', 'SI',
            'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SR', 'SS', 'ST', 'SV',
            'SX', 'SY', 'SZ', 'TC', 'TD', 'TF', 'TG', 'TH', 'TJ', 'TK',
            'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TV', 'TW', 'TZ', 'UA',
            'UG', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI',
            'VN', 'VU', 'WF', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW'
        ]

    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
        self.config_path = os.path.join(os.path.dirname(os.getcwd()), 'configurations')
        self.holiday_periods = pd.read_csv(os.path.join(self.config_path, 'holiday_periods.csv'))
        self.holiday_periods['holiday_starts'] = pd.to_datetime(self.holiday_periods['holiday_starts'], format='%d/%m/%Y')
        self.holiday_periods['holiday_ends'] = pd.to_datetime(self.holiday_periods['holiday_ends'], format='%d/%m/%Y')
        self._24HRS_IN_SEC = Utility._24HRS_IN_SEC
        self._path_to_IP_db = "GeoLite2-Country.mmdb"
        self._ALL_COUNTRY_CODE = Utility._ALL_COUNTRY_CODE
        self.major_country_code = [
            'US', 'CA', 'MX', 'BR', 'AR',
            'GB', 'FR', 'DE', 'IT', 'DK',
            'RU', 'ZA', 'SA', 'CN', 'JP',
            'KR', 'IN', 'ID', 'AU', 'TR',
            'ES', 'SE', 'LU', 'IE', 'PT',
            # Other significant economies
            'ES', 'NL', 'CH', 'SE', 'NO',
            'SG', 'MY', 'TH', 'VN', 'AE'
        ]

    def export_data_in_csv(self, data:pd.DataFrame, file_name:str):
        data.to_csv(os.path.join(self.data_path, f'{file_name}.csv'), index=False)

    def get_holiday_period(self, date_col:str, date_df:pd.DataFrame): #FIXME
        hoilday_period = list(
            date_df[date_col].apply(lambda x: 1 if (
                (x.month == 11 and x.day >= 24) or  # Black Friday and late November
                (x.month == 12 and x.day <= 31) or  # Christmas season
                (x.month == 1 and x.day <= 5)
            ) else 0)     # New Year spillover
        )
        return hoilday_period
    
    def gen_random_ipv4_from_country_code(
            self, 
            nullable: bool = False,
            country_code: str = 'random', 
            max_attempts: int = 10000,
            major_country: bool = False,
            max_retries: int = 3  # New parameter for max country code retries
        ):
        """
        Generate a random IPv4 address from a specified or random country code.
        
        Args:
            nullable: If True, may return None randomly
            country_code: 'random' or specific ISO country code
            max_attempts: Max IP generation attempts per country
            major_country: If True and country_code='random', uses only major countries
            max_retries: Max times to try with different country codes if IP generation fails
        """
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Determine country code
                if country_code == 'random':
                    if major_country:
                        country_code = random.choice(self.major_country_code)
                    else:
                        country_code = random.choice(self._ALL_COUNTRY_CODE)

                def _generate_random_ip():
                    """Generates a random public IPv4 address"""
                    while True:
                        ip = socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))
                        if not ipaddress.IPv4Address(ip).is_private:
                            return ip

                # Main execution flow
                with geoip2.database.Reader(self._path_to_IP_db) as reader:
                    for _ in range(max_attempts):
                        random_ip = _generate_random_ip()
                        try:
                            if reader.country(random_ip).country.iso_code == country_code:
                                if nullable:
                                    return random.choice([random_ip, None])
                                return random_ip
                        except:
                            continue

                # If we get here, all attempts failed for this country
                raise ValueError(f"Could not generate valid IP for {country_code} after {max_attempts} attempts")

            except ValueError:
                retry_count += 1
                if retry_count < max_retries:
                    # Reset country code to random for next attempt
                    country_code = random.choice(self.major_country_code)
                    continue
                raise  # Re-raise if max retries exceeded

        # This line should theoretically never be reached
        raise ValueError("Max retries exceeded for country code regeneration")
    
    def convert_sec_to_date_format(self, time_in_sec:int, start_date:str):
        date = pd.to_datetime(time_in_sec, unit='s', origin=start_date)
        return date
    
    def gen_transaction_time_sec_format(self, day_idx:int, daylight_center=False):
        #14400 = 4 hours
        tran_time_in_sec = int(np.random.normal(self._24HRS_IN_SEC/2, 14400)) if daylight_center else int(np.random.uniform(0, self._24HRS_IN_SEC))
        if 0 < tran_time_in_sec < self._24HRS_IN_SEC:
            tran_time_in_sec += day_idx * self._24HRS_IN_SEC
            return tran_time_in_sec
        else:
            return None
        
    def adjust_dates_to_holidays(
            self, 
            df:pd.DataFrame, 
            datetime_col:str, 
            holiday_periods:pd.DataFrame=None,
            start_date:str=None,
            adj_likelihood=[0.65, 0.35]
    ):
        """
        Adjusts dates in 'datetime' column if they fall within 30 days before/after a holiday period.
        Such dates are replaced with a random date from the holiday period.

        Args:
        df (pd.DataFrame): DataFrame containing a 'datetime' column.
        holiday_periods (list of tuples): List of (start_date, end_date) defining holiday periods.

        Returns:
        pd.DataFrame: DataFrame with adjusted dates.
        """
        df = df.copy()  # Avoid modifying the original DataFrame
        if holiday_periods is None:
            holiday_periods = self.holiday_periods
        holiday_periods['pre-holiday_consumption_starts'] = holiday_periods['holiday_starts'] - pd.Timedelta(days=30)

        for _, row in holiday_periods.iterrows():
            start_buffer = row["pre-holiday_consumption_starts"] - pd.Timedelta(days=30)
            end_buffer = row["holiday_ends"] + pd.Timedelta(days=30)

            # Find rows that fall within the buffer range
            date_to_adj = (df[datetime_col] >= start_buffer) & (df[datetime_col] <= end_buffer)

            # Replace these dates with a random date in the holiday period, based on adjustment likelihood
            adjustment_in_days = np.random.randint(0, (row["holiday_ends"] - row["pre-holiday_consumption_starts"]).days + 1)
            df.loc[date_to_adj, datetime_col] = df.loc[date_to_adj, datetime_col].apply(
                lambda x: 
                row["pre-holiday_consumption_starts"] + pd.Timedelta(days=adjustment_in_days) \
                    if row["pre-holiday_consumption_starts"] + pd.Timedelta(days=adjustment_in_days) >= pd.to_datetime(start_date) \
                        and random.choices([True, False], weights=adj_likelihood)[0] else x
            )         
            # assign random hour, mins, and sec to fraud trades, no time preference restriction for frauds
            df[datetime_col] += pd.to_timedelta(np.random.randint(0, 24, size=len(df)), unit='h')
            df[datetime_col] += pd.to_timedelta(np.random.randint(0, 60, size=len(df)), unit='m')
            df[datetime_col] += pd.to_timedelta(np.random.randint(0, 60, size=len(df)), unit='s')
        return df, date_to_adj
        
    def adjust_dates_to_weekends(
            self, 
            df:pd.DataFrame, 
            datetime_col:str, 
            adj_likelihood:float=0.2
    ):
        df = df.copy()
        # Identify non-weekend records
        non_weekend_mask = ~df[datetime_col].dt.weekday.isin([5, 6])  # Weekdays (Mon-Fri)

        # Randomly select 20% of non-weekend records
        change_mask = non_weekend_mask & (np.random.rand(len(df)) < adj_likelihood)

        # Function to adjust to the nearest weekend
        def shift_to_weekend(date):
            return date + pd.Timedelta(days=(5 - date.weekday()) if date.weekday() < 5 else 6 - date.weekday())

        # Apply the transformation
        df.loc[change_mask, datetime_col] = df.loc[change_mask, datetime_col].apply(shift_to_weekend)

        return df


class Customer():
    _NEXT_ID = 1
    _FULL_CUS_TBL = None
    UTIL = Utility()
    _CREDIT_CARD_CATEGORY = pd.read_csv(os.path.join(UTIL.config_path, 'credit_card_category.csv'))

    def __init__(self, random_state:int=None):
        if random_state != None:
            np.random.seed(random_state)
        self._customer_id = Customer._NEXT_ID
        Customer._NEXT_ID += 1
        self._income = np.random.choice(['low', 'medium', 'high'])
        self._set_income_related_attributes()
        self._location_x = np.random.uniform(0,100)
        self._location_y = np.random.uniform(0,100)
        self._std_trans_amt = self._mean_trans_amt/2 #FIXME
        self._nb_trans_per_day = np.random.uniform(0,5)
        self._ip_table = self._gen_ip_info_tbl()
        self._credit_card_owned = self._gen_credit_card_owned()
        Customer._FULL_CUS_TBL = self.gen_customer_tbl() if not isinstance(Customer._FULL_CUS_TBL, pd.DataFrame)\
             else pd.concat([Customer._FULL_CUS_TBL, self.gen_customer_tbl()])

    @classmethod
    def get_customer_tbl(cls):
        return cls._FULL_CUS_TBL

    @classmethod
    def get_credit_card_category(cls):
        return cls._CREDIT_CARD_CATEGORY

    @classmethod
    def reset_class_var(cls):
        cls._NEXT_ID = 1
        cls._FULL_CUS_TBL = None

    @property
    def income(self):
        return self._income

    @income.setter
    def income(self, val):
        print(f"Setting value to {val}")
        self._income = val
        self._set_income_related_attributes()

    @property
    def ip_table(self):
        return self._ip_table

    def _gen_ip_info_tbl(self):
        data = []
        primary_ip = self.UTIL.gen_random_ipv4_from_country_code(country_code='GB')
        second_ip = self.UTIL.gen_random_ipv4_from_country_code(
            country_code='random', major_country=random.choice([True, False]), nullable=True
        )
        third_ip = None if second_ip == None else self.UTIL.gen_random_ipv4_from_country_code(country_code='random', nullable=True)
        data.append([primary_ip, second_ip, third_ip])

        for idx, val in enumerate(data[0]):
            if val:
                if idx == 0:
                    ip_address_prob = [1, 0, 0]
                elif idx == 1:
                    first_prob = np.round(np.random.uniform(0.7, 0.9), decimals=2)
                    second_prob = np.round(1-first_prob, decimals=2)
                    ip_address_prob = [first_prob, second_prob, 0]
                elif idx == 2:
                    first_prob = np.round(np.random.uniform(0.4, 0.5), decimals=2)
                    second_prob = np.round(np.random.uniform(0.2, 0.4), decimals=2)
                    third_prob = np.round(1-first_prob-second_prob, decimals=2)
                    ip_address_prob = [first_prob, second_prob, third_prob]
        data[0].append(ip_address_prob)
        ip_tbl = pd.DataFrame(
            data=data,
            columns=['primary_IP_address',
                     'second_IP_address',
                     'third_IP_address',
                     'IP_address_probabilities']
        )
        return ip_tbl

    def _gen_credit_card_owned(self):
        credit_card_category = Customer.get_credit_card_category()['card_category'].values
        nb_credit_card_owned = random.randint(1, len(credit_card_category))
        credit_card_owned = random.sample(list(credit_card_category), nb_credit_card_owned)
        return credit_card_owned

    def _set_income_related_attributes(self):
        if self._income == 'low':
            self._mean_trans_amt = np.random.uniform(2, 200)
            self._online_offline_ratio = [0.5, 0.5]
        elif self._income == 'medium':
            self._mean_trans_amt = np.random.uniform(200, 500)
            self._online_offline_ratio = [0.75, 0.25]
        elif self._income == 'high':
            self._mean_trans_amt = np.random.uniform(500, 1000)
            self._online_offline_ratio = [0.75, 0.25]

    def gen_customer_tbl(self):
        data = [[self._customer_id, 
                 self._location_x,
                 self._location_y,
                 self._mean_trans_amt,
                 self._std_trans_amt,
                 self._nb_trans_per_day,
                 self._credit_card_owned,
                 self._online_offline_ratio]]
        cus_tbl = pd.DataFrame(
            data=data,
            columns=[
                'customer_id', 
                'location_x', 
                'location_y', 
                'mean_transaction_amount',
                'std_amount',
                'nb_transaction_per_day',
                'credit_card_owned',
                'online_offline_ratio'
            ]
        )
        cus_tbl = cus_tbl.merge(self.ip_table, left_index=True, right_index=True)
        return cus_tbl
    

class Merchant():
    _NEXT_MERCHANT_ID = 1
    _NEXT_POS_ID = 1
    _FULL_MERCHANT_TBL = None
    UTIL = Utility()

    def __init__(
            self, 
            random_state:int=None, 
            merchant_category_filename:str='merchant_category_codes',
            fake_or_not:bool=False     
    ):
        if random_state != None:
            np.random.seed(random_state)
        self._merchant_id = Merchant._NEXT_MERCHANT_ID
        Merchant._NEXT_MERCHANT_ID += 1
        self.fake_or_not = fake_or_not
        self._established_date = self.random_date()
        self._merchant_size = random.choice(['small', 'medium', 'large'])
        self._pos_tbl = self._gen_pos_terminal_tbl()
        self._mcc_tbl = pd.read_csv(os.path.join(Merchant.UTIL.config_path, f'{merchant_category_filename}.csv'))
        self._merchant_category_info = self.mcc_tbl.sample(n=1).values.flatten()
        Merchant._FULL_MERCHANT_TBL = self.gen_merchant_tbl() if not isinstance(Merchant._FULL_MERCHANT_TBL, pd.DataFrame)\
             else pd.concat([Merchant._FULL_MERCHANT_TBL, self.gen_merchant_tbl()])
    
    @classmethod
    def get_merchant_tbl(cls):
        return cls._FULL_MERCHANT_TBL

    @classmethod
    def reset_class_var(cls):
        cls._NEXT_MERCHANT_ID = 1
        cls._NEXT_POS_ID = 1
        cls._FULL_MERCHANT_TBL = None

    @property
    def pos_tbl(self):
        return self._pos_tbl

    @property
    def mcc_tbl(self):
        return self._mcc_tbl

    @property
    def merchant_id(self):
        return self._merchant_id

    def _gen_store_card_issued_or_not(self):
        if self.fake_or_not:
            return 'N'
        elif self._merchant_category_info[1]=='Grocery Stores/Supermarkets' and self._merchant_size=='large':
            store_card_issued_probability = [0.9, 0.1]
        elif self._merchant_category_info[1]=='Grocery Stores/Supermarkets' and self._merchant_size=='medium':
            store_card_issued_probability = [0.6, 0.4]
        else:
            store_card_issued_probability = [0.2, 0.8]
        is_store_card_issued = random.choices(['Y', 'N'], weights=store_card_issued_probability, k=1)[0]
        return is_store_card_issued

    def gen_merchant_tbl(self):
        data = []
        is_store_card_issued = self._gen_store_card_issued_or_not()
        for pos_id in self._pos_tbl['pos_id']:
            data.append(
                [self._merchant_id, 
                self._established_date,
                pos_id,
                self._merchant_category_info[0],
                self._merchant_category_info[1],
                is_store_card_issued,
                ]
            )

        mer_tbl = pd.DataFrame(data=data,
                               columns=['merchant_id', 
                                        'established_date',
                                        'pos_id',
                                        'MCCs',
                                        'merchant_category',
                                        'is_store_card_issued',])

        mer_tbl = mer_tbl.merge(self._pos_tbl, how='left', on='pos_id')
        return mer_tbl
    
    def _gen_pos_terminal_tbl(self):
        if self.fake_or_not == True:
            nb_pos = 1
        elif self._merchant_size == 'large':
            nb_pos = int(np.random.uniform(6,10))
        elif self._merchant_size == 'medium':
            nb_pos = int(np.random.uniform(3,5))
        elif self._merchant_size == 'small':
            nb_pos = int(np.random.uniform(1,2))

        pos_data = {'pos_id': [], 'location_x': [], 'location_y': []}
        for i in range(nb_pos):
            pos_data['pos_id'].append(Merchant._NEXT_POS_ID)
            Merchant._NEXT_POS_ID += 1
            pos_data['location_x'].append(np.random.uniform(0,100))
            pos_data['location_y'].append(np.random.uniform(0,100))

        pos_tbl = pd.DataFrame(data=pos_data)
        return pos_tbl

    @staticmethod
    def random_date(start_date=datetime(1900, 1, 1), end_date=datetime.today()):
        """
        Generate a random date between start_date and end_date.

        :param start_date: datetime object representing the start date
        :param end_date: datetime object representing the end date
        :return: random datetime object
        """
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randint(0, days_between_dates)
        return start_date + timedelta(days=random_number_of_days)

    @staticmethod
    def get_merchant_unique_id_tbl(target_col:str, merchant_tbl:pd.DataFrame):
        return merchant_tbl[['merchant_id', target_col]].groupby(['merchant_id']).agg(lambda x: x.mode().iloc[0]) #drop duplicates?


class Transaction(): #FIXME random state
    _24HRS_IN_SEC = 86400
    UTIL = Utility()

    def __init__(
        self, 
        start_date="2023-01-01", 
        time_period_in_day:int=365,
        blacklist_ip:str='blacklist_IP', 
        fake_merchant_nb:int=0
    ):
        self._unusual_merchant_id = []
        self._merchant_tbl = self.add_fake_merchant(fake_merchant_nb)
        self._customer_tbl = self._assign_merchant_id_to_store_credit_card(Customer.get_customer_tbl())
        self._credit_card_category = Customer.get_credit_card_category()
        self._start_date = start_date
        self._time_period_in_day = time_period_in_day
        self._24HRS_IN_SEC = Transaction._24HRS_IN_SEC
        self._blacklist_ip_tbl = pd.read_csv(os.path.join(Transaction.UTIL.config_path, f'{blacklist_ip}.csv'))
        if self.customer_tbl is None or self.merchant_tbl is None:
            raise ValueError("An instance of Customer and Merchant class must be initialized first.")

    @property
    def customer_tbl(self) -> pd.DataFrame:
        return self._customer_tbl
    
    @property
    def merchant_tbl(self) -> pd.DataFrame:
        return self._merchant_tbl

    @property
    def blacklist_ip_tbl(self) -> pd.DataFrame:
        return self._blacklist_ip_tbl

    @property
    def time_period_in_day(self):
        return self._time_period_in_day

    @time_period_in_day.setter
    def time_period_in_day(self, val:int):
        print(f"Setting time_period_in_day to {val}")
        self._time_period_in_day = val

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, val:str):
        print(f"Setting start_date to {val}")
        self._start_date = val

    @property
    def credit_card_category(self) -> pd.DataFrame:
        return self._credit_card_category

    @classmethod
    def reset_class_var(cls):
        cls._24HRS_IN_SEC = 86400

    def _gen_transaction_amt(
            self, 
            customer_id:int, 
            fraud_type:str=None
    ) -> float:
        
        if fraud_type=='cnp_testing':
            return np.random.uniform(10, 30)
        elif fraud_type in [None, 'cnp_monetization', 'cp_cloning']:
            customer_profile = self.customer_tbl[self.customer_tbl['customer_id']==customer_id]
            amount = np.random.normal(customer_profile['mean_transaction_amount'].item(), customer_profile['std_amount'].item())
                            
            # If amount negative, draw from a uniform distribution
            amount = np.random.uniform(0, customer_profile['mean_transaction_amount'].item()*2) if amount < 0 else amount
            amount = np.round(amount, decimals=2)
        else:
            raise ValueError("fraud_type must be either 'cnp_testing', 'cnp_monetization', 'cp_cloning', or None.")
        
        if fraud_type in ['cnp_monetization', 'cp_cloning']:
            amount = min(amount*3, random.uniform(2500, 3500))

        return amount

    def _gen_transaction_nb(self, customer_id:int):
        customer_profile = self.customer_tbl[self.customer_tbl['customer_id']==customer_id]
        nb_transaction = np.random.poisson(customer_profile['nb_transaction_per_day'].item())
        return nb_transaction

    def _gen_merchant_traded(self): #TODO merchant category effect, not random
        return random.choice(self.merchant_tbl['merchant_id'].values)

    def _gen_pos_id_of_merchant_traded(self, customer_id:int, merchant_id:int, radius:int=60, fraud_type:str=None):
        if fraud_type in ['cnp_testing', 'cnp_monetization']:
            return [None], [None]

        relevant_mer = self.merchant_tbl.loc[self.merchant_tbl['merchant_id']==merchant_id]
        mer_loc_info = relevant_mer[['location_x', 'location_y']].values
        cus_loc_info = self.customer_tbl.loc[self.customer_tbl['customer_id']==customer_id, 
                                              ['location_x', 'location_y']].values
        squared_diff = np.square(cus_loc_info - mer_loc_info)
        distance = np.sqrt(np.sum(squared_diff, axis=1))
        nearby_pos_idx, far_away_pos_idx = np.where(distance < radius)[0], np.where(distance >= radius)[0]
        nearby_pos_id = relevant_mer.iloc[nearby_pos_idx,:]['pos_id'].values
        nearby_pos_id = [None] if nearby_pos_id.size == 0 else nearby_pos_id
        far_away_pos_id = relevant_mer.iloc[far_away_pos_idx,:]['pos_id'].values
        far_away_pos_id = [None] if far_away_pos_id.size == 0 else far_away_pos_id

        return nearby_pos_id, far_away_pos_id

    def plot_location_graph(self): # TODO
        pass

    def _assign_merchant_id_to_store_credit_card(self, customer_tbl):
        '''assign merchant id to store credit card randomly based on merchant id that issue store credit card'''
        mer_card_info = Merchant.get_merchant_unique_id_tbl(target_col='is_store_card_issued', merchant_tbl=self.merchant_tbl)
        # no need to assign fake merchant as they wont have store credit card
        merchant_id_which_issue_card = mer_card_info[mer_card_info['is_store_card_issued'] == 'Y'].index.tolist()
        
        # if no traded merchant issued card, assign a random non-traded merchant id to Store Card
        if not merchant_id_which_issue_card:
            merchant_id_which_issue_card = [int(np.random.uniform(mer_card_info.index.values.max()+1, 100))]

        def apply_condition(group):
            group['store_card_merchant_id'] = group['credit_card_owned'].apply(lambda x: random.choice(merchant_id_which_issue_card) if 'Store Credit Card' in x else None)
            return group
        customer_tbl = customer_tbl.groupby('customer_id', group_keys=False).apply(apply_condition)

        return customer_tbl

    def _gen_credit_card_likely_used(self, customer_id, merchant_id, fraud_trade:bool=False): #FIXME
        card_owned = self.customer_tbl[self.customer_tbl['customer_id']==customer_id]['credit_card_owned'].values[0]
        # if fraud_trade, use random card
        if fraud_trade == True:
            card_used = random.choices(card_owned)[0]
        else:
            # category of merchant
            mer_category = Merchant.get_merchant_unique_id_tbl('merchant_category', self.merchant_tbl).loc[merchant_id, 'merchant_category']
            # card category of customer prefered based on merchant category
            card_likely_used = self.credit_card_category[
                (self.credit_card_category['prefer_consumption_category_1'] == mer_category) | \
                (self.credit_card_category['prefer_consumption_category_2'] == mer_category)
            ]
            card_likely_used = card_likely_used['card_category'].values

            if_customer_not_holding_merchant_card = self.customer_tbl[(self.customer_tbl['customer_id']==customer_id) & (self.customer_tbl['store_card_merchant_id']==merchant_id)].empty

            if 'Store Credit Card' in card_owned and if_customer_not_holding_merchant_card==False: #if False, record exist, card of merchant id being hold
                card_used = 'Store Credit Card'
            else:
                card_used = random.choices(np.intersect1d(card_owned, card_likely_used)) \
                            if np.isin(card_owned, card_likely_used).any() \
                            else random.choices(card_owned)
                card_used = card_used[0]
        return card_used

    def _gen_ip_address_used(self, customer_id, fraud_trade:bool=False):
        if fraud_trade == True:
            ip_address = self.UTIL.gen_random_ipv4_from_country_code()
            random_blacklisted_ip = random.choices(self.blacklist_ip_tbl.values)[0][0]
            ip_address = random.choices([ip_address, random_blacklisted_ip], weights=[0.8, 0.2], k=1)[0]
            return ip_address
        ip_table = self.customer_tbl[['customer_id', 'primary_IP_address', 'second_IP_address', 'third_IP_address', 'IP_address_probabilities']]
        ip_table = ip_table[ip_table['customer_id']==customer_id]
        ip_choices = ip_table.loc[:, ['primary_IP_address', 'second_IP_address', 'third_IP_address']].values.flatten()
        ip_probabilies = ip_table['IP_address_probabilities'][0]
        ip_address = random.choices(ip_choices, weights=ip_probabilies, k=1)[0]
        return ip_address

    def _gen_daily_transactions_per_day(
        self, 
        customer_id:int, 
        day_idx:int, # a bit explanation here
        fraud_trade:bool=False,
        nb_fraud_trades:int=None,
        fraud_type:str=None,
        specific_card_used=None
    ): #FIXME
        '''transaction generating logic'''
        if fraud_trade==True and fraud_type not in ['cnp_testing', 'cnp_monetization', 'cp_cloning']:
            raise ValueError("If fraud_trade is provided, fraud_type must be provided as well.")
        
        nb_transaction = self._gen_transaction_nb(customer_id) if fraud_trade==False else nb_fraud_trades
        if nb_transaction <= 0:
            return

        customer_data = self.customer_tbl[self.customer_tbl['customer_id']==customer_id]
        daily_transaction_info, card_used, datetime_in_sec = [], specific_card_used, None

        if nb_transaction > 0:
            for nb in range(nb_transaction):
                # trade during daylight
                if fraud_trade==False:
                    datetime_in_sec = self.UTIL.gen_transaction_time_sec_format(day_idx=day_idx, daylight_center=True)
                    customer_online_offline_ratio = customer_data['online_offline_ratio'].values[0]
                    online_or_offline = random.choices(['online', 'offline'], weights=customer_online_offline_ratio, k=1)[0]
                elif fraud_type=='cp_cloning':
                    online_or_offline = 'offline'
                    datetime_in_sec = self.UTIL.gen_transaction_time_sec_format(day_idx=day_idx, daylight_center=False)
                # fraud trades may trade at night
                elif (fraud_type=='cnp_testing' and datetime_in_sec is None) or fraud_type=='cnp_monetization':
                    online_or_offline = 'online'
                    datetime_in_sec = self.UTIL.gen_transaction_time_sec_format(day_idx=day_idx, daylight_center=False)
                # testing fraud trades trade closely with each other
                elif (fraud_type=='cnp_testing' and datetime_in_sec is not None):
                    online_or_offline = 'online'
                    datetime_in_sec = datetime_in_sec + int(random.uniform(10, 60*30)) #60 sec 30mins
                
                if datetime_in_sec:
                    transaction_date = self.UTIL.convert_sec_to_date_format(datetime_in_sec, self._start_date)
                    traded_merchant_id = self._gen_merchant_traded()
                    mer_established_date = Merchant.get_merchant_unique_id_tbl('established_date', self.merchant_tbl).loc[traded_merchant_id, 'established_date']
                    mer_MCCs = Merchant.get_merchant_unique_id_tbl('MCCs', self.merchant_tbl).loc[traded_merchant_id, 'MCCs']
                    mer_category = Merchant.get_merchant_unique_id_tbl('merchant_category', self.merchant_tbl).loc[traded_merchant_id, 'merchant_category']
                    
                    if online_or_offline=='online':
                        pos_id = None
                        card_present_or_not = 'card not present'
                        ip_address = self._gen_ip_address_used(customer_id, fraud_trade=fraud_trade)
                    elif online_or_offline=='offline' and fraud_trade==False:
                        nearby_pos_id, _ = self._gen_pos_id_of_merchant_traded(customer_id, traded_merchant_id, fraud_type=fraud_type)
                        pos_id = random.choice(nearby_pos_id)
                        # turn to online if no nearby pos
                        card_present_or_not = 'card present' if nearby_pos_id[0] != None else 'card not present'
                        ip_address = None if card_present_or_not == 'card present' else self._gen_ip_address_used(customer_id, fraud_trade=False)
                    elif online_or_offline=='offline' and fraud_type=='cp_cloning':
                        nearby_pos_id, far_away_pos_id = self._gen_pos_id_of_merchant_traded(customer_id, traded_merchant_id, fraud_type=fraud_type)
                        pos_id = random.choice(far_away_pos_id) if far_away_pos_id[0] != None else random.choice(nearby_pos_id)
                        card_present_or_not = 'card present'
                        ip_address = None

                    if (fraud_trade==True and card_used is None and fraud_type in ['cnp_testing', 'cp_cloning']) or fraud_trade==False: #Fixed compromised card
                        card_used = self._gen_credit_card_likely_used(customer_id, traded_merchant_id, fraud_trade=fraud_trade)
                    elif fraud_trade==True and fraud_type=='cnp_monetization' and card_used is None:
                        raise ValueError('card_used must not be None for monetization phase, should be consistent with card used in testing phase')
                    
                    daily_transaction_info.append(
                        [
                            transaction_date,
                            customer_id, 
                            self._gen_transaction_amt(customer_id, fraud_type=fraud_type), 
                            datetime_in_sec, 
                            day_idx + 1,
                            traded_merchant_id,
                            pos_id,
                            mer_established_date,
                            mer_MCCs,
                            mer_category,
                            ip_address,
                            card_used,
                            card_present_or_not
                        ]
                    )
        transaction_df = pd.DataFrame(
            data = daily_transaction_info, 
            columns = ['transaction_date','customer_id', 'amount','date_in_sec', 'No._of_day', 
                        'merchant_id', 'pos_id', 'merchant_established_date', 'MCCs', 
                        'merchant_category', 'IP_address', 'type_of_credit_card_used', 'card_present_or_not']
        )
        return transaction_df

    def gen_transaction_tbl_per_customer(self, customer_id:int, time_period_in_day:int):
        daily_transaction_tbl = []
        for day_idx in range(time_period_in_day):
            daily_transaction_tbl.append(self._gen_daily_transactions_per_day(customer_id, day_idx))

        transaction_tbl_per_customer = pd.concat(daily_transaction_tbl)
        return transaction_tbl_per_customer

    def gen_transaction_tbl( #TODO to optimize, e.g. weekend_adj param
        self, 
        with_frauds:bool=False,
        nb_test_fraud_per_day:int=int(random.uniform(3,6)),
        cnp_compromised_customer_nb_per_day:int=2,
        cp_compromised_customer_nb_per_day:int=1, #cp rarer than cnp
        nb_of_monetizating_per_day:int=1,
        monetizating_days:int=14,
        test_to_monetize_gap_in_days:int=30,
        consumption_season_adj:bool=False,
    ):
        transaction_df = self.customer_tbl.groupby('customer_id').apply(
            lambda x: self.gen_transaction_tbl_per_customer(
                x['customer_id'].item(), time_period_in_day=self.time_period_in_day
            )
        ).reset_index(drop=True)
        # weekend tendency adjustments
        transaction_df = self.UTIL.adjust_dates_to_weekends(
            df=transaction_df, 
            datetime_col='transaction_date', 
            adj_likelihood=0.2
        ).reset_index(drop=True)
        # refresh No._of_day, date in sec
        transaction_df = self._refresh_time_derived_col(transaction_df)

        if with_frauds == True:
            cnp_testing_frauds_records = self._gen_test_fraud_records(
                transaction_df, 
                nb_fraud_trades=nb_test_fraud_per_day, 
                compromised_customer_nb_per_day=cnp_compromised_customer_nb_per_day,
                fraud_type='cnp_testing'
            )
            test_fraud_id_n_card_type_tbl = self._gen_testing_fraud_id_n_card_tbl(cnp_testing_frauds_records)
            cnp_monetization_frauds_records, _ = self._gen_monetization_fraud_record_n_tbl(
                fraud_type='cnp_monetization',
                test_fraud_id_n_card_type_tbl=test_fraud_id_n_card_type_tbl,
                nb_of_monetizating_per_day=nb_of_monetizating_per_day,
                monetizating_days=monetizating_days,
                test_to_monetize_gap_in_days=test_to_monetize_gap_in_days,
                consumption_season_adj=consumption_season_adj,
                weekend_adj=True
            )
            cp_cloning_frauds_records, _ = self.gen_cp_fraud_card_cloning(
                transaction_df=transaction_df,
                nb_fraud_trades=int(random.uniform(1,2)),
                compromised_customer_nb_per_day=cp_compromised_customer_nb_per_day,
                consumption_season_adj=consumption_season_adj,
                holiday_fraud_prob=[0.65, 0.35]
            )
            cnp_testing_frauds_records['is_fraud'], cnp_monetization_frauds_records['is_fraud'] = 1, 1
            cp_cloning_frauds_records['is_fraud'] = 1
            transaction_df['is_fraud'] = 0
            transaction_df = pd.concat(
                [
                    transaction_df, 
                    cnp_testing_frauds_records, 
                    cnp_monetization_frauds_records, 
                    cp_cloning_frauds_records
                ]
            )
        transaction_df = transaction_df.merge(self.customer_tbl[['customer_id', 'store_card_merchant_id']], on='customer_id', how='left') #add back store_card_merchant_id
        transaction_df.sort_values('transaction_date', inplace=True)
        transaction_df.reset_index(drop=True, inplace=True)
        
        # remove transactions after the user defined time period
        transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'], dayfirst=True, errors='coerce')
        correct_latest_month = (transaction_df['transaction_date'].min() + pd.to_timedelta(self.time_period_in_day, unit="D")).month
        correct_latest_year = (transaction_df['transaction_date'].min() + pd.to_timedelta(self.time_period_in_day, unit="D")).year
        if transaction_df['transaction_date'].max().month > correct_latest_month or transaction_df['transaction_date'].max().year > correct_latest_year:
            transaction_df = transaction_df[
                (transaction_df['transaction_date'].dt.year < correct_latest_year) |
                ((transaction_df['transaction_date'].dt.year == correct_latest_year) & 
                (transaction_df['transaction_date'].dt.month <= correct_latest_month))
            ]
        return transaction_df

    def add_fake_merchant(self, nb_fake_merchant:int=1):
        for nb in range(nb_fake_merchant):
            unusual_merchant = Merchant(fake_or_not=True)
            self._unusual_merchant_id.append(unusual_merchant.merchant_id)
        merchant_tbl = Merchant.get_merchant_tbl()
        return merchant_tbl

    def gen_cp_fraud_card_cloning(
        self, 
        transaction_df:pd.DataFrame,
        nb_fraud_trades:int=int(random.uniform(1,2)),
        compromised_customer_nb_per_day:int=1,
        consumption_season_adj:bool=False,
        holiday_fraud_prob:list=[0.65, 0.35]
    ):
        cloning_frauds_records = self._gen_test_fraud_records(
            transaction_df, 
            nb_fraud_trades=nb_fraud_trades, 
            compromised_customer_nb_per_day=compromised_customer_nb_per_day,
            fraud_type='cp_cloning',
            holiday_fraud_prob=holiday_fraud_prob
        )
        fraud_id_n_card_type_tbl = self._gen_testing_fraud_id_n_card_tbl(cloning_frauds_records)
        monetization_frauds_records, monetization_fraud_tbl = self._gen_monetization_fraud_record_n_tbl(
            fraud_type='cp_cloning',
            test_fraud_id_n_card_type_tbl=fraud_id_n_card_type_tbl, 
            nb_of_monetizating_per_day=1,
            monetizating_days=7,
            test_to_monetize_gap_in_days=1,
            consumption_season_adj=consumption_season_adj,
            weekend_adj=True
        )
        return monetization_frauds_records, monetization_fraud_tbl

    def gen_cnp_fraud_comprimsed_merchant(self, transaction_df:pd.DataFrame): #TODO
        # fraud logic 2: Fake QR code
        # fake merchant id
        # select established date early records, replace with fake
        pass

    def _gen_monetization_fraud_record_n_tbl(
            self,
            fraud_type:str,
            test_fraud_id_n_card_type_tbl:pd.DataFrame,
            nb_of_monetizating_per_day:int=1,
            monetizating_days:int=7,
            test_to_monetize_gap_in_days:int=30,
            consumption_season_adj:bool=False,
            weekend_adj:bool=True
    ):
        test_fraud_id_n_card_type_tbl = test_fraud_id_n_card_type_tbl.copy()
        test_fraud_id_n_card_type_tbl['num of day - monetizating start'] = test_fraud_id_n_card_type_tbl['num of day - testing'] + test_to_monetize_gap_in_days
        test_fraud_id_n_card_type_tbl['num of day - monetizating end'] = test_fraud_id_n_card_type_tbl['num of day - monetizating start'] + monetizating_days
        
        monetization_fraud_info, monetization_frauds_records = [], []
        for _, row in test_fraud_id_n_card_type_tbl.iterrows():
            num_samples = random.randint(1, 5)
            for _ in range(num_samples):
                day_idx_of_fraud = random.randint(row['num of day - monetizating start'], row['num of day - monetizating end'])
                if day_idx_of_fraud+1 < self.time_period_in_day:
                    monetization_fraud_info.append({
                        "No._of_day": day_idx_of_fraud,
                        "customer_id": row["customer_id"],
                        "type_of_credit_card_used": row["type_of_credit_card_used"],
                    })
                    monetization_frauds_records.append(
                        self._gen_daily_transactions_per_day(
                                row["customer_id"], 
                                day_idx_of_fraud, 
                                True, 
                                nb_of_monetizating_per_day,
                                fraud_type,
                                row["type_of_credit_card_used"]
                        )
                    )
        monetization_fraud_tbl_no_date_adj = pd.DataFrame(monetization_fraud_info).sort_values('No._of_day') #FIXME KeyError: 'No._of_day'
        monetization_frauds_records = pd.concat(monetization_frauds_records).sort_values('transaction_date')
        # adjusted for weekend
        if weekend_adj == True:
            monetization_frauds_records = self.UTIL.adjust_dates_to_weekends(
                df=monetization_frauds_records, 
                datetime_col='transaction_date', 
                adj_likelihood=0.3
            )
            self._refresh_time_derived_col(monetization_frauds_records)
        # adjusted for seasonality
        if consumption_season_adj == True:
            monetization_frauds_records, _ = self.UTIL.adjust_dates_to_holidays(
                monetization_frauds_records, 
                'transaction_date',
                start_date=self._start_date
            )
            self._refresh_time_derived_col(monetization_frauds_records)
            
        return monetization_frauds_records, monetization_fraud_tbl_no_date_adj

    def _refresh_time_derived_col(self, df):
        # refresh No._of_day, date in sec
        df['No._of_day'] = (df['transaction_date'] - pd.to_datetime(self._start_date, format='%Y-%m-%d')).dt.days + 1
        df['date_in_sec'] = df['transaction_date'].apply(lambda x: int((x - pd.Timestamp(self._start_date)).total_seconds()))
        return df

    def _gen_testing_fraud_id_n_card_tbl(
            self, 
            testing_frauds_records:pd.DataFrame,
        ): # fraud logic 3: identity fraud
        '''generate fraud records for testing phase'''

        test_fraud_id_n_card_type_tbl = testing_frauds_records.groupby('No._of_day')
        test_fraud_id_n_card_type_tbl = pd.DataFrame(
            data = test_fraud_id_n_card_type_tbl.apply(
                lambda x: x.groupby('customer_id')['type_of_credit_card_used'].agg(
                    lambda x: x.mode().iloc[0]
                ),
            )).reset_index(drop=False)
        test_fraud_id_n_card_type_tbl.columns = [
                'num of day - testing', 
                'customer_id', 
                'type_of_credit_card_used', 
        ]   
        return test_fraud_id_n_card_type_tbl
        
    def _gen_test_fraud_records(
            self, 
            transaction_df:pd.DataFrame,
            nb_fraud_trades:int=1,
            compromised_customer_nb_per_day:int=1,
            fraud_type:str=None,
            holiday_fraud_prob:list=[0.8, 0.2]
        ): 
        if fraud_type not in ['cnp_testing', 'cp_cloning']:
            raise ValueError("Please provide a valid fraud type: 'cnp_testing', 'cp_cloning'")
        holiday_period = Transaction.UTIL.get_holiday_period('transaction_date', transaction_df) # increase prob of both test and compromise
        daily_fraud_records = []

        # for everyday that has transaction
        for idx, day in enumerate(range(transaction_df['No._of_day'].max())):
            # check if it is holiday
            holiday_or_not, holiday_fraud_prob = holiday_period[idx], holiday_fraud_prob
            # randomly pick 1-2 customer id to compromise
            compromised_customer_id = random.choices(
                transaction_df['customer_id'].unique(), 
                k=compromised_customer_nb_per_day
            )
            compromised_customer_id = list(set(compromised_customer_id))
            # each compromised customer id, generate fraud records based on probability, if holiday, increase fraud prob
            for customer_id in compromised_customer_id:
                fraud_occur_or_not = random.choices([True, False])[0] if holiday_or_not==0 else \
                    random.choices([True, False], weights=holiday_fraud_prob)[0]
                if fraud_occur_or_not == 1:
                    daily_fraud_records.append(self._gen_daily_transactions_per_day(
                        customer_id, 
                        day, 
                        True,
                        nb_fraud_trades = nb_fraud_trades,
                        fraud_type=fraud_type)
                    )
        test_frauds_records = pd.concat(daily_fraud_records)
        return test_frauds_records


# In[2] function to generate dataset
'''Function to generate dataset'''
def gen_credit_fraud_dataset(n_customer:int, n_merchant:int, n_days:int, start_date:str):
    # generate customer profiles table
    for nb in range(n_customer):
        cus_cls = Customer(nb)
        del cus_cls
    # generate merchant profiles table
    for nb in range(n_merchant):
        mer_cls = Merchant(nb)
        del mer_cls
    # generate transaction table
    transaction_cls = Transaction(start_date, n_days)
    transaction_df = transaction_cls.gen_transaction_tbl(
        with_frauds=True, 
        consumption_season_adj=True
    )

    return transaction_df


# In[3] main function
'''Main Function'''

def main(
        n_customer:int, 
        n_merchant:int, 
        n_days:int, 
        start_date:str, 
        dataset_name:str, 
        reset=False
):
    util = Utility()
    transaction_df = gen_credit_fraud_dataset(n_customer, n_merchant, n_days, start_date)
    util.export_data_in_csv(transaction_df, dataset_name)
    if reset == True:
        Customer.reset_class_var()
        Merchant.reset_class_var()
        Transaction.reset_class_var()


# In[4]
if __name__ == '__main__':
    main(250, 30, 730, "2022-11-01", 'credit_fraud_data', True)
    #main(500, 50, 360, "2022-11-01", 'credit_fraud_data', True)
    #main(10, 5, 730, "2022-11-01", 'credit_fraud_data', True)

# %%

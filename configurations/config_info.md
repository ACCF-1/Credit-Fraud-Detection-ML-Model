# Config

The config file contains meta information around all of the data columns that will be loaded into autoMl and custom ML

_data column name_ dictates the column name

_foriegn table config_ can be
- 0: column not relating to foreign table - nulls handled as subsequently configured
- 1: a key to foregin table - nulls are not accepted, offending rows dropped
- 2: value from foreign table - no need to check nulls as the key is checked

_is feature_ can be
- 0: this column should not be used as a feature
- 1: use this column as a feature

The _nullable_ dictate whether a feature can be null for a specific record. For example any reference to active joining date will be accepted as null for a dependant record.
- 0: this column can be null for the specified record type
- 1: this column cannot be null for the specififed record type

# Columns to remove
This file is a list of all columns which shouldn't be used as a feature specifically for a single ML problem

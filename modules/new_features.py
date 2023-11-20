import gc
import pandas as pd
import pyarrow

from modules.reduce_mem_usage import reduce_mem_usage


# Path to the folder with raw data.
path = 'raw_data/'

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """new feature creation function"""

    targets = pd.read_csv(f'{path}targets.csv')

    # Set lists for all flag-features and for separate groups of them.
    # List of all flags.
    flag_cols = [
        'is_zero_loans5',
        'is_zero_loans530',
        'is_zero_loans3060',
        'is_zero_loans6090',
        'is_zero_loans90',
        'is_zero_util',
        'is_zero_over2limit',
        'is_zero_maxover2limit',
        'pclose_flag',
        'fclose_flag'
    ]

    # List of flags with data on the known date of loan closing.
    unlimited_loans = [
                'pclose_flag',
                'fclose_flag'
    ]

    # List of flags with data on overdue payments.
    is_zero_loans = [
                "is_zero_loans5",
                "is_zero_loans530",
                "is_zero_loans3060",
                "is_zero_loans6090",
                "is_zero_loans90"
    ]

    # A slice
    loans_df = df[['id', 'rn']+flag_cols].copy()

    # A new feature indicating that the closing date is not defined.
    loans_df["unlimited_loan"] = loans_df[unlimited_loans].any(axis=1)

    # Change the markings to see the fact of presence, not absence of the event.
    loans_df[flag_cols] = loans_df[flag_cols].replace([0, 1], [1, 0])

    # Create a new feature 'overdue' if there was any delay among payments.
    loans_df["overdue"] = loans_df[is_zero_loans].any(axis=1)

    # Aggregate all the original features with flags by id and aggregate them by mean value.
    # So we get how many facts of presence of this or that action are in average per one client.
    mn = loans_df.groupby('id')[flag_cols].mean()*100
    mn.columns = [x+'mean' for x in mn.columns]

    # Aggregate all initial features with flags by id and summarize them.
    # So we get the total number of facts of presence of this or that action for one client.
    sm = loans_df.groupby('id')[flag_cols].sum()
    sm.columns = [x+'total' for x in sm.columns]

    # Create a new feature containing the total number of overdues for one client
    ov = loans_df.groupby('id')['overdue'].sum()

    # Create a new feature containing the total number of unlimited credits for one client
    ul = loans_df.groupby('id')['unlimited_loan'].sum()

    # Create a new feature containing the total number of loans for one client
    total_loans = loans_df.groupby('id')['rn'].max()

    # Put it all back together.
    new_features = pd.DataFrame(data=ov).reset_index()
    new_features = pd.concat([new_features, total_loans, ul, mn, sm], axis=1)

    # Rename the new features
    new_features.rename(columns={'rn': 'total_loans', 'overdue': 'overdue_total_num'}, inplace=True)

    # Create a new feature 'frequent_client':
    # If the client has less than 4 credits (25quantile) he is assigned 0 if up to 13 (75quantile) - 1 and 0.5 if more.
    new_features['frequent_client'] = new_features['total_loans'].apply(lambda x: 0.0 if x <= 4 else (1.0 if x <= 12 else 0.5))

    # Create a new feature clint overdue share as the number of overdues divided by the total number of loans.
    new_features["overdue_share"] = (new_features['overdue_total_num']*100/new_features['total_loans']).astype('float32')

    # Add a target and reduce its type to int8.
    new_features = new_features.merge(targets, on=['id'])

    del loans_df, total_loans, ul, mn, sm, ov, targets
    gc.collect()

    # Apply the optimization function to save memory
    reduce_mem_usage(new_features)
    gc.collect()

    return new_features

import gc
import pandas as pd
import pyarrow

from sklearn.model_selection import train_test_split


# A list of features to be removed.
cols_to_drop = [
    'pre_loans_total_overdue',
    'pre_loans3060',
    'pre_loans6090',
    'pre_loans90',
    'is_zero_loans5',
    'is_zero_loans530',
    'is_zero_loans3060',
    'is_zero_loans6090',
    'is_zero_loans90',
    'is_zero_util',
    'is_zero_over2limit',
    'is_zero_maxover2limit',
    'pclose_flag',
    'fclose_flag',
    'rn',
]


def parts_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """OHE encoding of features via Pandas get_dummies"""

    df = df.copy()
    encoded_df = pd.DataFrame()
    for i in range(1, len(df.columns), 1):
        block = df.iloc[:, i:i+1].copy()
        cols = list(block.columns.values)

        encoded = pd.get_dummies(block[cols], columns=cols, dtype='int8')
        encoded_data = pd.concat([df['id'], encoded], axis=1)

        part = encoded_data.groupby('id').agg('sum')
        encoded_df = pd.concat([encoded_df, part], axis=1)

    encoded_df.reset_index(inplace=True)

    print('Processing completed.')

    return encoded_df


def final_df(df : pd.DataFrame, new_features: pd.DataFrame):
    """
    a function that merges new and coded features,
    removes unnecessary features
    splitting into Train and Test in 80/20 ratio
    prepares a new dataset with test ID for further filling of predicted values
    """

    df = df.drop(cols_to_drop, axis=1)
    encoded_df = parts_encoder(df)

    data = encoded_df.merge(new_features, on=['id'])

    train, test = train_test_split(data, test_size=0.2, stratify=data['flag'], random_state=73)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    pred_df = pd.DataFrame({'id': test['id']})

    del data, df, encoded_df

    X = train.drop(['id', 'flag'], axis=1)
    y = train['flag']
    X_test = test.drop(['id', 'flag'], axis=1)
    y_test = test['flag']

    gc.collect()

    return X, y, X_test, y_test, pred_df

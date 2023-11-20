import datetime
import dill
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from catboost import CatBoostClassifier

from modules.merging_data import prepare_transactions_dataset
from modules.new_features import create_features
from modules.final_df import final_df


# Path to the folder with raw data.
path = 'raw_data/'


class CustomPipeline(BaseEstimator, TransformerMixin):
    """Pipeline own class with fit, predict, predict_proba methods"""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def pipeline() -> None:
    """the pipeline that pulls everything together"""

    df = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=12, num_parts_total=2)
    new_features = create_features(df)
    X, y, X_test, y_test, pred_df = final_df(df, new_features)

    model = CatBoostClassifier(
        depth=5,
        l2_leaf_reg=1,
        learning_rate=0.05,
        random_strength=0.5,
        eval_metric='AUC',
        auto_class_weights='Balanced',
        task_type='GPU',
        devices='0:1',
        iterations=3000,
        early_stopping_rounds=500,
        bootstrap_type='Bayesian',
        random_seed=73
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=73)
    score = cross_val_score(model, X, y, scoring='roc_auc', cv=skf, n_jobs=None, verbose=20)
    mean_score = score.mean()
    print(f'\nMean ROC-AUC score on train: {mean_score:.5f}')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', CustomPipeline(model))
    ])

    # Fit the pipeline on all training data
    pipe.fit(X, y)

    # Make predictions on the test data
    predictions = pipe.predict(X_test)
    pred_df['prediction'] = predictions.tolist()
    pred_df.to_csv('Data/predictions.csv', index=False)
    print(f'\nPredictions are saved in csv')

    # Get ROC-AUC score
    y_pred = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'\nROC-AUC score on train: {roc_auc:.5f}')

    # Evaluate the performance, print accuracy for example
    model_filename = f'Data/credit_defolt_prediction_model.pkl'
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'Name': 'Credit Defolt Prediction Model',
                'Author': 'Sergey Jangozyan',
                'Version': 1.0,
                'Date': datetime.datetime.now(),
                'Type': 'CatboostClassifier',
                'ROC-AUC on Test': roc_auc,
                'Mean ROC-AUC on CV': mean_score
            }
        }, file, recurse=True)

    print(f'\nModel is saved in pickle')


if __name__ == '__main__':
    pipeline()

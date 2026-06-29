import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'store', 'item',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_365',
    'rolling_mean_7', 'rolling_mean_28', 'rolling_mean_84',
    'rolling_std_7', 'rolling_std_28',
    'day_of_week', 'week_of_year', 'month',
    'year', 'quarter', 'is_weekend'
]

def engineer_features(df):
    df = df.sort_values(['store','item','date']).reset_index(drop=True)

    for lag in [1, 7, 14, 28, 365]:
        df[f'lag_{lag}'] = df.groupby(['store','item'])['sales'].shift(lag)

    for w in [7, 28, 84]:
        df[f'rolling_mean_{w}'] = df.groupby(['store','item'])['sales'].transform(
            lambda x: x.shift(1).rolling(w).mean())

    for w in [7, 28]:
        df[f'rolling_std_{w}'] = df.groupby(['store','item'])['sales'].transform(
            lambda x: x.shift(1).rolling(w).std())

    df['day_of_week']  = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month']        = df['date'].dt.month
    df['year']         = df['date'].dt.year
    df['quarter']      = df['date'].dt.quarter
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)

    df = df.dropna()
    return df

def run_models(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = engineer_features(df)

    train = df[df['date'] < '2017-01-01']
    test  = df[df['date'] >= '2017-01-01']

    X_train = train[FEATURE_COLS]
    X_test  = test[FEATURE_COLS]
    y_train = train['sales']
    y_test  = test['sales']

    def evaluate(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return {'Model': name,
                'MAE':  round(mae, 4),
                'RMSE': round(rmse, 4),
                'R2':   round(r2, 4)}

    # Moving Average
    ma_pred    = X_test['rolling_mean_28'].fillna(y_train.mean())
    results_ma = evaluate('Moving Average (Baseline)', y_test, ma_pred)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred    = rf.predict(X_test)
    results_rf = evaluate('Random Forest', y_test, rf_pred)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred    = xgb.predict(X_test)
    results_xgb = evaluate('XGBoost', y_test, xgb_pred)

    # Comparison table
    results_df = pd.DataFrame([results_ma, results_rf, results_xgb])
    baseline   = results_ma['MAE']
    results_df['Improvement_%'] = (
        (baseline - results_df['MAE']) / baseline * 100
    ).round(1)

    # Sample predictions for chart — Store 1 Item 1
    sample      = test[(test['store']==1) & (test['item']==1)].copy()
    sample_feat = sample[FEATURE_COLS]

    chart_data = pd.DataFrame({
        'date':     sample['date'].values,
        'actual':   sample['sales'].values,
        'xgb_pred': xgb.predict(sample_feat),
        'ma_pred':  sample['rolling_mean_28'].fillna(y_train.mean()).values,
    })

    # Inventory simulation
    sim             = test.copy()
    sim['pred_ma']  = ma_pred.values
    sim['pred_xgb'] = xgb_pred

    sim['waste_ma']     = (sim['pred_ma']  - sim['sales']).clip(lower=0)
    sim['waste_xgb']    = (sim['pred_xgb'] - sim['sales']).clip(lower=0)
    sim['stockout_ma']  = (sim['sales'] - sim['pred_ma'] ).clip(lower=0)
    sim['stockout_xgb'] = (sim['sales'] - sim['pred_xgb']).clip(lower=0)

    tw_ma  = sim['waste_ma'].sum()
    tw_xgb = sim['waste_xgb'].sum()
    ts_ma  = sim['stockout_ma'].sum()
    ts_xgb = sim['stockout_xgb'].sum()

    simulation = {
        'waste_ma':           int(tw_ma),
        'waste_xgb':          int(tw_xgb),
        'waste_reduction':    round((tw_ma - tw_xgb) / tw_ma * 100, 1),
        'stockout_ma':        int(ts_ma),
        'stockout_xgb':       int(ts_xgb),
        'stockout_reduction': round((ts_ma - ts_xgb) / ts_ma * 100, 1),
    }

    return results_df, chart_data, simulation
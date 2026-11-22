import pandas as pd
import numpy as np
import datetime as dt
import os
import warnings
from tqdm import tqdm

from models import (HeterocedasticQuantileRegression, 
                    QuantileRegressionAveraging,
                    AdaptiveConformalInference,
                    WidthAdaptiveConformalInference,
                    # WidthAdaptiveConformalInference_2,
                    dcp_opt)

warnings.filterwarnings(action='ignore')
np.random.seed(123)

df = pd.read_csv("Data//df_2024.csv")
df['full_date'] = pd.to_datetime(df.full_date)

df['mean_pred'] = df[['pred1', 'pred2', 'pred3']].mean(axis=1)
df['std_pred'] = df[['pred1', 'pred2', 'pred3']].std(axis=1, ddof=0)

date_test = dt.datetime(2024, 1, 1)
idx_date_test = df[pd.to_datetime(df.date) == date_test].head(1).index.values[0]
idx_first_date_qr = idx_date_test - 6*30*24
m = 3
alphas = [0.1, 0.2]

for alpha in alphas:
    if not os.path.isfile(f"Data//df_quantile_regression_alpha_{alpha}.csv"):
        # Heterocedastic Quantile Regression
        preds_inf_hqr, preds_sup_hqr = HeterocedasticQuantileRegression(idx_first_date_qr, df, alpha, by_hour=False, save_coefs=True)
        df.loc[df.tail(len(preds_inf_hqr)).index, 'preds_inf_hqr'] = preds_inf_hqr
        df.loc[df.tail(len(preds_sup_hqr)).index, 'preds_sup_hqr'] = preds_sup_hqr

        # # Heterocedastic Quantile Regression by hour
        # preds_inf_hqr_by_hour, preds_sup_hqr_by_hour = HeterocedasticQuantileRegression(idx_first_date_qr, df, alpha, by_hour=True)
        # df.loc[df.tail(len(preds_inf_hqr_by_hour)).index, 'preds_inf_hqr_by_hour'] = preds_inf_hqr_by_hour
        # df.loc[df.tail(len(preds_sup_hqr_by_hour)).index, 'preds_sup_hqr_by_hour'] = preds_sup_hqr_by_hour

        # Heterocedastic Quantile Regression with varying weights
        preds_inf_hqr_w, preds_sup_hqr_w = HeterocedasticQuantileRegression(idx_first_date_qr, df, alpha, by_hour=False, varying_weights=True, num_preds=m)
        df.loc[df.tail(len(preds_inf_hqr_w)).index, 'preds_inf_hqr_w'] = preds_inf_hqr_w
        df.loc[df.tail(len(preds_sup_hqr_w)).index, 'preds_sup_hqr_w'] = preds_sup_hqr_w

        # # Heterocedastic Quantile Regression with varying weights by hour
        # preds_inf_hqr_w_by_hour, preds_sup_hqr_w_by_hour = HeterocedasticQuantileRegression(idx_first_date_qr, df, alpha, by_hour=True, varying_weights=True, num_preds=m)
        # df.loc[df.tail(len(preds_inf_hqr_w_by_hour)).index, 'preds_inf_hqr_w_by_hour'] = preds_inf_hqr_w_by_hour
        # df.loc[df.tail(len(preds_sup_hqr_w_by_hour)).index, 'preds_sup_hqr_w_by_hour'] = preds_sup_hqr_w_by_hour

        # Quantile Regression Averaging
        preds_inf_qra, preds_sup_qra = QuantileRegressionAveraging(idx_first_date_qr, df, alpha, m=m, by_hour=False)
        df.loc[df.tail(len(preds_inf_qra)).index, 'preds_inf_qra'] = preds_inf_qra
        df.loc[df.tail(len(preds_sup_qra)).index, 'preds_sup_qra'] = preds_sup_qra

        # # Quantile Regression Averaging by hour
        # preds_inf_qra_by_hour, preds_sup_qra_by_hour = QuantileRegressionAveraging(idx_first_date_qr, df, alpha, m=m, by_hour=True)
        # df.loc[df.tail(len(preds_inf_qra_by_hour)).index, 'preds_inf_qra_by_hour'] = preds_inf_qra_by_hour
        # df.loc[df.tail(len(preds_sup_qra_by_hour)).index, 'preds_sup_qra_by_hour'] = preds_sup_qra_by_hour

        # Save the dataframe (just in case)
        df.to_csv(f"Data//df_quantile_regression_alpha_{alpha}.csv", index=False)
    else:
        df = pd.read_csv(f"Data//df_quantile_regression_alpha_{alpha}.csv")
        # df = pd.read_csv(f"Data//df_final_alpha_{alpha}.csv")
        df['full_date'] = pd.to_datetime(df.full_date)
       
    
    # Distributional Conformal Prediction with QRA
    preds_inf_dcp_qra, preds_sup_dcp_qra = np.array([]), np.array([])
    df_test = df[df.index >= idx_date_test][['date', 'hour', 'real', 'pred1', 'pred2', 'pred3']].copy()
    taus = np.linspace(0.001, 0.999, 200)
    days_val = 6*30
    for date_aux in tqdm(df_test.date.unique(), desc="Distributional Conformal Prediction with QRA"):
        df_date = df_test[df_test.date == date_aux].copy()
        for hour_aux in tqdm(df_test.hour.unique(), desc="Hours", leave=False):
            df_test_aux = df_date[df_date.hour == hour_aux].copy()
            df_cal_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (df.hour == hour_aux)][['date', 'real', 'hour', 'pred1', 'pred2', 'pred3']].copy()
            df_train_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val*2))][['date', 'hour', 'real', 'pred1', 'pred2', 'pred3']].copy()

            Y0 = df_train_aux.real.values.reshape(-1)
            X0 = df_train_aux[['pred1', 'pred2', 'pred3']].values.reshape(-1, 3)
            Y1 = df_cal_aux.real.values.reshape(-1)
            X1 = df_cal_aux[['pred1', 'pred2', 'pred3']].values.reshape(-1, 3)
            Y_test = df_test_aux.real.values.reshape(-1)
            X_test = df_test_aux[['pred1', 'pred2', 'pred3']].values.reshape(-1, 3)

            ys = np.quantile(np.unique(np.concatenate((Y0, Y1))), np.linspace(0.001, 0.999, 200))

            preds_sup_dcp_qra_obs, preds_inf_dcp_qra_obs, _, _ = dcp_opt(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha)
            preds_inf_dcp_qra = np.concatenate((preds_inf_dcp_qra, preds_inf_dcp_qra_obs))
            preds_sup_dcp_qra = np.concatenate((preds_sup_dcp_qra, preds_sup_dcp_qra_obs))
    
    df.loc[df_test.index, 'preds_inf_dcp_qra'] = preds_inf_dcp_qra
    df.loc[df_test.index, 'preds_sup_dcp_qra'] = preds_sup_dcp_qra
    
    # Distributional Conformal Prediction with HQR with varying weights
    preds_inf_dcp_hqr_w, preds_sup_dcp_hqr_w = np.array([]), np.array([])
    df_test = df[df.index >= idx_date_test][['date', 'hour', 'real', 'pred1', 'pred2', 'pred3', 'std_pred']].copy()
    taus = np.linspace(0.001, 0.999, 200)
    days_val = 6*30
    for date_aux in tqdm(df_test.date.unique(), desc="Distributional Conformal Prediction with HQR with varying weights"):
        df_date = df_test[df_test.date == date_aux].copy()
        for hour_aux in tqdm(df_test.hour.unique(), desc="Hours", leave=False):
            df_test_aux = df_date[df_date.hour == hour_aux].copy()
            df_cal_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (df.hour == hour_aux)][['date', 'real', 'hour', 'pred1', 'pred2', 'pred3', 'std_pred']].copy()
            df_train_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val*2))][['date', 'hour', 'real', 'pred1', 'pred2', 'pred3', 'std_pred']].copy()

            Y0 = df_train_aux.real.values.reshape(-1)
            X0 = df_train_aux[['pred1', 'pred2', 'pred3', 'std_pred']].values.reshape(-1, 4)
            Y1 = df_cal_aux.real.values.reshape(-1)
            X1 = df_cal_aux[['pred1', 'pred2', 'pred3', 'std_pred']].values.reshape(-1, 4)
            Y_test = df_test_aux.real.values.reshape(-1)
            X_test = df_test_aux[['pred1', 'pred2', 'pred3', 'std_pred']].values.reshape(-1, 4)

            ys = np.quantile(np.unique(np.concatenate((Y0, Y1))), np.linspace(0.001, 0.999, 200))

            preds_sup_dcp_hqr_w_obs, preds_inf_dcp_hqr_w_obs, _, _ = dcp_opt(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha)
            preds_inf_dcp_hqr_w = np.concatenate((preds_inf_dcp_hqr_w, preds_inf_dcp_hqr_w_obs))
            preds_sup_dcp_hqr_w = np.concatenate((preds_sup_dcp_hqr_w, preds_sup_dcp_hqr_w_obs))
    
    df.loc[df_test.index, 'preds_inf_dcp_hqr_w'] = preds_inf_dcp_hqr_w
    df.loc[df_test.index, 'preds_sup_dcp_hqr_w'] = preds_sup_dcp_hqr_w

    # Distributional Conformal Prediction with HQR
    preds_inf_dcp_hqr, preds_sup_dcp_hqr = np.array([]), np.array([])
    df_test = df[df.index >= idx_date_test][['date', 'hour', 'real', 'mean_pred', 'std_pred']].copy()
    taus = np.linspace(0.001, 0.999, 200)
    days_val = 6*30
    for date_aux in tqdm(df_test.date.unique(), desc="Distributional Conformal Prediction with HQR"):
        df_date = df_test[df_test.date == date_aux].copy()
        for hour_aux in tqdm(df_test.hour.unique(), desc="Hours", leave=False):
            df_test_aux = df_date[df_date.hour == hour_aux].copy()
            df_cal_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (df.hour == hour_aux)][['date', 'real', 'hour', 'mean_pred', 'std_pred']].copy()
            df_train_aux = df[(pd.to_datetime(df.date) < pd.to_datetime(date_aux) - dt.timedelta(days=days_val)) & (pd.to_datetime(df.date) >= pd.to_datetime(date_aux) - dt.timedelta(days=days_val*2))][['date', 'hour', 'real', 'mean_pred', 'std_pred']].copy()

            Y0 = df_train_aux.real.values.reshape(-1)
            X0 = df_train_aux[['mean_pred', 'std_pred']].values.reshape(-1, 2)
            Y1 = df_cal_aux.real.values.reshape(-1)
            X1 = df_cal_aux[['mean_pred', 'std_pred']].values.reshape(-1, 2)
            Y_test = df_test_aux.real.values.reshape(-1)
            X_test = df_test_aux[['mean_pred', 'std_pred']].values.reshape(-1, 2)

            ys = np.quantile(np.unique(np.concatenate((Y0, Y1))), np.linspace(0.001, 0.999, 200))

            preds_sup_dcp_hqr_obs, preds_inf_dcp_hqr_obs, _, _ = dcp_opt(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha)
            preds_inf_dcp_hqr = np.concatenate((preds_inf_dcp_hqr, preds_inf_dcp_hqr_obs))
            preds_sup_dcp_hqr = np.concatenate((preds_sup_dcp_hqr, preds_sup_dcp_hqr_obs))
    
    df.loc[df_test.index, 'preds_inf_dcp_hqr'] = preds_inf_dcp_hqr
    df.loc[df_test.index, 'preds_sup_dcp_hqr'] = preds_sup_dcp_hqr

    df['length_hqr'] = df['preds_sup_hqr'] - df['preds_inf_hqr']
    # df['length_hqr_by_hour'] = df['preds_sup_hqr_by_hour'] - df['preds_inf_hqr_by_hour']
    df['length_hqr_w'] = df['preds_sup_hqr_w'] - df['preds_inf_hqr_w']
    # df['length_hqr_w_by_hour'] = df['preds_sup_hqr_w_by_hour'] - df['preds_inf_hqr_w_by_hour']
    df['length_qra'] = df['preds_sup_qra'] - df['preds_inf_qra']
    # df['length_qra_by_hour'] = df['preds_sup_qra_by_hour'] - df['preds_inf_qra_by_hour']

    # Adaptive Conformal Inference over HQR
    preds_inf_aci_hqr, preds_sup_aci_hqr = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='hqr', by_hour=False, save_alphas=True)
    df.loc[df.tail(len(preds_inf_aci_hqr)).index, 'preds_inf_aci_hqr'] = preds_inf_aci_hqr
    df.loc[df.tail(len(preds_sup_aci_hqr)).index, 'preds_sup_aci_hqr'] = preds_sup_aci_hqr
    df.loc[df.preds_inf_aci_hqr == -np.inf, 'preds_inf_aci_hqr'] = df[df.index < idx_date_test].preds_inf_hqr.min()
    df.loc[df.preds_sup_aci_hqr == np.inf, 'preds_sup_aci_hqr'] = df[df.index < idx_date_test].preds_sup_hqr.max()

    # # Adaptive Conformal Inference over HQR by hour
    # preds_inf_aci_hqr_by_hour, preds_sup_aci_hqr_by_hour = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='hqr', by_hour=True)
    # df.loc[df.tail(len(preds_inf_aci_hqr_by_hour)).index, 'preds_inf_aci_hqr_by_hour'] = preds_inf_aci_hqr_by_hour
    # df.loc[df.tail(len(preds_sup_aci_hqr_by_hour)).index, 'preds_sup_aci_hqr_by_hour'] = preds_sup_aci_hqr_by_hour
    # df.loc[df.preds_inf_aci_hqr_by_hour == -np.inf, 'preds_inf_aci_hqr_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_by_hour.min()
    # df.loc[df.preds_sup_aci_hqr_by_hour == np.inf, 'preds_sup_aci_hqr_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_by_hour.max()

    # Adaptive Conformal Inference over HQR with varying weights
    preds_inf_aci_hqr_w, preds_sup_aci_hqr_w = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='hqr_w', by_hour=False)
    df.loc[df.tail(len(preds_inf_aci_hqr_w)).index, 'preds_inf_aci_hqr_w'] = preds_inf_aci_hqr_w
    df.loc[df.tail(len(preds_sup_aci_hqr_w)).index, 'preds_sup_aci_hqr_w'] = preds_sup_aci_hqr_w
    df.loc[df.preds_inf_aci_hqr_w == -np.inf, 'preds_inf_aci_hqr_w'] = df[df.index < idx_date_test].preds_inf_hqr_w.min()
    df.loc[df.preds_sup_aci_hqr_w == np.inf, 'preds_sup_aci_hqr_w'] = df[df.index < idx_date_test].preds_sup_hqr_w.max()

    # # Adaptive Conformal Inference over HQR with varying weights by hour
    # preds_inf_aci_hqr_w_by_hour, preds_sup_aci_hqr_w_by_hour = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='hqr_w', by_hour=True)
    # df.loc[df.tail(len(preds_inf_aci_hqr_w_by_hour)).index, 'preds_inf_aci_hqr_w_by_hour'] = preds_inf_aci_hqr_w_by_hour
    # df.loc[df.tail(len(preds_sup_aci_hqr_w_by_hour)).index, 'preds_sup_aci_hqr_w_by_hour'] = preds_sup_aci_hqr_w_by_hour
    # df.loc[df.preds_inf_aci_hqr_w_by_hour == -np.inf, 'preds_inf_aci_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_w_by_hour.min()
    # df.loc[df.preds_sup_aci_hqr_w_by_hour == np.inf, 'preds_sup_aci_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_w_by_hour.max()
    
    # Adaptive Conformal Inference over QRA
    preds_inf_aci_qra, preds_sup_aci_qra = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_aci_qra)).index, 'preds_inf_aci_qra'] = preds_inf_aci_qra
    df.loc[df.tail(len(preds_sup_aci_qra)).index, 'preds_sup_aci_qra'] = preds_sup_aci_qra
    df.loc[df.preds_inf_aci_qra == -np.inf, 'preds_inf_aci_qra'] = df[df.index < idx_date_test].preds_inf_qra.min()
    df.loc[df.preds_sup_aci_qra == np.inf, 'preds_sup_aci_qra'] = df[df.index < idx_date_test].preds_sup_qra.max()

    # # Adaptive Conformal Inference over QRA by hour
    # preds_inf_aci_qra_by_hour, preds_sup_aci_qra_by_hour = AdaptiveConformalInference(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=True)
    # df.loc[df.tail(len(preds_inf_aci_qra_by_hour)).index, 'preds_inf_aci_qra_by_hour'] = preds_inf_aci_qra_by_hour
    # df.loc[df.tail(len(preds_sup_aci_qra_by_hour)).index, 'preds_sup_aci_qra_by_hour'] = preds_sup_aci_qra_by_hour
    # df.loc[df.preds_inf_aci_qra_by_hour == -np.inf, 'preds_inf_aci_qra_by_hour'] = df[df.index < idx_date_test].preds_inf_qra_by_hour.min()
    # df.loc[df.preds_sup_aci_qra_by_hour == np.inf, 'preds_sup_aci_qra_by_hour'] = df[df.index < idx_date_test].preds_sup_qra_by_hour.max()
        
    # Width Adaptive Conformal Inference over HQR
    preds_inf_waci_hqr, preds_sup_waci_hqr = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='hqr', by_hour=False, save_alphas=True)
    df.loc[df.tail(len(preds_inf_waci_hqr)).index, 'preds_inf_waci_hqr'] = preds_inf_waci_hqr
    df.loc[df.tail(len(preds_sup_waci_hqr)).index, 'preds_sup_waci_hqr'] = preds_sup_waci_hqr
    df.loc[df.preds_inf_waci_hqr == -np.inf, 'preds_inf_waci_hqr'] = df[df.index < idx_date_test].preds_inf_hqr.min()
    df.loc[df.preds_sup_waci_hqr == np.inf, 'preds_sup_waci_hqr'] = df[df.index < idx_date_test].preds_sup_hqr.max()

    # # Width Adaptive Conformal Inference over HQR by hour
    # preds_inf_waci_hqr_by_hour, preds_sup_waci_hqr_by_hour = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='hqr', by_hour=True)
    # df.loc[df.tail(len(preds_inf_waci_hqr_by_hour)).index, 'preds_inf_waci_hqr_by_hour'] = preds_inf_waci_hqr_by_hour
    # df.loc[df.tail(len(preds_sup_waci_hqr_by_hour)).index, 'preds_sup_waci_hqr_by_hour'] = preds_sup_waci_hqr_by_hour
    # df.loc[df.preds_inf_waci_hqr_by_hour == -np.inf, 'preds_inf_waci_hqr_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_by_hour.min()
    # df.loc[df.preds_sup_waci_hqr_by_hour == np.inf, 'preds_sup_waci_hqr_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_by_hour.max()

    # Width Adaptive Conformal Inference over HQR with varying weights
    preds_inf_waci_hqr_w, preds_sup_waci_hqr_w = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='hqr_w', by_hour=False)
    df.loc[df.tail(len(preds_inf_waci_hqr_w)).index, 'preds_inf_waci_hqr_w'] = preds_inf_waci_hqr_w
    df.loc[df.tail(len(preds_sup_waci_hqr_w)).index, 'preds_sup_waci_hqr_w'] = preds_sup_waci_hqr_w
    df.loc[df.preds_inf_waci_hqr_w == -np.inf, 'preds_inf_waci_hqr_w'] = df[df.index < idx_date_test].preds_inf_hqr_w.min()
    df.loc[df.preds_sup_waci_hqr_w == np.inf, 'preds_sup_waci_hqr_w'] = df[df.index < idx_date_test].preds_sup_hqr_w.max()

    # # Width Adaptive Conformal Inference over HQR with varying weights by hour
    # preds_inf_waci_hqr_w_by_hour, preds_sup_waci_hqr_w_by_hour = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='hqr_w', by_hour=True)
    # df.loc[df.tail(len(preds_inf_waci_hqr_w_by_hour)).index, 'preds_inf_waci_hqr_w_by_hour'] = preds_inf_waci_hqr_w_by_hour
    # df.loc[df.tail(len(preds_sup_waci_hqr_w_by_hour)).index, 'preds_sup_waci_hqr_w_by_hour'] = preds_sup_waci_hqr_w_by_hour
    # df.loc[df.preds_inf_waci_hqr_w_by_hour == -np.inf, 'preds_inf_waci_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_w_by_hour.min()
    # df.loc[df.preds_sup_waci_hqr_w_by_hour == np.inf, 'preds_sup_waci_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_w_by_hour.max()

    # Width Adaptive Conformal Inference over QRA
    preds_inf_waci_qra, preds_sup_waci_qra = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='qra', by_hour=False)
    df.loc[df.tail(len(preds_inf_waci_qra)).index, 'preds_inf_waci_qra'] = preds_inf_waci_qra
    df.loc[df.tail(len(preds_sup_waci_qra)).index, 'preds_sup_waci_qra'] = preds_sup_waci_qra
    df.loc[df.preds_inf_waci_qra == -np.inf, 'preds_inf_waci_qra'] = df[df.index < idx_date_test].preds_inf_qra.min()
    df.loc[df.preds_sup_waci_qra == np.inf, 'preds_sup_waci_qra'] = df[df.index < idx_date_test].preds_sup_qra.max()

    # # Width Adaptive Conformal Inference over QRA by hour
    # preds_inf_waci_qra_by_hour, preds_sup_waci_qra_by_hour = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=3, method='qra', by_hour=True)
    # df.loc[df.tail(len(preds_inf_waci_qra_by_hour)).index, 'preds_inf_waci_qra_by_hour'] = preds_inf_waci_qra_by_hour
    # df.loc[df.tail(len(preds_sup_waci_qra_by_hour)).index, 'preds_sup_waci_qra_by_hour'] = preds_sup_waci_qra_by_hour
    # df.loc[df.preds_inf_waci_qra_by_hour == -np.inf, 'preds_inf_waci_qra_by_hour'] = df[df.index < idx_date_test].preds_inf_qra_by_hour.min()
    # df.loc[df.preds_sup_waci_qra_by_hour == np.inf, 'preds_sup_waci_qra_by_hour'] = df[df.index < idx_date_test].preds_sup_qra_by_hour.max()

    # # Width Adaptive Conformal Inference 2 over HQR
    # preds_inf_waci_2_hqr, preds_sup_waci_2_hqr = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='hqr', by_hour=False, save_alphas=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_hqr)).index, 'preds_inf_waci_2_hqr'] = preds_inf_waci_2_hqr
    # df.loc[df.tail(len(preds_sup_waci_2_hqr)).index, 'preds_sup_waci_2_hqr'] = preds_sup_waci_2_hqr
    # df.loc[df.preds_inf_waci_2_hqr == -np.inf, 'preds_inf_waci_2_hqr'] = df[df.index < idx_date_test].preds_inf_hqr.min()
    # df.loc[df.preds_sup_waci_2_hqr == np.inf, 'preds_sup_waci_2_hqr'] = df[df.index < idx_date_test].preds_sup_hqr.max()

    # # Width Adaptive Conformal Inference 2 over HQR by hour
    # preds_inf_waci_2_hqr_by_hour, preds_sup_waci_2_hqr_by_hour = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='hqr', by_hour=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_hqr_by_hour)).index, 'preds_inf_waci_2_hqr_by_hour'] = preds_inf_waci_2_hqr_by_hour
    # df.loc[df.tail(len(preds_sup_waci_2_hqr_by_hour)).index, 'preds_sup_waci_2_hqr_by_hour'] = preds_sup_waci_2_hqr_by_hour
    # df.loc[df.preds_inf_waci_2_hqr_by_hour == -np.inf, 'preds_inf_waci_2_hqr_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_by_hour.min()
    # df.loc[df.preds_sup_waci_2_hqr_by_hour == np.inf, 'preds_sup_waci_2_hqr_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_by_hour.max()

    # # Width Adaptive Conformal Inference 2 over HQR with varying weights
    # preds_inf_waci_2_hqr_w, preds_sup_waci_2_hqr_w = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='hqr_w', by_hour=False, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_hqr_w)).index, 'preds_inf_waci_2_hqr_w'] = preds_inf_waci_2_hqr_w
    # df.loc[df.tail(len(preds_sup_waci_2_hqr_w)).index, 'preds_sup_waci_2_hqr_w'] = preds_sup_waci_2_hqr_w
    # df.loc[df.preds_inf_waci_2_hqr_w == -np.inf, 'preds_inf_waci_2_hqr_w'] = df[df.index < idx_date_test].preds_inf_hqr_w.min()
    # df.loc[df.preds_sup_waci_2_hqr_w == np.inf, 'preds_sup_waci_2_hqr_w'] = df[df.index < idx_date_test].preds_sup_hqr_w.max()

    # # Width Adaptive Conformal Inference 2 over HQR with varying weights by hour
    # preds_inf_waci_2_hqr_w_by_hour, preds_sup_waci_2_hqr_w_by_hour = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='hqr_w', by_hour=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_hqr_w_by_hour)).index, 'preds_inf_waci_2_hqr_w_by_hour'] = preds_inf_waci_2_hqr_w_by_hour
    # df.loc[df.tail(len(preds_sup_waci_2_hqr_w_by_hour)).index, 'preds_sup_waci_2_hqr_w_by_hour'] = preds_sup_waci_2_hqr_w_by_hour
    # df.loc[df.preds_inf_waci_2_hqr_w_by_hour == -np.inf, 'preds_inf_waci_2_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_inf_hqr_w_by_hour.min()
    # df.loc[df.preds_sup_waci_2_hqr_w_by_hour == np.inf, 'preds_sup_waci_2_hqr_w_by_hour'] = df[df.index < idx_date_test].preds_sup_hqr_w_by_hour.max()

    # # Width Adaptive Conformal Inference 2 over QRA
    # preds_inf_waci_2_qra, preds_sup_waci_2_qra = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=False, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_qra)).index, 'preds_inf_waci_2_qra'] = preds_inf_waci_2_qra
    # df.loc[df.tail(len(preds_sup_waci_2_qra)).index, 'preds_sup_waci_2_qra'] = preds_sup_waci_2_qra
    # df.loc[df.preds_inf_waci_2_qra == -np.inf, 'preds_inf_waci_2_qra'] = df[df.index < idx_date_test].preds_inf_qra.min()
    # df.loc[df.preds_sup_waci_2_qra == np.inf, 'preds_sup_waci_2_qra'] = df[df.index < idx_date_test].preds_sup_qra.max()

    # # Width Adaptive Conformal Inference 2 over QRA by hour
    # preds_inf_waci_2_qra_by_hour, preds_sup_waci_2_qra_by_hour = WidthAdaptiveConformalInference_2(idx_date_test, df, alpha, gamma=0.02, method='qra', by_hour=True, rate=1.02)
    # df.loc[df.tail(len(preds_inf_waci_2_qra_by_hour)).index, 'preds_inf_waci_2_qra_by_hour'] = preds_inf_waci_2_qra_by_hour
    # df.loc[df.tail(len(preds_sup_waci_2_qra_by_hour)).index, 'preds_sup_waci_2_qra_by_hour'] = preds_sup_waci_2_qra_by_hour
    # df.loc[df.preds_inf_waci_2_qra_by_hour == -np.inf, 'preds_inf_waci_2_qra_by_hour'] = df[df.index < idx_date_test].preds_inf_qra_by_hour.min()
    # df.loc[df.preds_sup_waci_2_qra_by_hour == np.inf, 'preds_sup_waci_2_qra_by_hour'] = df[df.index < idx_date_test].preds_sup_qra_by_hour.max()

    # Save dataframe
    df.to_csv(f"Data//df_final_alpha_{alpha}.csv", index=False)












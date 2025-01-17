import pandas as pd
import numpy as np
import datetime as dt
import os
import warnings
from tqdm import tqdm

from models import WidthAdaptiveConformalInference
from metrics import empirical_coverage, average_length, median_length, winkler_score, pearson_correlation, ILS_coverage, mean_deviation_coverage_by_interval_length

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
alphas = [0.2]

for alpha in alphas:
    if not os.path.exists(f"Results//df_predictions_sigma_impact_alpha_{alpha}.csv"):
        df = pd.read_csv(f"Data//df_quantile_regression_alpha_{alpha}.csv")
        df['full_date'] = pd.to_datetime(df.full_date)
        df['length_hqr'] = df['preds_sup_hqr'] - df['preds_inf_hqr']

        df_predictions = pd.DataFrame()
        df_predictions['real'] = df[df.index >= idx_date_test].real
        df_predictions['preds_inf_hqr'] = df[df.index >= idx_date_test].preds_inf_hqr
        df_predictions['preds_sup_hqr'] = df[df.index >= idx_date_test].preds_sup_hqr
        for sigma in tqdm(np.arange(0.1, 200.1, 0.1)):

            # Width Adaptive Conformal Inference over HQR
            preds_inf_waci_hqr, preds_sup_waci_hqr = WidthAdaptiveConformalInference(idx_date_test, df, alpha, gamma = 0.02, sigma=sigma, method='hqr', by_hour=False, save_alphas=True)
            df.loc[df.tail(len(preds_inf_waci_hqr)).index, 'preds_inf_waci_hqr'] = preds_inf_waci_hqr
            df.loc[df.tail(len(preds_sup_waci_hqr)).index, 'preds_sup_waci_hqr'] = preds_sup_waci_hqr
            df.loc[df.preds_inf_waci_hqr == -np.inf, 'preds_inf_waci_hqr'] = df[df.index < idx_date_test].preds_inf_hqr.min()
            df.loc[df.preds_sup_waci_hqr == np.inf, 'preds_sup_waci_hqr'] = df[df.index < idx_date_test].preds_sup_hqr.max()

            df_predictions[f'preds_inf_waci_hqr_{sigma}'] = df[df.index >= idx_date_test].preds_inf_waci_hqr
            df_predictions[f'preds_sup_waci_hqr_{sigma}'] = df[df.index >= idx_date_test].preds_sup_waci_hqr
        
        df_predictions.to_csv(f"Results//df_predictions_sigma_impact_alpha_{alpha}.csv", index=False)
    else:
        df_predictions = pd.read_csv(f"Results//df_predictions_sigma_impact_alpha_{alpha}.csv")
    
    df_results = pd.DataFrame(columns=['empirical_coverage', 'average_length', 'median_length', 'winkler_score', 'pearson_correlation', 'ILS_10_coverage', 'mean_deviation_coverage_by_interval_length'])
    for sigma in tqdm(np.arange(0.1, 200.1, 0.1)):
        df_results.loc[sigma] = [empirical_coverage(df_predictions['real'], df_predictions[f'preds_sup_waci_hqr_{sigma}'], df_predictions[f'preds_inf_waci_hqr_{sigma}']),
                                average_length(df_predictions[f'preds_sup_waci_hqr_{sigma}'], df_predictions[f'preds_inf_waci_hqr_{sigma}']),
                                median_length(df_predictions[f'preds_sup_waci_hqr_{sigma}'], df_predictions[f'preds_inf_waci_hqr_{sigma}']),
                                winkler_score(df_predictions['real'], df_predictions[f'preds_sup_waci_hqr_{sigma}'], df_predictions[f'preds_inf_waci_hqr_{sigma}'], alpha),
                                pearson_correlation(df_predictions['real'], df_predictions[f'preds_sup_waci_hqr_{sigma}'], df_predictions[f'preds_inf_waci_hqr_{sigma}']),
                                ILS_coverage(df_predictions, f'waci_hqr_{sigma}', pct_threshold=10),
                                mean_deviation_coverage_by_interval_length(df_predictions, f'waci_hqr_{sigma}', 0.05, alpha)
                                ]
        
    df_results.to_csv(f"Results//df_results_sigma_impact_alpha_{alpha}.csv", index=True)













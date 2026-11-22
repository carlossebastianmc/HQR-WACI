from metrics import (winkler_score,
                     empirical_coverage,
                     average_length,
                     median_length,
                     pearson_correlation,
                     spearman_correlation,
                     ILS_coverage,
                     mean_deviation_coverage_by_interval_length,
                     error_by_interval_length_plot
                     )

import pandas as pd
from tqdm import tqdm
import datetime as dt
from arch.bootstrap import StationaryBootstrap
import numpy as np
import warnings
warnings.filterwarnings("ignore")
alphas = [0.2, 0.1]
date_test = dt.datetime(2024, 1, 1)

for alpha in alphas:
    df = pd.read_csv(f"Data//df_final_alpha_{alpha}.csv")
    df['full_date'] = pd.to_datetime(df.full_date)

    idx_date_test = df[pd.to_datetime(df.date) == date_test].head(1).index.values[0]
    df = df[df.index >= idx_date_test]

    # models = ['hqr', 'hqr_by_hour', 'hqr_w', 'hqr_w_by_hour', 'qra', 'qra_by_hour', 
    #           'cqr_hqr', 'cqr_hqr_by_hour', 'cqr_hqr_w', 'cqr_hqr_w_by_hour', 'cqr_qra', 'cqr_qra_by_hour',
    #           'aci_hqr', 'aci_hqr_by_hour', 'aci_hqr_w', 'aci_hqr_w_by_hour', 'aci_qra', 'aci_qra_by_hour',
    #           'waci_hqr', 'waci_hqr_by_hour', 'waci_hqr_w', 'waci_hqr_w_by_hour', 'waci_qra', 'waci_qra_by_hour']

    # models_by_hour = ['hqr_by_hour', 'hqr_w_by_hour', 'qra_by_hour',
    #         # 'cqr_hqr_by_hour', 'cqr_hqr_w_by_hour', 'cqr_qra_by_hour',
    #         'aci_hqr_by_hour', 'aci_hqr_w_by_hour', 'aci_qra_by_hour',
    #         'waci_hqr_by_hour', 'waci_hqr_w_by_hour', 'waci_qra_by_hour']

    models_without_by_hour = ['hqr', 'hqr_w', 'qra', 
            # 'cqr_hqr', 'cqr_hqr_w', 'cqr_qra', 
            'aci_hqr', 'aci_hqr_w', 'aci_qra',
            'waci_hqr', 'waci_hqr_w', 'waci_qra',
            'dcp_hqr', 'dcp_hqr_w', 'dcp_qra']


    df_results = pd.DataFrame(columns=['empirical_coverage', 'average_length', 'median_length', 'winkler_score', 'pearson_correlation', 'ILS_10_coverage', 'mean_deviation_coverage_by_interval_length', 'spearman_correlation', 'interval_length_std'])
    for model in tqdm(models_without_by_hour, desc=f"Alpha: {alpha}"):
        df_results.loc[model] = [empirical_coverage(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                average_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                median_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                winkler_score(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}'], alpha),
                                pearson_correlation(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}']),
                                ILS_coverage(df, model, pct_threshold=90, alpha=alpha),
                                mean_deviation_coverage_by_interval_length(df, model, 0.05, alpha),
                                spearman_correlation((df['real']-df['mean_pred']).abs(), df[f"preds_sup_{model}"], df[f"preds_inf_{model}"]),
                                (df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']).std()
                                ]
        
        # ILS_coverage(df.copy(), model, pct_threshold=90, alpha=alpha)
    df_results.to_csv(f"Results//df_results_alpha_{alpha}.csv", index=True)
    
        
    # df_winkler_score = pd.DataFrame(columns=models_without_by_hour)
    # df_pearson_correlation = pd.DataFrame(columns=models_without_by_hour)
    # df_spearman_correlation = pd.DataFrame(columns=models_without_by_hour)
    # df_empirical_coverage = pd.DataFrame(columns=models_without_by_hour)
    # df_average_length = pd.DataFrame(columns=models_without_by_hour)
    # df_median_length = pd.DataFrame(columns=models_without_by_hour)
    df_ILS_10_coverage = pd.DataFrame(columns=models_without_by_hour)
    # df_mean_deviation_coverage_by_interval_length = pd.DataFrame(columns=models_without_by_hour)
    # df_interval_length_std = pd.DataFrame(columns=models_without_by_hour)
    

    for model in tqdm(models_without_by_hour, desc=f"Alpha: {alpha}"):
        metrics = {
                # 'empirical_coverage': [],
                # 'average_length': [],
                # 'median_length': [],
                # 'winkler_score': [],
                # 'pearson_correlation': [],
                'ILS_10_coverage': [],
                # 'mean_deviation_coverage_by_interval_length': [],
                # 'spearman_correlation': [],
                # 'interval_length_std': []
        }
        index = df.index.values
        true_values = df['real'].values
        bs = StationaryBootstrap(1000, df, seed=123)

        for data in tqdm(bs.bootstrap(1000), desc='Bootstrap', leave=False, total=1000):
                df_aux = data[0][0]
                # metrics['empirical_coverage'].append(empirical_coverage(df_aux['real'], df_aux[f'preds_sup_{model}'], df_aux[f'preds_inf_{model}']))
                # metrics['average_length'].append(average_length(df_aux[f'preds_sup_{model}'], df_aux[f'preds_inf_{model}']))
                # metrics['median_length'].append(median_length(df_aux[f'preds_sup_{model}'], df_aux[f'preds_inf_{model}']))
                # metrics['winkler_score'].append(winkler_score(df_aux['real'], df_aux[f'preds_sup_{model}'], df_aux[f'preds_inf_{model}'], alpha))
                # metrics['pearson_correlation'].append(pearson_correlation(df_aux['real'], df_aux[f'preds_sup_{model}'], df_aux[f'preds_inf_{model}']))
                metrics['ILS_10_coverage'].append(ILS_coverage(df_aux, model, pct_threshold=90, alpha=alpha))
                # metrics['mean_deviation_coverage_by_interval_length'].append(mean_deviation_coverage_by_interval_length(df_aux, model, 0.05, alpha))
                # metrics['spearman_correlation'].append(spearman_correlation((df_aux['real']-df_aux['mean_pred']).abs(), df_aux[f"preds_sup_{model}"], df_aux[f"preds_inf_{model}"]))
                # metrics['interval_length_std'].append((df_aux[f'preds_sup_{model}'] - df_aux[f'preds_inf_{model}']).std())
        
        # df_winkler_score[model] = metrics['winkler_score']
        # df_pearson_correlation[model] = metrics['pearson_correlation']
        # df_spearman_correlation[model] = metrics['spearman_correlation']
        # df_empirical_coverage[model] = metrics['empirical_coverage']
        # df_average_length[model] = metrics['average_length']
        # df_median_length[model] = metrics['median_length']
        df_ILS_10_coverage[model] = metrics['ILS_10_coverage']
        # df_mean_deviation_coverage_by_interval_length[model] = metrics['mean_deviation_coverage_by_interval_length']
        # df_interval_length_std[model] = metrics['interval_length_std']
        
        
    # df_winkler_score.to_csv(f"Results//df_winkler_score_alpha_{alpha}.csv", index=True)
    # df_pearson_correlation.to_csv(f"Results//df_pearson_correlation_alpha_{alpha}.csv", index=True)
    # df_spearman_correlation.to_csv(f"Results//df_spearman_correlation_alpha_{alpha}.csv", index=True)
    # df_empirical_coverage.to_csv(f"Results//df_empirical_coverage_alpha_{alpha}.csv", index=True)
    # df_average_length.to_csv(f"Results//df_average_length_alpha_{alpha}.csv", index=True)
    # df_median_length.to_csv(f"Results//df_median_length_alpha_{alpha}.csv", index=True)
    df_ILS_10_coverage.to_csv(f"Results//df_ILS_10_coverage_alpha_{alpha}.csv", index=True)
    # df_mean_deviation_coverage_by_interval_length.to_csv(f"Results//df_mean_deviation_coverage_by_interval_length_alpha_{alpha}.csv", index=True)
    # df_interval_length_std.to_csv(f"Results//df_interval_length_std_alpha_{alpha}.csv", index=True)

    df_winkler_score = pd.read_csv(f"Results//df_winkler_score_alpha_{alpha}.csv", index_col=0)
    df_pearson_correlation = pd.read_csv(f"Results//df_pearson_correlation_alpha_{alpha}.csv", index_col=0)
    df_spearman_correlation = pd.read_csv(f"Results//df_spearman_correlation_alpha_{alpha}.csv", index_col=0)
    df_empirical_coverage = pd.read_csv(f"Results//df_empirical_coverage_alpha_{alpha}.csv", index_col=0)
    df_average_length = pd.read_csv(f"Results//df_average_length_alpha_{alpha}.csv", index_col=0)
    df_median_length = pd.read_csv(f"Results//df_median_length_alpha_{alpha}.csv", index_col=0)
    df_ILS_10_coverage = pd.read_csv(f"Results//df_ILS_10_coverage_alpha_{alpha}.csv", index_col=0)
    df_mean_deviation_coverage_by_interval_length = pd.read_csv(f"Results//df_mean_deviation_coverage_by_interval_length_alpha_{alpha}.csv", index_col=0)
    df_interval_length_std = pd.read_csv(f"Results//df_interval_length_std_alpha_{alpha}.csv", index_col=0)
    
    df_standard_deviations_bootstrap = pd.DataFrame(columns=['empirical_coverage', 'average_length', 'median_length', 'winkler_score', 'pearson_correlation', 'ILS_10_coverage', 'mean_deviation_coverage_by_interval_length', 'spearman_correlation', 'interval_length_std'])
    for model in tqdm(models_without_by_hour, desc=f"Alpha: {alpha}"):
        metrics_std = {
            'empirical_coverage': df_empirical_coverage[model].std() / np.sqrt(len(df_empirical_coverage[model])),
            'average_length': df_average_length[model].std() / np.sqrt(len(df_average_length[model])),
            'median_length': df_median_length[model].std() / np.sqrt(len(df_median_length[model])),
            'winkler_score': df_winkler_score[model].std() / np.sqrt(len(df_winkler_score[model])),
            'pearson_correlation': df_pearson_correlation[model].std() / np.sqrt(len(df_pearson_correlation[model])),
            'ILS_10_coverage': df_ILS_10_coverage[model].std() / np.sqrt(len(df_ILS_10_coverage[model])),
            'mean_deviation_coverage_by_interval_length': df_mean_deviation_coverage_by_interval_length[model].std() / np.sqrt(len(df_mean_deviation_coverage_by_interval_length[model])),
            'spearman_correlation': df_spearman_correlation[model].std() / np.sqrt(len(df_spearman_correlation[model])),
            'interval_length_std': df_interval_length_std[model].std() / np.sqrt(len(df_interval_length_std[model]))
        }
        df_standard_deviations_bootstrap.loc[model] = [metrics_std['empirical_coverage'],
                                                        metrics_std['average_length'],
                                                        metrics_std['median_length'],
                                                        metrics_std['winkler_score'],
                                                        metrics_std['pearson_correlation'],
                                                        metrics_std['ILS_10_coverage'],
                                                        metrics_std['mean_deviation_coverage_by_interval_length'],
                                                        metrics_std['spearman_correlation'],
                                                        metrics_std['interval_length_std']
                                                        ]
    df_standard_deviations_bootstrap.to_csv(f"Results//df_standard_deviations_bootstrap_alpha_{alpha}.csv", index=True)

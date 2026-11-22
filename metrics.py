import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif"
# })

# plt.style.use('seaborn-white')
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
colors = sns.color_palette("colorblind")


color_by_method_dict = {'hqr': colors[0], 'qra': colors[1], 'cqr_hqr': colors[2], 'aci_hqr': colors[4], 'waci_hqr': colors[7], 'dcp_hqr': colors[3], 'dcp_hqr_w': colors[5], 'dcp_qra': colors[6]}
label_by_method_dict = {'hqr': 'HQR', 'qra': 'QRA', 'cqr_hqr': 'CQR-HQR', 'aci_hqr': 'ACI-HQR', 'waci_hqr': 'WACI-HQR', 'dcp_hqr': 'DCP-HQR'}
marker_by_method_dict = {'hqr': 'o', 'qra': 's', 'cqr_hqr': 'D', 'aci_hqr': '^', 'waci_hqr': 'v', 'dcp_hqr': 'X'}

def winkler_score(y_true, upper_pred, lower_pred, alpha):
    lower_diff = lower_pred - y_true
    upper_diff = y_true - upper_pred
    interval_length = upper_pred - lower_pred

    lower_score = interval_length + (2 / alpha) * np.maximum(lower_diff, 0)
    upper_score = interval_length + (2 / alpha) * np.maximum(upper_diff, 0)

    score = np.where(y_true < lower_pred, lower_score, np.where(y_true > upper_pred, upper_score, interval_length))

    return np.mean(score)

def empirical_coverage(y_true, upper_pred, lower_pred):
    return np.where((y_true >= lower_pred) & (y_true <= upper_pred), True, False).mean()*100

def average_length(upper_pred, lower_pred):
    return np.mean(upper_pred - lower_pred)

def median_length(upper_pred, lower_pred):
    return np.median(upper_pred - lower_pred)

def pearson_correlation(y_true, upper_pred, lower_pred):
    coverage = (np.where((y_true >= lower_pred) & (y_true <= upper_pred), 1, 0))
    length = (upper_pred - lower_pred)
    return stats.pearsonr(coverage, length)[0]

def spearman_correlation(absolut_errors, upper_pred, lower_pred):
    length = (upper_pred - lower_pred)
    return stats.spearmanr(absolut_errors, length)[0]

def HSIC_original(x, y, s_x=1, s_y=1):
    def pairwise_distances(x):
        instances_norm = np.sum(x**2, axis=-1)[:, np.newaxis]
        return -2 * np.dot(x, x.T) + instances_norm + instances_norm.T

    def GaussianKernelMatrix(x, sigma=1):
        pairwise_distances_ = pairwise_distances(x)
        return np.exp(-pairwise_distances_ / sigma)
    
    m, _ = x.shape
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = np.eye(m) - 1.0 / m * np.ones((m, m))
    HKH = np.dot(H, np.dot(K, H))
    HSIC_value = np.trace(np.dot(L, HKH)) / ((m - 1) ** 2)
    return HSIC_value

def HSIC(y_true, upper_pred, lower_pred, s_coverage=1, s_length=1):
    coverage = np.where((y_true >= lower_pred) & (y_true <= upper_pred), 1, 0).reshape(-1, 1)
    length = (upper_pred - lower_pred).to_numpy().reshape(-1, 1)
    return HSIC_original(coverage, length, s_x=s_coverage, s_y=s_length)

def ILS_coverage(df, model, pct_threshold=90, alpha=0.2):
    if model == 'hqr' or model == 'qra' or model == 'hqr_w' or model == 'hqr_by_hour' or model == 'qra_by_hour' or model == 'hqr_w_by_hour':
        return np.nan
    elif model == 'cqr_hqr' or model == 'aci_hqr' or model == 'waci_hqr' or model == 'dcp_hqr' or "hqr" in model or model == 'cqr_hqr_by_hour' or model == 'aci_hqr_by_hour' or model == 'waci_hqr_by_hour':
        base_model = 'hqr'
        interval_lengths_base_model = df[f'preds_sup_{base_model}'] - df[f'preds_inf_{base_model}']
    elif model == 'cqr_qra' or model == 'aci_qra' or model == 'waci_qra' or model == 'dcp_qra' or model == 'cqr_qra_by_hour' or model == 'aci_qra_by_hour' or model == 'waci_qra_by_hour':
        base_model = 'qra'
        interval_lengths_base_model = df[f'preds_sup_{base_model}'] - df[f'preds_inf_{base_model}']
    elif model == 'cqr_hqr_w' or model == 'aci_hqr_w' or model == 'waci_hqr_w' or model == 'dcp_hqr_w' or model == 'cqr_hqr_w_by_hour' or model == 'aci_hqr_w_by_hour' or model == 'waci_hqr_w_by_hour':
        base_model = 'hqr_w'
        interval_lengths_base_model = df[f'preds_sup_{base_model}'] - df[f'preds_inf_{base_model}']
    else:
        base_model = 'initial'
        interval_lengths_base_model = df[f'preds_sup_{base_model}'] - df[f'preds_inf_{base_model}']
    
    lengths = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    # coverage = empirical_coverage(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}'])

    threshold_diff_lengths = np.quantile((lengths - interval_lengths_base_model).abs(), pct_threshold/100)

    df_ils = df[(lengths - interval_lengths_base_model).abs() > threshold_diff_lengths]
    # df['apply_ils_' + model] = 0
    # df.loc[df.index.isin(df_ils.index), 'apply_ils_' + model] = 1
    # df.to_csv(f"Results//df_ils_{model}_pct_threshold_{pct_threshold}_alpha_{alpha}.csv", index=False)
    coverage_ils = empirical_coverage(df_ils['real'], df_ils[f'preds_sup_{model}'], df_ils[f'preds_inf_{model}'])
    ils_diff = np.abs(coverage_ils - (1-alpha)*100)

    return ils_diff

def mean_deviation_coverage_by_interval_length(df, model, quantile_step, alpha):
    df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)]}
    dict_coverages = {model: []}
    for i_quantile in range(len(dict_lengths[model])-1):
        length_min_model, length_max_model = dict_lengths[model][i_quantile], dict_lengths[model][i_quantile+1]
        df_aux_model = df[(df[f'length_{model}'] >= length_min_model) & (df[f'length_{model}'] <= length_max_model)]
        dict_coverages[model].append(empirical_coverage(df_aux_model['real'], df_aux_model[f'preds_sup_{model}'], df_aux_model[f'preds_inf_{model}']))
    return np.mean(np.abs(np.array(dict_coverages[model]) - (1-alpha)*100))

def coverage_vs_interval_length_plot(df, models, alpha):
    obj_coverage = (1-alpha)*100
    fig, ax = plt.subplots(figsize=(10, 4))
    for model in models:
        coverage_model = empirical_coverage(df['real'], df[f'preds_sup_{model}'], df[f'preds_inf_{model}'])
        average_length_model = average_length(df[f'preds_sup_{model}'], df[f'preds_inf_{model}'])

        ax.scatter(coverage_model, average_length_model, label=label_by_method_dict[model], color=color_by_method_dict[model], marker=marker_by_method_dict[model], s=100)
    ax.axvline(obj_coverage, color='k', lw=2, linestyle='--')
    ax.grid()
    ax.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_title(f"Coverage vs interval length $\\alpha = {alpha}$", fontsize=16)
    ax.set_xlabel("Empirical Coverage", fontsize=13)
    ax.set_ylabel("Average Interval Length", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"Results//coverage_vs_interval_length_alpha_{alpha}.pdf", bbox_inches='tight', format='pdf')

def coverage_by_interval_length_plot(df, models, alpha, quantile_step, bar_width = 0.1):
    df_coverage_by_interval_length = pd.DataFrame(columns=['mean_abs_deviation', 'max_abs_deviation'])
    obj_coverage = (1-alpha)*100

    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_coverages = {model: [] for model in models}
    labels = []
    for i_quantile in range(len(dict_lengths[models[0]])-1):
        labels.append(str((round(np.arange(0, 1.01, quantile_step)[i_quantile], 2), round(np.arange(0, 1.01, quantile_step)[i_quantile+1], 2))))
        for model in models:
            length_min_model, length_max_model = dict_lengths[model][i_quantile], dict_lengths[model][i_quantile+1]
            df_aux_model = df[(df[f'length_{model}'] >= length_min_model) & (df[f'length_{model}'] <= length_max_model)]
            dict_coverages[model].append(empirical_coverage(df_aux_model['real'], df_aux_model[f'preds_sup_{model}'], df_aux_model[f'preds_inf_{model}']))
    
    for model in models:
        df_coverage_by_interval_length.loc[model] = [np.mean(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.max(np.abs(np.array(dict_coverages[model]) - obj_coverage))]
    df_coverage_by_interval_length.to_csv(f"Results//df_coverage_by_interval_length_alpha_{alpha}_quantile_step_{quantile_step}.csv", index=True)


    fig, ax = plt.subplots(figsize=(10, 4))
    bar_heights = list(dict_coverages.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.set_yticks(np.arange(10, 101, 5))
    ax.tick_params(axis='y', labelsize=12)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.axhline(np.round(obj_coverage), color='k', lw=2)
    ax.set_title(f"Coverage by interval length $\\alpha = {alpha}$", fontsize=16)
    ax.set_xlabel("Quantile (IW)", fontsize=13)
    ax.set_ylabel("Empirical Coverage", fontsize=13)
    if obj_coverage > 98:
        ax.set_ylim(obj_coverage-5, 100)
    elif obj_coverage > 94:
        ax.set_ylim(obj_coverage-15, 100)
    elif obj_coverage > 89:
        ax.set_ylim(obj_coverage-25, 100)
    else:
        ax.set_ylim(obj_coverage-35, 100)
    plt.tight_layout()
    plt.savefig(f"Results//coverage_by_interval_length_alpha_{alpha}_quantile_step_{quantile_step}.pdf", bbox_inches='tight', format='pdf')

def error_by_interval_length_plot(df, models, quantile_step, alpha):
    df['error_abs'] = (df['real'] - df['mean_pred']).abs()
    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(0, 1.01, quantile_step)] for model in models}
    dict_abs_error = {model: [] for model in models}
    labels = []
    for i_quantile in range(len(dict_lengths[models[0]])-1):
        labels.append(f"({round(np.arange(0, 1.01, quantile_step)[i_quantile], 2)}, {round(np.arange(0, 1.01, quantile_step)[i_quantile+1], 2)})")        
        for model in models:
            length_min_model, length_max_model = dict_lengths[model][i_quantile], dict_lengths[model][i_quantile+1]
            df_aux_model = df[(df[f'length_{model}'] >= length_min_model) & (df[f'length_{model}'] <= length_max_model)]
            dict_abs_error[model].append(df_aux_model.error_abs.mean())

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_heights = list(dict_abs_error.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=90)
    ax.tick_params(axis='y', labelsize=12)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_title(f"MAE by interval length $\\alpha = {alpha}$", fontsize=16)
    ax.set_xlabel("Quantile (IW)", fontsize=13)
    ax.set_ylabel("MAE", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"Results//error_by_interval_length_alpha_{alpha}.pdf", bbox_inches='tight', format='pdf')

def error_by_interval_length_plot_2(df, alpha):
    df['error_abs'] = (df['real'] - df['mean_pred']).abs()
    for model in ['aci_hqr', 'waci_hqr']:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(df[f'length_aci_hqr'], df['error_abs'], label=label_by_method_dict['aci_hqr'], color=colors[0], alpha=0.5)
    ax[1].scatter(df[f'length_waci_hqr'], df['error_abs'], label=label_by_method_dict['waci_hqr'], color=colors[1], alpha=0.5)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel("Interval Length")
    ax[1].set_xlabel("Interval Length")
    ax[0].set_ylabel("MAE")
    ax[1].set_ylabel("MAE")
    plt.tight_layout()
    plt.savefig(f"Results//error_by_interval_length_alpha_{alpha}_scatter.pdf", bbox_inches='tight', format='pdf')

def interval_length_distribution_plot(df, models, quantile_step, alpha):
    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_lengths = {model: [np.quantile(df[f'length_{model}'], i) for i in np.arange(quantile_step, 1-quantile_step+0.0001, quantile_step)] for model in models}

    fig, ax = plt.subplots(figsize=(10, 4))
    n = len(models)
    for model in models:
        ax.plot(np.arange(quantile_step, 1-quantile_step+0.0001, quantile_step), dict_lengths[model], "-o", label=label_by_method_dict[model], color=color_by_method_dict[model])
    ax.grid()
    ax.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.set_title(f"Interval length distribution $\\alpha = {alpha}$", fontsize=16)
    ax.set_xlabel("Quantile", fontsize=13)
    ax.set_ylabel("Interval length", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Results//interval_length_distribution_alpha_{alpha}.pdf", bbox_inches='tight', format='pdf')

def coverage_by_hour_plot(df, models, alpha, bar_width = 0.1):
    df_coverage_by_hour = pd.DataFrame(columns=['mean_abs_deviation', 'max_abs_deviation', 'std_empirical_coverage'])
    obj_coverage = (1-alpha)*100

    for model in models:
        df['length_' + model] = df[f'preds_sup_{model}'] - df[f'preds_inf_{model}']
    dict_coverages = {model: [] for model in models}
    labels = []
    for hour in range(1, 25):
        labels.append(str(hour))
        for model in models:
            df_aux_model = df[df.hour == hour]
            dict_coverages[model].append(empirical_coverage(df_aux_model['real'], df_aux_model[f'preds_sup_{model}'], df_aux_model[f'preds_inf_{model}']))
    
    for model in models:
        df_coverage_by_hour.loc[model] = [np.mean(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.max(np.abs(np.array(dict_coverages[model]) - obj_coverage)), np.std(dict_coverages[model], ddof=0)]
    df_coverage_by_hour.to_csv(f"Results//df_coverage_by_hour_alpha_{alpha}.csv", index=True)


    fig, ax = plt.subplots(figsize=(10, 4))
    bar_heights = list(dict_coverages.values())
    n = len(models)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    if obj_coverage > 98:
        ax.set_ylim(obj_coverage-3, 100)
    elif obj_coverage > 94:
        ax.set_ylim(obj_coverage-5, 100)
    elif obj_coverage > 89:
        ax.set_ylim(obj_coverage-6, 100)
    else:
        ax.set_ylim(obj_coverage-12, 100)
    for i in range(n):
        ax.plot(x, bar_heights[i], '--o', label=label_by_method_dict[models[i]], color=color_by_method_dict[models[i]])
    ax.grid()
    ax.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax.axhline(np.round(obj_coverage), color='k', lw=2)
    ax.set_title(f"Coverage by hour $\\alpha = {alpha}$", fontsize=16)
    ax.set_xlabel("Hour", fontsize=13)
    ax.set_ylabel("Empirical Coverage", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"Results//coverage_by_hour_alpha_{alpha}.pdf", bbox_inches='tight', format='pdf')


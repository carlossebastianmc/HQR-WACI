import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from os.path import dirname, abspath
import datetime as dt
import pickle
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import os



# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif"
# })
# colors = list(mcolors.TABLEAU_COLORS.values())

# Estilo de los gráficos tipo LaTex
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

def generate_expected_error_3d_plot():
    d = dirname(abspath(__file__))

    mu1 = np.array([-1., 2.])
    sigma1 = np.array([[1., .5], [.5, 1.]])

    mu2 = np.array([2., -1.])
    sigma2 = np.array([[1.5, .5], [.5, 1.5]])

    mu3 = np.array([-2., -2.])
    sigma3 = np.array([[0.5, .2], [.2, 0.5]])

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)

    rv1 = multivariate_normal(mu1, sigma1)
    rv2 = multivariate_normal(mu2, sigma2)
    rv3 = multivariate_normal(mu3, sigma3)
    z = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(y)):
            pos = np.array([x[i, j], y[i, j]])
            z[i, j] = rv1.pdf(pos) + rv2.pdf(pos) + rv3.pdf(pos)
    z = z / np.sum(z)
    pos = np.dstack((x, y))


    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x, y, z, cmap='viridis')
    ax1.set_title("Distribution of explanatory features", fontsize=14, fontname='serif')

    z2 = 1 - z
    z2 = (z2 - np.min(z2)) / (np.max(z2) - np.min(z2))

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, z2, cmap='viridis')
    ax2.set_title("Expected error of a point forecaster by region", fontsize=14, fontname='serif')

    ax2.set_zticks([0, 1])
    ax2.set_zticklabels(["  Min Error", "  Max Error"], fontsize=10)
    # plt.show()
    plt.savefig(d + "\\Images\\expected_error_3d_plot.pdf")

def generate_day_ahead_market_figure(test_date=dt.datetime(2024, 1, 1)):
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df_2024.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df['full_date'], df['real'], color=colors[0])
    ax.plot(df.loc[df[df.full_date > test_date].index - 6*30*24, 'full_date'], df.loc[df[df.full_date > test_date].index - 6*30*24, 'real'], color=colors[1])
    ax.plot(df.loc[df[df.full_date > test_date].index, 'full_date'], df.loc[df[df.full_date > test_date].index, 'real'], color=colors[2])
    ax.axvline(x=test_date, color='k', linestyle='--', lw=2)
    ax.axvline(x=test_date - dt.timedelta(days=6*30), color='k', linestyle='--', lw=2)
    ax.grid()
    # ax.set_title("Spanish Day-Ahead market price", fontsize=16)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Price (€/MWh)", fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.text(test_date - dt.timedelta(days=10), 200, "Quantile Regression predictions", ha='right', va='top', rotation=0, fontsize=8)
    ax.annotate("", xy=(test_date - dt.timedelta(days=5), 180), xytext=(test_date - dt.timedelta(days=6*30), 180),
                arrowprops=dict(arrowstyle="->"))
    
    ax.text(test_date + dt.timedelta(days=6*30) - dt.timedelta(days=15), 200, "Conformal prediction methods", ha='right', va='top', rotation=0, fontsize=8)
    ax.annotate("", xy=(test_date + dt.timedelta(days=6*30), 180), xytext=(test_date, 180),
                arrowprops=dict(arrowstyle="->"))

    plt.savefig(d + "\\Images\\day_ahead_market_figure.pdf", format='pdf', bbox_inches='tight')
    plt.tight_layout()
    # plt.show()
    
def generate_day_ahead_market_predictions():
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + "\\Data\\df_2024.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])

    df_aux = df[(pd.to_datetime(df.date) >= dt.datetime(2024, 5, 5)) & (pd.to_datetime(df.date) < dt.datetime(2024, 5, 5) + dt.timedelta(days=7))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(df_aux['full_date'], df_aux['real'], label = 'DA market price', color=colors[0])
    ax.plot(df_aux['full_date'], df_aux['pred1'], '--', label='Model 1', color=colors[1])
    ax.plot(df_aux['full_date'], df_aux['pred2'], '--', label='Model 2', color=colors[2])
    ax.plot(df_aux['full_date'], df_aux['pred3'], '--', label='Model 3', color=colors[4])
    # ax.plot(df_aux['full_date'], df_aux['pred4'], '--', label='Model 4', color=colors[7])
    for date in df_aux.date.unique():
        ax.axvline(x=pd.to_datetime(date), color='k', linestyle='--', lw=1)
    ax.grid()
    # ax.set_title("Spanish Day-Ahead market price forecasts", fontsize=16)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Price (€/MWh)", fontsize=13)
    ax.legend(prop={'size': 12}, loc='upper right', frameon=True)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    plt.savefig(d + "\\Images\\day_ahead_market_predictions.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


def generate_progression_alphas(hour=13):
    d = dirname(abspath(__file__))
    with open('dict_list_alphas_aci.pkl', 'rb') as file:
        list_alphas_aci = pickle.load(file)

    with open('dict_list_alphas_waci.pkl', 'rb') as file:
        list_alphas_waci = pickle.load(file)

    list_alphas_aci = list_alphas_aci[hour]
    list_alphas_waci = list_alphas_waci[hour]

    for i in tqdm(range(len(list_alphas_aci))):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
        ax.plot(np.arange(0, 500, 0.1), list_alphas_waci[i], color=colors[0], label='WACI', lw=1)
        ax.axhline(list_alphas_aci[i], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
        ax.set_xlabel('Interval length', fontsize=14)
        ax.set_ylabel("$\\alpha_t$", fontsize=14)
        ax.grid()
        ax.set_ylim(0.1, 0.3)
        ax.set_xlim(0, 100)
        ax.legend(prop={'size': 14})
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()
        plt.savefig(d + f"\\Images\\alpha_progression_3\\alpha_progression_hour{hour}_{i}.pdf", format='pdf', bbox_inches='tight')
    
def generate_coefs_variation():
    d = dirname(abspath(__file__))
    with open('dict_coefs_hqr_alpha_0.05.pkl', 'rb') as file:
        dict_coefs_cfqra_005 = pickle.load(file)
    with open('dict_coefs_hqr_alpha_0.1.pkl', 'rb') as file:
        dict_coefs_cfqra_01 = pickle.load(file)
    with open('dict_coefs_hqr_alpha_0.2.pkl', 'rb') as file:
        dict_coefs_cfqra_02 = pickle.load(file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(dict_coefs_cfqra_005['inf'], label='$\\alpha=0.05$', color=colors[0])
    ax.plot(dict_coefs_cfqra_005['sup'], '--', color=colors[0])
    ax.plot(dict_coefs_cfqra_01['inf'], label='$\\alpha=0.10$', color=colors[1])
    ax.plot(dict_coefs_cfqra_01['sup'], '--', color=colors[1])
    ax.plot(dict_coefs_cfqra_02['inf'], label='$\\alpha=0.20$', color=colors[2])
    ax.plot(dict_coefs_cfqra_02['sup'], '--', color=colors[2])

    ax.set_title("$\\hat{\\lambda}_{2}(\\frac{\\alpha}{2})$ and $\\hat{\\lambda}_{2}(1-\\frac{\\alpha}{2})$ over time", fontsize=16)
    ax.set_xlabel("T+1", fontsize=13)
    ax.set_ylabel("Coefficient value", fontsize=13)
    ax.legend(prop={'size': 12}, loc='upper left', frameon=True, bbox_to_anchor=(1, 1))
    ax.grid()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(d+"\\Images\\coefs_variation.pdf", format='pdf', bbox_inches='tight')

def generate_coefs_models_qra():
    d = dirname(abspath(__file__))
    with open('dict_coefs_qra_alpha_0.01.pkl', 'rb') as file:
        dict_coefs_cfqra_001 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.05.pkl', 'rb') as file:
        dict_coefs_cfqra_005 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.1.pkl', 'rb') as file:
        dict_coefs_cfqra_01 = pickle.load(file)
    with open('dict_coefs_qra_alpha_0.2.pkl', 'rb') as file:
        dict_coefs_cfqra_02 = pickle.load(file)
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 16))
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred1'], '--', color=colors[0])
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred2'], '--', color=colors[1])
    ax[0].plot(dict_coefs_cfqra_001['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[0].plot(dict_coefs_cfqra_001['sup']['pred3'], '--', color=colors[2])
    ax[0].set_title("$\\alpha = 0.01$", fontsize=16)
    ax[0].set_xlabel("T+1", fontsize=13)
    ax[0].set_ylabel("Coefficient value", fontsize=13)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(dict_coefs_cfqra_005['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred1'], '--', color=colors[0])
    ax[1].plot(dict_coefs_cfqra_005['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred2'], '--', color=colors[1])
    ax[1].plot(dict_coefs_cfqra_005['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[1].plot(dict_coefs_cfqra_005['sup']['pred3'], '--', color=colors[2])
    ax[1].set_title("$\\alpha = 0.05$", fontsize=16)
    ax[1].set_xlabel("T+1", fontsize=13)
    ax[1].set_ylabel("Coefficient value", fontsize=13)
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(dict_coefs_cfqra_01['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred1'], '--', color=colors[0])
    ax[2].plot(dict_coefs_cfqra_01['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred2'], '--', color=colors[1])
    ax[2].plot(dict_coefs_cfqra_01['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[2].plot(dict_coefs_cfqra_01['sup']['pred3'], '--', color=colors[2])
    ax[2].set_title("$\\alpha = 0.10$", fontsize=16)
    ax[2].set_xlabel("T+1", fontsize=13)
    ax[2].set_ylabel("Coefficient value", fontsize=13)
    ax[2].tick_params(axis='x', labelsize=12)
    ax[2].tick_params(axis='y', labelsize=12)
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(dict_coefs_cfqra_02['inf']['pred1'], label='Model 1', color=colors[0])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred1'], '--', color=colors[0])
    ax[3].plot(dict_coefs_cfqra_02['inf']['pred2'], label='Model 2', color=colors[1])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred2'], '--', color=colors[1])
    ax[3].plot(dict_coefs_cfqra_02['inf']['pred3'], label='Model 3', color=colors[2])
    # ax[3].plot(dict_coefs_cfqra_02['sup']['pred3'], '--', color=colors[3])
    ax[3].set_title("$\\alpha = 0.20$", fontsize=16)
    ax[3].set_xlabel("T+1", fontsize=13)
    ax[3].set_ylabel("Coefficient value", fontsize=13)
    ax[3].tick_params(axis='x', labelsize=12)
    ax[3].tick_params(axis='y', labelsize=12)
    ax[3].legend()
    ax[3].grid()

    plt.tight_layout()
    # plt.show()
    plt.savefig(d+"\\Images\\coefs_variation_qra.pdf", format='pdf', bbox_inches='tight')

def generate_progression_alphas_2(hour=13):
    d = dirname(abspath(__file__))
    with open('dict_list_alphas_aci.pkl', 'rb') as file:
        list_alphas_aci = pickle.load(file)

    with open('dict_list_alphas_waci.pkl', 'rb') as file:
        list_alphas_waci = pickle.load(file)
    
    with open('dict_list_alphas_waci_2.pkl', 'rb') as file:
        list_alphas_waci_2 = pickle.load(file)

    list_alphas_aci = list_alphas_aci[hour]
    list_alphas_waci = list_alphas_waci[hour]
    list_alphas_waci_2 = list_alphas_waci_2[hour]

    # for i in tqdm(range(len(list_alphas_aci))):
    i = 366
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
    ax.plot(np.arange(0, 500, 0.1), list_alphas_waci[i], color=colors[0], label='WACI', lw=1)
    ax.axhline(list_alphas_aci[i], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
    ax.plot(np.arange(0, 500, 0.1), list_alphas_waci_2[i], color=colors[2], label='WACI 2', lw=1)
    # ax.set_xlabel('Interval length', fontsize=13)
    # ax.set_ylabel("$\\alpha_t$", fontsize=13)
    ax.set_xlabel('Interval length')
    ax.set_ylabel("$\\alpha_{366}$")
    ax.grid()
    ax.set_ylim(0.1, 0.3)
    ax.set_xlim(0, 100)
    # ax.legend(prop={'size':12}, frameon=True, loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(frameon=True)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.savefig(d + f"\\Images\\alpha_progression_4\\alpha_progression_hour{hour}_{i}_2.pdf")

colors_by_model = {
    'hqr': colors[0],
    'hqr_w': colors[1],
    'qra': colors[2],
    'aci_hqr': colors[0],
    'aci_hqr_w': colors[1],
    'aci_qra': colors[2],
    'waci_hqr': colors[0],
    'waci_hqr_w': colors[1],
    'waci_qra': colors[2],
    'dcp_hqr': colors[0],
    'dcp_hqr_w': colors[1],
    'dcp_qra': colors[2],
}

labels_by_model = {
    'hqr': 'HQR',
    'hqr_w': 'HQR-W',
    'qra': 'QRA',
    'aci_hqr': 'ACI-HQR',
    'aci_hqr_w': 'ACI-HQR-W',
    'aci_qra': 'ACI-QRA',
    'waci_hqr': 'WACI-HQR',
    'waci_hqr_w': 'WACI-HQR-W',
    'waci_qra': 'WACI-QRA',
    'dcp_hqr': 'DCP-HQR',
    'dcp_hqr_w': 'DCP-HQR-W',
    'dcp_qra': 'DCP-QRA',
}

markers_by_model = {
    'hqr': 'o',
    'hqr_w': 'o',
    'qra': 'o',
    'aci_hqr': 's',
    'aci_hqr_w': 's',
    'aci_qra': 's',
    'waci_hqr': '^',
    'waci_hqr_w': '^',
    'waci_qra': '^',
    'dcp_hqr': 'X',
    'dcp_hqr_w': 'X',
    'dcp_qra': 'X'
    }

def generate_average_interval_length_vs_mean_empirical_coverage_plot():
    d = dirname(abspath(__file__))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax_number = 0
    for alpha in [0.1, 0.2]:
        ax[ax_number].grid(zorder=0)
        df = pd.read_csv(d + f"\\Results\\df_results_alpha_{alpha}.csv", index_col=0)
        obj_coverage = 100*(1 - alpha)

        df = df.loc[['hqr', 'hqr_w', 'qra', 'aci_hqr', 'aci_hqr_w', 'aci_qra', 'waci_hqr', 'waci_hqr_w', 'waci_qra', 'dcp_hqr', 'dcp_hqr_w', 'dcp_qra']]

        # df_base_models = df.loc[['hqr', 'hqr_w', 'qra']]
        # df_aci_models = df.loc[['aci_hqr', 'aci_hqr_w', 'aci_qra']]
        # df_waci_models = df.loc[['waci_hqr', 'waci_hqr_w', 'waci_qra']]

        # ax[ax_number].plot(df_base_models['empirical_coverage'],df_base_models['average_length'], color='grey', alpha=0.5, zorder=1)
        # ax[ax_number].plot(df_aci_models['empirical_coverage'],df_aci_models['average_length'], color='grey', alpha=0.5, zorder=1)
        # ax[ax_number].plot(df_waci_models['empirical_coverage'],df_waci_models['average_length'], color='grey', alpha=0.5, zorder=1)

        for model in df.index:
            ax[ax_number].scatter(df.loc[model, 'empirical_coverage'], df.loc[model, 'average_length'], label=labels_by_model[model], color=colors_by_model[model], marker=markers_by_model[model], zorder=2)
            if model == 'waci_hqr':
                circle = plt.Circle((df.loc[model, 'empirical_coverage'], df.loc[model, 'average_length']), 0.12, color='red', fill=False, lw=2, zorder=3)
                ax[ax_number].add_patch(circle)
        ax[ax_number].axvline(obj_coverage, color='k', linestyle='--', lw=1.2, label='Objective coverage', zorder=1)
        ax[ax_number].set_xlabel("Mean empirical coverage")
        ax[ax_number].set_ylabel("Average interval length")
        ax[ax_number].set_title(f"$\\alpha={alpha}$")
        if ax_number == 1:
            ax[ax_number].legend(frameon=True, loc='upper left', bbox_to_anchor=(1, 1))
            
        ax_number += 1
    plt.tight_layout()
    plt.savefig(d + "\\Images\\average_interval_length_vs_mean_empirical_coverage.pdf", format='pdf', bbox_inches='tight')


def generate_marginal_coverage_vs_conditional_coverage():
    n = 1000 
    X = np.sort(np.random.uniform(0, 10, n))
    y_true = np.random.normal(0, X)  

    y_upper_marginal = 11.5
    y_lower_marginal = -11.5

    y_pred_lower = -1.95*X
    y_pred_upper = 1.95*X

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].scatter(X, y_true, alpha=0.9, zorder=2, label='Data')
    ax[0].fill_between(X, y_lower_marginal, y_upper_marginal, alpha=0.5, zorder=1, label='Prediction interval')
    ax[0].set_title("Marginal coverage", fontsize=16)
    ax[0].set_xlabel("X", fontsize=13)
    ax[0].set_ylabel("Y", fontsize=13)
    ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].legend(frameon=True)

    ax[1].scatter(X, y_true, alpha=0.9, zorder=2, label='Data')
    ax[1].fill_between(X, y_pred_lower, y_pred_upper, alpha=0.5, zorder=1, label='Prediction interval')
    ax[1].set_title("Conditional coverage", fontsize=16)
    ax[1].set_xlabel("X", fontsize=13)
    ax[1].set_ylabel("Y", fontsize=13)
    ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig("Images\\marginal_vs_conditional_coverage.pdf", format='pdf', bbox_inches='tight')

def generate_progression_alphas_square(hour=13):
    d = dirname(abspath(__file__))
    with open('dict_list_alphas_aci.pkl', 'rb') as file:
        list_alphas_aci = pickle.load(file)

    with open('dict_list_alphas_waci.pkl', 'rb') as file:
        list_alphas_waci = pickle.load(file)

    list_alphas_aci = list_alphas_aci[hour]
    list_alphas_waci = list_alphas_waci[hour]

    first_iteration = 0
    second_iteration = 1
    third_iteration = 100
    fourth_iteration = 320

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[0, 0].axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
    ax[0, 0].plot(np.arange(0, 500, 0.1), list_alphas_waci[first_iteration], color=colors[0], label='WACI', lw=1)
    ax[0, 0].axhline(list_alphas_aci[first_iteration], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
    ax[0, 0].set_xlabel('Interval length')
    ax[0, 0].set_ylabel("$\\alpha_1$")
    ax[0, 0].grid()
    ax[0, 0].set_ylim(0.1, 0.3)
    ax[0, 0].set_xlim(0, 100)
    ax[0, 0].legend(frameon=True, prop={'size': 8})
    ax[0, 0].set_title("$T+1 = 1$")

    ax[0, 1].axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
    ax[0, 1].plot(np.arange(0, 500, 0.1), list_alphas_waci[second_iteration], color=colors[0], label='WACI', lw=1)
    ax[0, 1].axhline(list_alphas_aci[second_iteration], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
    ax[0, 1].set_xlabel('Interval length')
    ax[0, 1].set_ylabel("$\\alpha_2$")
    ax[0, 1].grid()
    ax[0, 1].set_ylim(0.1, 0.3)
    ax[0, 1].set_xlim(0, 100)
    ax[0, 1].legend(frameon=True, prop={'size': 8})
    ax[0, 1].set_title("$T+1 = 2$")

    ax[1, 0].axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
    ax[1, 0].plot(np.arange(0, 500, 0.1), list_alphas_waci[third_iteration], color=colors[0], label='WACI', lw=1)
    ax[1, 0].axhline(list_alphas_aci[third_iteration], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
    ax[1, 0].set_xlabel('Interval length')
    ax[1, 0].set_ylabel("$\\alpha_{100}$")
    ax[1, 0].grid()
    ax[1, 0].set_ylim(0.1, 0.3)
    ax[1, 0].set_xlim(0, 100)
    ax[1, 0].legend(frameon=True, prop={'size': 8})
    ax[1, 0].set_title("$T+1 = 100$")

    ax[1, 1].axhline(0.2, xmin= 0, xmax=500, lw=1, color='k', linestyle='--', label='Objective miscoverage', alpha=0.4)
    ax[1, 1].plot(np.arange(0, 500, 0.1), list_alphas_waci[fourth_iteration], color=colors[0], label='WACI', lw=1)
    ax[1, 1].axhline(list_alphas_aci[fourth_iteration], xmin= 0, xmax=500, lw=1, color=colors[1], label='ACI')
    ax[1, 1].set_xlabel('Interval length')
    ax[1, 1].set_ylabel("$\\alpha_{320}$")
    ax[1, 1].grid()
    ax[1, 1].set_ylim(0.1, 0.3)
    ax[1, 1].set_xlim(0, 100)
    ax[1, 1].legend(frameon=True, prop={'size': 8})
    ax[1, 1].set_title("$T+1 = 320$")

    plt.tight_layout()
    plt.savefig(d + f"\\Images\\alpha_progression_square.pdf", format='pdf', bbox_inches='tight')


def generate_average_interval_length_vs_mean_empirical_coverage_plot_sigma_impact():
    d = dirname(abspath(__file__))
    df = pd.read_csv(d + f"\\Results\\df_results_sigma_impact_alpha_0.2.csv", index_col=0)
    # df = df[df.index <= 100]
    df_results_aci = pd.read_csv(d + f"\\Results\\df_results_alpha_0.2.csv", index_col=0).loc['aci_hqr']

    obj_coverage = 100 * (1 - 0.2)

    transition_point = 30
    sigma_min = 0.1
    sigma_max = 200
    transition_relative = (transition_point - sigma_min) / (sigma_max - sigma_min)

    cdict = {
        'red':   [(0.0, 68/255, 68/255),
                  (transition_relative, 68/255, 68/255),
                  (1.0, 253/255, 253/255)],
        'green': [(0.0, 1/255, 1/255),
                  (transition_relative, 148/255, 148/255),
                  (1.0, 231/255, 231/255)],
        'blue':  [(0.0, 84/255, 84/255),
                  (transition_relative, 226/255, 226/255),
                  (1.0, 37/255, 37/255)]
    }
    cmap = LinearSegmentedColormap('custom_cmap', segmentdata=cdict, N=256)

    norm = plt.Normalize(vmin=df.index.min(), vmax=df.index.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    for i, sigma in enumerate(df.index):
        color = cmap(norm(sigma))
        ax0.plot(df.loc[sigma, 'empirical_coverage'], df.loc[sigma, 'average_length'], marker='o', color=color, zorder=1)
        ax1.plot(df.loc[sigma, 'mean_deviation_coverage_by_interval_length'], df.loc[sigma, 'ILS_10_coverage'], marker='o', color=color, zorder=1)
        ax2.plot(sigma, df.loc[sigma, 'pearson_correlation'], marker='o', color=color, zorder=1)
    ax0.scatter(df_results_aci['empirical_coverage'], df_results_aci['average_length'], marker='*', color='k', label='ACI-HQR', zorder=2)
    ax1.scatter(df_results_aci['mean_deviation_coverage_by_interval_length'], df_results_aci['ILS_10_coverage'], marker='*', color='k', label='ACI-HQR', zorder=2)
    ax2.scatter(200.1, df_results_aci['pearson_correlation'], marker='*', color='k', label='ACI-HQR', zorder=2)

    ax0.axvline(obj_coverage, color='k', linestyle='--', lw=1.2, label='Objective coverage')
    ax0.set_xlabel("Mean empirical coverage")
    ax0.set_ylabel("Average interval length")
    ax0.legend(frameon=True)
    ax0.grid()

    ax1.set_xlabel("MCD")
    ax1.set_ylabel("ILS 10% coverage")
    ax1.grid()
    ax1.legend(frameon=True)

    ax2.set_xlabel("$\sigma$")
    ax2.set_ylabel("Pearson correlation")
    ax2.grid()
    ax2.legend(frameon=True)

    # Create a single color bar for the entire figure
    norm = plt.Normalize(vmin=df.index.min(), vmax=df.index.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('$\sigma$')

    plt.tight_layout()
    plt.savefig(d + "\\Images\\sigma_impact.pdf", format='pdf', bbox_inches='tight')
    # plt.show()

def generate_significance_results():
    d = dirname(abspath(__file__))
    axis_names = ['HQR', 'HQR-W', 'QRA', 'ACI HQR', 'ACI HQR-W', 'ACI QRA', 'WACI HQR', 'WACI HQR-W', 'WACI QRA', 'DCP HQR', 'DCP HQR-W', 'DCP QRA']

    metric_specs = [
        ('empirical_coverage', 'Empirical coverage', 0, 0),
        ('average_length', 'Average length', 0, 1),
        ('winkler_score', 'Winkler score', 0, 2),
        ('interval_length_std', 'Interval length std', 1, 0),
        ('spearman_correlation', 'Spearman correlation', 1, 1),
        ('mean_deviation_coverage_by_interval_length', 'MCD 5', 2, 0),
        ('pearson_correlation', 'Pearson correlation', 2, 1),
        ('ILS_10_coverage', 'ILS 10', 2, 2),
    ]
    
    group_colors = {
        'HQR': colors[0],
        'HQR-W': colors[1],
        'QRA': colors[2]
    }

    model_group = []
    for name in axis_names:
        if 'HQR-W' in name:
            model_group.append('HQR-W')
        elif 'QRA' in name:
            model_group.append('QRA')
        else:
            model_group.append('HQR')

    grouped_models = {
        'Base': ['HQR', 'HQR-W', 'QRA'],
        'ACI': ['ACI HQR', 'ACI HQR-W', 'ACI QRA'],
        'WACI': ['WACI HQR', 'WACI HQR-W', 'WACI QRA'],
        'DCP': ['DCP HQR', 'DCP HQR-W', 'DCP QRA']
    }

    x_positions = {}
    group_centers = []
    group_boundaries = []
    current_x = 0
    intra_group_step = 1
    inter_group_gap = 1

    for group_name, models in grouped_models.items():
        start_x = current_x
        for model in models:
            x_positions[model] = current_x
            current_x += intra_group_step
        end_x = current_x - intra_group_step
        group_centers.append((start_x + end_x) / 2)
        group_boundaries.append(end_x + (inter_group_gap / 2))
        current_x += inter_group_gap
    
    for alpha in [0.1, 0.2]:
        # df_results = pd.read_csv(d + f"\\Results\\df_results_alpha_{alpha}.csv", index_col=0)
        fig, ax = plt.subplots(3, 3, figsize=(12, 9))

        for metric_name, metric_title, row, col in metric_specs:
            file = f"df_{metric_name}_alpha_{alpha}.csv"
            df_metrics = pd.read_csv(d + f"\\Results\\{file}", index_col=0)
            # df_metrics_mean = df_results[metric_name]
            df_metrics_mean = df_metrics.mean(axis=0)
            df_metrics_mean.index = axis_names
            df_metrics_std = df_metrics.std(axis=0)
            df_metrics_std.index = axis_names
            n = df_metrics.shape[0]
            df_intervals = 1.96 * df_metrics_std / np.sqrt(n)

            for j, model in enumerate(df_metrics_mean.index):
                color = group_colors[model_group[j]]
                ax[row, col].errorbar(x_positions[model], df_metrics_mean.loc[model], fmt='o', yerr=df_intervals.loc[model], color=color, zorder=2, markersize=5, capsize=5, elinewidth=2)

            for models in grouped_models.values():
                xs = [x_positions[m] for m in models if m in df_metrics_mean.index]
                ys = [df_metrics_mean[m] for m in models if m in df_metrics_mean.index]
                if len(xs) > 1:
                    ax[row, col].plot(xs, ys, linestyle='--', color='grey', alpha=0.7, zorder=1)

            for boundary in group_boundaries[:-1]:
                ax[row, col].axvline(boundary, color='grey', linestyle='-', lw=1, alpha=0.35, zorder=1)

            ax[row, col].set_title(metric_title)
            ax[row, col].set_xticks(group_centers)
            ax[row, col].set_xticklabels(list(grouped_models.keys()), fontsize=8)
            ax[row, col].tick_params(axis='y', labelsize=8)
            ax[row, col].grid(zorder=0)

        ax[1, 2].set_xticks([])
        ax[1, 2].set_yticks([])
        for spine in ax[1, 2].spines.values():
            spine.set_visible(False)

        legend_handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[0], markeredgecolor=colors[0], markersize=7, label='HQR'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[1], markeredgecolor=colors[1], markersize=7, label='HQR-W'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[2], markeredgecolor=colors[2], markersize=7, label='QRA'),
        ]
        fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))

        # plt.suptitle(f"$\\alpha = {alpha}$", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(d + f"\\Images\\significance_results_alpha_{alpha}.pdf", format='pdf', bbox_inches='tight')

# generate_day_ahead_market_figure()
# generate_day_ahead_market_predictions()
# generate_progression_alphas(hour=13)
# generate_coefs_variation()
# generate_coefs_models_qra()
# generate_progression_alphas_2(hour=13)
# generate_marginal_coverage_vs_conditional_coverage()
# generate_average_interval_length_vs_mean_empirical_coverage_plot()
# generate_progression_alphas_square()
# generate_average_interval_length_vs_mean_empirical_coverage_plot_sigma_impact()
generate_significance_results()


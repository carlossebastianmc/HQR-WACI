import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
import pickle

def HeterocedasticQuantileRegression(first_idx, df, alpha, by_hour, save_coefs=False, varying_weights=False, num_preds=1):
    """
    Perform Heterocedastic Quantile Regression.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe containing the data.
        alpha (float): The objective miscoverage rate.
        by_hour (bool): Flag indicating whether to perform the calculation by hour.

    Returns:
        tuple: A tuple containing two lists: preds_inf and preds_sup.
            - preds_inf (list): The lower quantile predictions.
            - preds_sup (list): The upper quantile predictions.
    """
    preds_inf = []
    preds_sup = []
    alpha_inf = alpha/2
    alpha_sup = 1 - alpha_inf
    if save_coefs:
        dict_coefs = {'sup': [], 'inf': []}
    
    if varying_weights:
        explanatory_features = [f'pred{i}' for i in range(1, num_preds+1)] + ['std_pred']
    else:
        explanatory_features = ['mean_pred', 'std_pred']

    for obs in tqdm(range(first_idx, len(df)), desc=f'Heterocedastic Quantile Regression. By hours = {by_hour}. Varying weights = {varying_weights}'):
        if by_hour:
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])][['real'] + explanatory_features]
            if len(df_cal) > 6*30:
                df_cal = df_cal.tail(6*30)
        else:   
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date']))][['real'] + explanatory_features]
            if len(df_cal) > 6*30*24:
                df_cal = df_cal.tail(6*30*24)
        
        df_test = pd.DataFrame(df.loc[obs][['real'] + explanatory_features]).T
        df_test['const'] = 1.0

        X_train = sm.add_constant(df_cal[explanatory_features])
        y_train = df_cal['real']

        X_test = df_test[['const'] + explanatory_features]

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_inf)
        preds_inf.append(res.predict(X_test).values[0])
        if save_coefs:
            dict_coefs['inf'].append(res.params['std_pred'])

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_sup)
        preds_sup.append(res.predict(X_test).values[0])
        if save_coefs:
            dict_coefs['sup'].append(res.params['std_pred'])
        
    if save_coefs:
        with open(f'dict_coefs_hqr_alpha_{alpha}.pkl', 'wb') as f:
            pickle.dump(dict_coefs, f)

    return preds_inf, preds_sup

def QuantileRegressionAveraging(first_idx, df, alpha, by_hour, m):
    """
    Perform Quantile Regression Averaging.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe containing the data.
        alpha (float): The objective miscoverage rate.
        by_hour (bool): Flag indicating whether to perform the calculation by hour.
        m (int): The number of predictors.

    Returns:
        tuple: A tuple containing two lists: preds_inf and preds_sup.
            - preds_inf (list): The lower quantile predictions.
            - preds_sup (list): The upper quantile predictions.
    """
    preds_inf = []
    preds_sup = []
    alpha_inf = alpha/2
    alpha_sup = 1 - alpha_inf
    for obs in tqdm(range(first_idx, len(df)), desc=f'Quantile Regression Averaging. By hours = {by_hour}'):
        if by_hour:
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])][['real'] + [f'pred{i}' for i in range(1, m+1)]]
            if len(df_cal) > 6*30:
                df_cal = df_cal.tail(6*30)
        else:   
            df_cal = df[(df.index < obs) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date']))][['real'] + [f'pred{i}' for i in range(1, m+1)]]
            if len(df_cal) > 6*30*24:
                df_cal = df_cal.tail(6*30*24)
        
        df_test = pd.DataFrame(df.loc[obs][[f'pred{i}' for i in range(1, m+1)]]).T
        df_test['const'] = 1.0

        X_train = sm.add_constant(df_cal[[f'pred{i}' for i in range(1, m+1)]])
        y_train = df_cal['real']

        X_test = df_test[['const'] + [f'pred{i}' for i in range(1, m+1)]]

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_inf)
        preds_inf.append(res.predict(X_test).values[0])

        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=alpha_sup)
        preds_sup.append(res.predict(X_test).values[0])

    return preds_inf, preds_sup

def ConformalizedQuantileRegression(first_idx, df, alpha, method, by_hour):
    """
    Perform Conformalized Quantile Regression.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_cqr_inf (list): The conformalized lower quantile predictions.
            - preds_cqr_sup (list): The conformalized upper quantile predictions.

    """
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    preds_cqr_inf, preds_cqr_sup = [], []

    for obs in tqdm(range(first_idx, len(df)), desc=f'Conformalized Quantile Regression. Method = {method}. By hours = {by_hour}'):
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == df.loc[obs, 'hour'])].tail(500)
        
        n=len(df_cal)
        
        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
        qhat = np.quantile(cal_scores.to_numpy(), quantile)
        
        preds_cqr_inf.append(df.loc[obs, f'preds_inf_{add_string}'] - qhat)
        preds_cqr_sup.append(df.loc[obs, f'preds_sup_{add_string}'] + qhat)
    return preds_cqr_inf, preds_cqr_sup

def AdaptiveConformalInference(first_idx, df, alpha, gamma, method, by_hour, save_alphas=False):
    """
    Perform adaptive conformal inference.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_aci_inf (list): The adaptive conformalized lower quantile predictions.
            - preds_aci_sup (list): The adaptive conformalized upper quantile predictions.
    """
    preds_aci_inf, preds_aci_sup = [], []
    alpha_obj = alpha
    # list_alpha_hora_13 = []
    
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    alpha_by_hour = {hour: alpha_obj for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [alpha_obj] for hour in range(1, 25)}

    for obs in tqdm(range(first_idx, len(df)), desc=f'Adaptive Conformal Inference. Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        alpha = alpha_by_hour[hour]

        # if hour == 13:
        #     list_alpha_hora_13.append(alpha)

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        if alpha < 1 and alpha > 0:
            quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
            qhat = np.quantile(cal_scores.to_numpy(), quantile)
            pred_aci_inf = df_test[f'preds_inf_{add_string}'] - qhat
            pred_aci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        elif alpha >= 1:
            pred_aci_inf = df_test[f'preds_inf_{add_string}']
            pred_aci_sup = df_test[f'preds_sup_{add_string}']
        else:
            pred_aci_inf = -np.inf
            pred_aci_sup = np.inf
        
        preds_aci_inf.append(pred_aci_inf)
        preds_aci_sup.append(pred_aci_sup)

        err = 1 - float(df_test.real >= pred_aci_inf and df_test.real <= pred_aci_sup)
        alpha = alpha + gamma*(alpha_obj - err)
        
        alpha_by_hour[hour] = alpha
        if save_alphas:
            dict_list_alphas[hour].append(alpha)

    if save_alphas:
        with open(f'dict_list_alphas_aci.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_aci_inf, preds_aci_sup

def WidthAdaptiveConformalInference(first_idx, df, alpha, gamma, sigma, method, by_hour, save_alphas=False):
    """
    Perform width-adaptive conformal inference.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        sigma (float): The impact of the gaussian kernel.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_waci_inf (list): The width-adaptive conformalized lower quantile predictions.
            - preds_waci_sup (list): The width-adaptive conformalized upper quantile predictions.
    """
    preds_waci_inf, preds_waci_sup = [], []
    alpha_obj = alpha
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    lengths_by_hour = {hour: np.arange(0, 500, 0.1) for hour in range(1, 25)}
    alphas_by_hour = {hour: np.array([alpha_obj] * len(lengths_by_hour[hour])) for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [np.array([alpha_obj] * len(lengths_by_hour[hour]))] for hour in range(1, 25)}
    
    for obs in tqdm(range(first_idx, len(df)), desc=f'Width-Adaptive Conformal Inference. Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        lengths = lengths_by_hour[hour]
        alphas = alphas_by_hour[hour]

        length_obs = df_test[f'length_{add_string}']
        closest_iw = min(lengths, key=lambda x:abs(x-length_obs))
        idx_closest_iw = list(lengths).index(closest_iw)
        alpha = alphas[idx_closest_iw]

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)
        if alpha < 1 and alpha > 0:
            quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
            qhat = np.quantile(cal_scores.to_numpy(), quantile)
            pred_waci_inf = df_test[f'preds_inf_{add_string}'] - qhat
            pred_waci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        elif alpha >= 1:
            pred_waci_inf = df_test[f'preds_inf_{add_string}']
            pred_waci_sup = df_test[f'preds_sup_{add_string}']
        else:
            pred_waci_inf = -np.inf
            pred_waci_sup = np.inf

        preds_waci_inf.append(pred_waci_inf)
        preds_waci_sup.append(pred_waci_sup)

        err = 1 - float(df_test.real >= pred_waci_inf and df_test.real <= pred_waci_sup)
        
        distances = np.abs(np.array(lengths) - length_obs)
        unnormalized_weights = np.exp(-distances**2/(2*sigma**2))
        max_weight = np.max(unnormalized_weights)
        weights = unnormalized_weights / max_weight
        
        alphas = alphas + gamma * weights * (alpha_obj - err)

        alphas_by_hour[hour] = alphas

        if save_alphas:
            dict_list_alphas[hour].append(alphas)

    if save_alphas:
        with open(f'dict_list_alphas_waci.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_waci_inf, preds_waci_sup

def WidthAdaptiveConformalInference_2(first_idx, df, alpha, gamma, rate, method, by_hour, save_alphas=False):
    """
    Perform width-adaptive conformal inference.

    Args:
        first_idx (int): The index of the first observation.
        df (pandas.DataFrame): The input dataframe.
        alpha (float): The objective miscoverage rate.
        gamma (float): The adaptation rate.
        sigma (float): The impact of the gaussian kernel.
        method (str): The method used for quantile regression.
        by_hour (bool): Whether the previous quantile regression was done by hour.

    Returns:
        tuple: A tuple containing two lists: preds_cqr_inf and preds_cqr_sup.
            - preds_waci_inf (list): The width-adaptive conformalized lower quantile predictions.
            - preds_waci_sup (list): The width-adaptive conformalized upper quantile predictions.
    """
    preds_waci_inf, preds_waci_sup = [], []
    alpha_obj = alpha
    if by_hour:
        add_string = method + '_by_hour'
    else:
        add_string = method
    
    lengths_by_hour = {hour: np.arange(0, 500, 0.1) for hour in range(1, 25)}
    alphas_by_hour = {hour: np.array([alpha_obj] * len(lengths_by_hour[hour])) for hour in range(1, 25)}
    if save_alphas:
        dict_list_alphas = {hour: [np.array([alpha_obj] * len(lengths_by_hour[hour]))] for hour in range(1, 25)}
    
    for obs in tqdm(range(first_idx, len(df)), desc=f'Width-Adaptive Conformal Inference 2 . Method = {method}. By hours = {by_hour}'):
        hour = df.loc[obs, 'hour']
        df_cal = df[(df.index < obs) & (~df[f'length_{add_string}'].isna()) & (pd.to_datetime(df.date) < pd.to_datetime(df.loc[obs, 'date'])) & (df.hour == hour)].tail(500)
        df_test = df.loc[obs]

        lengths = lengths_by_hour[hour]
        alphas = alphas_by_hour[hour]

        length_obs = df_test[f'length_{add_string}']
        closest_iw = min(lengths, key=lambda x:abs(x-length_obs))
        idx_closest_iw = list(lengths).index(closest_iw)
        alpha = alphas[idx_closest_iw]

        n = len(df_cal)

        cal_scores = np.maximum(df_cal.real - df_cal[f'preds_sup_{add_string}'], df_cal[f'preds_inf_{add_string}'] - df_cal.real)

        if alpha < 1 and alpha > 0:
            quantile = np.min([1, np.max([0, (np.ceil(n+1)*(1-alpha))/n])])
            qhat = np.quantile(cal_scores.to_numpy(), quantile)
            pred_waci_inf = df_test[f'preds_inf_{add_string}'] - qhat
            pred_waci_sup = df_test[f'preds_sup_{add_string}'] + qhat
        elif alpha >= 1:
            pred_waci_inf = df_test[f'preds_inf_{add_string}']
            pred_waci_sup = df_test[f'preds_sup_{add_string}']
        else:
            pred_waci_inf = -np.inf
            pred_waci_sup = np.inf

        preds_waci_inf.append(pred_waci_inf)
        preds_waci_sup.append(pred_waci_sup)

        err = 1 - float(df_test.real >= pred_waci_inf and df_test.real <= pred_waci_sup)
        
        distances = np.abs(np.array(lengths) - length_obs)
        min_distance_position = np.argmin(distances)
        position_distances = np.abs(np.arange(len(distances)) - min_distance_position)
        weights = 1 / (rate ** position_distances)
        weights[weights < 0] = 0
        weights[np.isinf(weights)] = 0
                
        alphas = alphas + gamma * weights * (alpha_obj - err)

        alphas_by_hour[hour] = alphas

        if save_alphas:
            dict_list_alphas[hour].append(alphas)

    if save_alphas:
        with open(f'dict_list_alphas_waci_2.pkl', 'wb') as f:
            pickle.dump(dict_list_alphas, f)

    return preds_waci_inf, preds_waci_sup

def dcp_opt(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha_sig):
    """
    Perform Distributional Conformal Prediction with Optimization (DCP-OPT).

    This function computes prediction intervals for a test dataset using quantile regression
    and an optimization-based approach to minimize the length of the intervals while maintaining
    the desired coverage level.

    Args:
        Y0 (numpy.ndarray): Response variable for the training set used to fit quantile regression models.
        X0 (numpy.ndarray): Predictor variables for the training set used to fit quantile regression models.
        Y1 (numpy.ndarray): Response variable for the holdout set used to compute conformal scores.
        X1 (numpy.ndarray): Predictor variables for the holdout set used to compute conformal scores.
        Y_test (numpy.ndarray): Response variable for the test set (used for evaluation).
        X_test (numpy.ndarray): Predictor variables for the test set (used to compute prediction intervals).
        taus (numpy.ndarray): Array of quantile levels used for quantile regression.
        alpha_sig (float): Significance level (e.g., 0.1 for a 90% prediction interval).

    Returns:
        tuple:
            - UB (numpy.ndarray): Upper bounds of the prediction intervals for the test set.
            - LB (numpy.ndarray): Lower bounds of the prediction intervals for the test set.
            - cov_opt (numpy.ndarray): Boolean array indicating whether each test point is covered by the prediction interval.
            - leng_opt (numpy.ndarray): Lengths of the prediction intervals for the test set.

    Steps:
        1. Fit quantile regression models for each quantile level in `taus` using the training data `(Y0, X0)`.
        2. Compute the predicted quantiles for the combined holdout and test sets `(X1, X_test)`.
        3. Estimate the conformal scores (`cs_opt`) based on the difference between the predicted quantiles and the observed values.
        4. Determine a threshold for the conformal scores to ensure the desired coverage level.
        5. Compute the prediction intervals (lower and upper bounds) for the test set based on the threshold.

    Notes:
        - The function uses quantile regression to model the conditional distribution of the response variable.
        - The optimization step minimizes the length of the prediction intervals while maintaining the desired coverage.
    """
    XXX = np.vstack((X1, X_test))
    YYY = np.concatenate((Y1, Y_test))

    beta_qr = np.empty((X0.shape[1] + 1, len(taus)))
    for t, tau in enumerate(taus):
        model = sm.QuantReg(Y0, sm.add_constant(X0))
        res = model.fit(q=tau)
        beta_qr[:, t] = res.params

    tQ_yx = sm.add_constant(XXX) @ beta_qr
    Q_yx = np.sort(tQ_yx, axis=1)
    u_hat = np.mean(Q_yx <= YYY[:, None], axis=1)

    bhat = np.empty(len(YYY))
    b_grid = taus[taus <= alpha_sig]
    for t in range(len(YYY)):
        leng = np.empty(len(b_grid))
        for b, b_val in enumerate(b_grid):
            Q_yx_u = np.interp(b_val + 1 - alpha_sig, taus, Q_yx[t, :])
            leng[b] = Q_yx_u - Q_yx[t, b]
        bhat[t] = b_grid[np.argmin(leng)]

    ind_test = np.arange(len(Y1), len(YYY))
    cs_opt = np.abs(u_hat - bhat - (1 - alpha_sig) / 2)

    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs_opt[:len(Y1)])[k]

    cov_opt = cs_opt[ind_test] <= threshold

    leng_opt = []
    UB = []
    LB = []
    
    for t in ind_test:
        ci_grid = np.abs(taus - bhat[t] - (1 - alpha_sig) / 2)
        ci = Q_yx[t, ci_grid <= threshold]
        ub = np.max(ci)
        lb = np.min(ci)
        
        leng_opt.append(ub - lb)
        UB.append(ub)
        LB.append(lb)

    leng_opt = np.array(leng_opt)
    leng_opt[leng_opt == -np.inf] = np.nan

    return UB, LB, cov_opt, leng_opt






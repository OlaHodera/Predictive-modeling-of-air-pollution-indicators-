import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import boxcox, inv_boxcox
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools


##################### ORDERS SEARCH ###############################

def sarima_grid_search(df_train, target, exog, p_range, q_range, P_range, Q_range, m):
    y_train = df_train[target]
    
    best_aic = np.inf
    best_order = None
    best_sorder = None
    best_fit = None
    
    for p, q in itertools.product(p_range, q_range):
        for P, Q in itertools.product(P_range, Q_range):
            try:
                model = SARIMAX(
                    y_train,
                    exog=exog,
                    trend='c',
                    order=(p, 0, q),
                    seasonal_order=(P, 0, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fit = model.fit(disp=False)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, 0, q)
                    best_sorder = (P, 0, Q, m)
                    best_fit = fit 
            except Exception as e:
                continue

    print(f"{target}: Best SARIMA order={best_order}, sorder={best_sorder}, AIC={best_aic:.2f}")
    
    print(best_fit.summary())
    print('\n')
    
    return best_order, best_sorder, best_aic, best_fit


def SARIMAX_grid_search(df_train, target, exog, p_values, d_values, q_values, verbose=False):
    y_train = df_train[target]
    
    best_aic = np.inf
    best_order = None
    best_fit = None
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = SARIMAX(
                y_train,
                exog=exog,
                order=(p, d, q),
                seasonal_order=(0, 0, 0, 0),
                trend='n',
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit = model.fit(disp=False)
            if verbose:
                print(f"  ({p},{d},{q}): AIC={fit.aic:.2f}")
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
                best_fit = fit
        except Exception:
            continue
    
    print(f"{target}: Best SARIMA order={best_order}, AIC={best_aic:.2f}")
    
    print(best_fit.summary())
    print('\n')
    
    return best_fit, best_order, best_aic


##################### PLOT FULL PREDICTIONS ###############################

def plot_full_predictions(fit, df_train, df_test, target, exog_test, model,
                          lambda_bc=False, alpha_ci=0.05, burnin=5):
    y_train_obs = df_train[target]                
    y_test_obs = df_test[target]
    
    y_train_model = df_train[f'{target}_bc']         
    y_test_model = df_test[f'{target}_bc']
    
    # Train: in-sample 1-step 
    pred_train = fit.get_prediction(start=0, end=len(y_train_model) - 1)
    pred_train_mean_bc = pred_train.predicted_mean
    pred_train_ci_bc = pred_train.conf_int(alpha=alpha_ci)
    
    # Test: walk-forward 1-step 
    if exog_test is not None:
        fit_on_test = fit.apply(endog=y_test_model, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=y_test_model)
    
    pred_test = fit_on_test.get_prediction(start=0, end=len(y_test_model) - 1)
    pred_test_mean_bc = pred_test.predicted_mean
    pred_test_ci_bc = pred_test.conf_int(alpha=alpha_ci)
    
    pred_train_mean = pd.Series(
        inv_boxcox(pred_train_mean_bc.values, lambda_bc),
        index=pred_train_mean_bc.index
    )
    pred_train_ci = pd.DataFrame({
        'lower': inv_boxcox(pred_train_ci_bc.iloc[:, 0].values, lambda_bc),
        'upper': inv_boxcox(pred_train_ci_bc.iloc[:, 1].values, lambda_bc)
    }, index=pred_train_ci_bc.index)
    
    pred_test_mean = pd.Series(
        inv_boxcox(pred_test_mean_bc.values, lambda_bc),
        index=pred_test_mean_bc.index
    )
    pred_test_ci = pd.DataFrame({
        'lower': inv_boxcox(pred_test_ci_bc.iloc[:, 0].values, lambda_bc),
        'upper': inv_boxcox(pred_test_ci_bc.iloc[:, 1].values, lambda_bc)
    }, index=pred_test_ci_bc.index)
    
    dates_train = y_train_obs.index
    dates_test  = y_test_obs.index
    dates_full  = dates_train.append(dates_test)
    full_obs = np.concatenate([y_train_obs.values, y_test_obs.values])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.xaxis.grid()
    
    ax.plot(dates_full, full_obs, 'k.', alpha=0.6, label='Real data', zorder=2)
    
    ax.plot(dates_train[burnin:], pred_train_mean[burnin:].values, color='C0', 
            label='In-sample 1-step (train)', zorder=3)
    ax.fill_between(dates_train[burnin:],
                    pred_train_ci['lower'].values[burnin:],
                    pred_train_ci['upper'].values[burnin:],
                    color='C0', alpha=0.4, zorder=1)
    
    ax.plot(dates_test, pred_test_mean.values, c='C2',
            label='Walk-forward 1-step (test)', zorder=4)
    ax.fill_between(dates_test,
                    pred_test_ci['lower'].values, 
                    pred_test_ci['upper'].values,
                    color='C2', alpha=0.4, zorder=1)
    
    ax.axvline(dates_test[0], color='C3', linewidth=2, zorder=4)
    
    y_min = np.nanmin(full_obs)
    y_max = np.nanmax(full_obs)
    margin = (y_max - y_min) * 0.05
    ax.set_ylim(max(0, y_min - margin), y_max + margin)
    
    ax.set_title(f'{model} - {target}')
    ax.legend(loc='upper left',  prop = { "size": 13 })
    plt.tight_layout()
    return fig


def plot_full_predictions_stationary(fit, df_train, df_test, target, exog_test, model,
                          lambda_bc=False, alpha_ci=0.05, burnin=5):
    y_train_obs = df_train[target]
    y_test_obs = df_test[target]

    y_train_model = df_train[f'{target}_bc']
    y_test_model = df_test[f'{target}_bc']

    y_train_sin = df_train[f'{target}_sinus']
    y_test_sin = df_test[f'{target}_sinus']

    # Train: in-sample 1-step
    pred_train = fit.get_prediction(start=0, end=len(y_train_model) - 1)
    pred_train_mean_model = pred_train.predicted_mean + y_train_sin
    pred_train_ci_model = pred_train.conf_int(alpha=alpha_ci)
    pred_train_ci_model = pred_train_ci_model.add(y_train_sin, axis=0)
    
    # Test: walk-forward 1-step
    fit_on_test = fit.apply(endog=(y_test_model - y_test_sin), exog=exog_test)
    pred_test = fit_on_test.get_prediction(start=0, end=len(y_test_obs) - 1)
    pred_test_mean_model = pred_test.predicted_mean + y_test_sin
    pred_test_ci_model = pred_test.conf_int(alpha=alpha_ci)
    pred_test_ci_model = pred_test_ci_model.add(y_test_sin, axis=0)

    pred_train_mean = pd.Series(
            inv_boxcox(pred_train_mean_model.values, lambda_bc),
            index=pred_train_mean_model.index
        )
    pred_train_ci = pd.DataFrame({
            'lower': inv_boxcox(pred_train_ci_model.iloc[:, 0].values, lambda_bc),
            'upper': inv_boxcox(pred_train_ci_model.iloc[:, 1].values, lambda_bc)
        }, index=pred_train_ci_model.index)
        
    pred_test_mean = pd.Series(
            inv_boxcox(pred_test_mean_model.values, lambda_bc),
            index=pred_test_mean_model.index
        )
    pred_test_ci = pd.DataFrame({
            'lower': inv_boxcox(pred_test_ci_model.iloc[:, 0].values, lambda_bc),
            'upper': inv_boxcox(pred_test_ci_model.iloc[:, 1].values, lambda_bc)
        }, index=pred_test_ci_model.index)

    
    dates_train = y_train_obs.index
    dates_test  = y_test_obs.index
    dates_full  = dates_train.append(dates_test)
    full_obs = np.concatenate([y_train_obs.values, y_test_obs.values])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.xaxis.grid()
    
    ax.plot(dates_full, full_obs, 'k.', alpha=0.6, label='Real data', zorder=2)
    
    ax.plot(dates_train[burnin:], pred_train_mean[burnin:].values, color='C0', 
            label='In-sample 1-step (train)', zorder=3)
    ax.fill_between(dates_train[burnin:],
                    pred_train_ci['lower'].values[burnin:],
                    pred_train_ci['upper'].values[burnin:],
                    color='C0', alpha=0.4, zorder=1)
    
    ax.plot(dates_test, pred_test_mean.values, c='C2',
            label='Walk-forward 1-step (test)', zorder=4)
    ax.fill_between(dates_test,
                    pred_test_ci['lower'].values, 
                    pred_test_ci['upper'].values,
                    color='C2', alpha=0.4, zorder=1)
    
    ax.axvline(dates_test[0], color='C3', linewidth=2, zorder=4)
    
    y_min = np.nanmin(full_obs)
    y_max = np.nanmax(full_obs)
    margin = (y_max - y_min) * 0.05
    ax.set_ylim(max(0, y_min - margin), y_max + margin)
    
    ax.set_title(f'{model} - {target}')
    ax.legend(loc='upper left',  prop = { "size": 13 })
    plt.tight_layout()
    # plt.show()
    return fig


##################### ERROR METRICS ON TEST SET ###############################

def evaluate_on_test(fit, df_test, target, exog_test, lambda_bc):
    y_test_obs = df_test[target]                     
    y_test_model = df_test[f'{target}_bc']          
    
    # 1-step walk-forward
    if exog_test is not None:
        pred_model = fit.apply(endog=y_test_model, exog=exog_test)
    else:
        pred_model = fit.apply(endog=y_test_model)
    
    pred_model = pred_model.fittedvalues

    pred = pd.Series(
            inv_boxcox(pred_model.values, lambda_bc),
            index=pred_model.index
        )
    
    yt = y_test_obs
    yp = pred
    
    rmse = np.sqrt(((yt - yp)**2).mean())
    mae = (yt - yp).abs().mean()
    r2 = 1 - (((yt - yp)**2).sum() / ((yt - yt.mean())**2).sum())
    mape = ((yt - yp) / yt.replace(0, np.nan)).abs().mean() * 100
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}



def evaluate_on_test_stationary(fit, df_test, target, exog_test, lambda_bc):
    y_sinus_test = df_test[f'{target}_sinus']
    
    y_test_obs = df_test[target]
    y_test_model = df_test[f'{target}_bc']
    
    fit_on_test = fit.apply(endog=(y_test_model - y_sinus_test), exog=exog_test)
    pred_model = fit_on_test.fittedvalues + y_sinus_test

    pred = pd.Series(
            inv_boxcox(pred_model.values, lambda_bc),
            index=pred_model.index
        )
    
    yt = y_test_obs
    yp = pred
    
    rmse = np.sqrt(((yt - yp)**2).mean())
    mae = (yt - yp).abs().mean()
    r2 = 1 - ((yt - yp)**2).sum() / ((yt - yt.mean())**2).sum()
    mape = ((yt - yp) / yt.replace(0, np.nan)).abs().mean() * 100
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

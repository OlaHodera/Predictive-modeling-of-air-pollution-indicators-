import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import boxcox, inv_boxcox
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

EAQI_THRESHOLDS = {
    "PM25":  [5, 15, round((50 - 0.476)/1.849), round((90 - 0.476)/1.849), round((140 - 0.476)/1.849)],
    "PM10":  [15, 45, round((120 + 1.137)/1.956), round((195 + 1.137)/1.956), round((270 + 1.137)/1.956)],
    "O3":    [60, 100, round((120 - 6.908)/0.998), round((160 - 6.908)/0.998), round((180 - 6.908)/0.998)],
    "NO2":   [10, 25, 60, 100, 150],
    "SO2":   [20, 40, 125, 190, 275],
}

EAQI_LABELS = ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"]
EAQI_COLORS = ["#50f0e6", "#50ccaa", "#f0e641", "#ff5050", "#960032", "#7d2181"]

def assign_subindex(value, thresholds):
    if pd.isna(value):
        return np.nan
    for i, t in enumerate(thresholds):
        if value <= t:
            return i + 1
    return 6


def compute_eaqi(df_concentrations):
    available = [p for p in df_concentrations.columns if p in EAQI_THRESHOLDS]
    result = pd.DataFrame(index=df_concentrations.index)
    for p in available:
        result[f"sub_{p}"] = df_concentrations[p].apply(
            lambda x: assign_subindex(x, EAQI_THRESHOLDS[p])
        )
    sub_cols = [c for c in result.columns if c.startswith("sub_")]
    result["EAQI"] = result[sub_cols].max(axis=1)
    return result


############ uniVARIATE #########################

def get_predictions_ugm3_arima(fitted_models, df_train, df_test, pollutants, lambdas,
                                exog_test=None):
    pred_train = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        fit = fitted_models[p]
        lam = lambdas[p]
        bc_col = f'{p}_bc'
        
        # in-sample train
        pred_train_bc = fit.get_prediction(
            start=0, end=len(df_train) - 1
        ).predicted_mean
        
        # walk-forward test
        endog_test = df_test[bc_col]
        if exog_test is not None:
            fit_on_test = fit.apply(endog=endog_test, exog=exog_test)
        else:
            fit_on_test = fit.apply(endog=endog_test)
        pred_test_bc = fit_on_test.get_prediction(
            start=0, end=len(df_test) - 1
        ).predicted_mean
        
        # inverse Box-Cox
        pred_train[p] = inv_boxcox(pred_train_bc.values, lam)
        pred_test[p]  = inv_boxcox(pred_test_bc.values, lam)
    
    return pred_train, pred_test


def get_predictions_ugm3_sinus_arima(fitted_models, df_train, df_test, pollutants, lambdas,
                                       exog_test=None):
    pred_train = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        fit = fitted_models[p]
        lam = lambdas[p]
        resid_col = f'{p}_non_sinus'
        sinus_train = df_train[f'{p}_sinus'].values
        sinus_test = df_test[f'{p}_sinus'].values
        
        # in-sample train
        pred_train_resid = fit.get_prediction(
            start=0, end=len(df_train) - 1
        ).predicted_mean
        
        # walk-forward test
        endog_test = df_test[resid_col]
        if exog_test is not None:
            fit_on_test = fit.apply(endog=endog_test, exog=exog_test)
        else:
            fit_on_test = fit.apply(endog=endog_test)
        pred_test_resid = fit_on_test.get_prediction(
            start=0, end=len(df_test) - 1
        ).predicted_mean
        
        # dodaj sinus + inverse Box-Cox
        pred_train[p] = inv_boxcox(pred_train_resid.values + sinus_train, lam)
        pred_test[p]  = inv_boxcox(pred_test_resid.values + sinus_test, lam)
    
    return pred_train, pred_test


def evaluate_index_arima(fitted_models, df_train, df_test, pollutants, lambdas,
                          sinus=False, THRESHOLDS=EAQI_THRESHOLDS, exog_test=None, 
                          model_name="SARIMAX"):
    if sinus:
        pred_train, pred_test = get_predictions_ugm3_sinus_arima(
            fitted_models, df_train, df_test, pollutants, lambdas, exog_test
        )
    else:
        pred_train, pred_test = get_predictions_ugm3_arima(
            fitted_models, df_train, df_test, pollutants, lambdas, exog_test
        )
    
    eaqi_poll = [p for p in pollutants if p in THRESHOLDS]
    
    eaqi_obs_train  = compute_eaqi(df_train[eaqi_poll])["EAQI"]
    eaqi_pred_train = compute_eaqi(pred_train[eaqi_poll])["EAQI"]
    eaqi_obs_test   = compute_eaqi(df_test[eaqi_poll])["EAQI"]
    eaqi_pred_test  = compute_eaqi(pred_test[eaqi_poll])["EAQI"]
    
    return {
        "model": model_name,
        "obs_train": eaqi_obs_train,
        "pred_train": eaqi_pred_train,
        "obs_test": eaqi_obs_test,
        "pred_test": eaqi_pred_test,
    }

def plot_index_train_test(res,
                           model_name="VARX", figsize=(11, 5), COLORS=EAQI_COLORS, LABELS=EAQI_LABELS):
    obs_train  = res["obs_train"]
    pred_train = res["pred_train"]
    obs_test   = res["obs_test"]
    pred_test  = res["pred_test"]

    
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.grid(alpha=0.9)

    dates_train = obs_train.index
    dates_test  = obs_test.index
    dates_full  = dates_train.append(dates_test)

    full_obs = np.concatenate([obs_train.values, obs_test.values])
    ax.plot(dates_full, full_obs, lw=1.5,
             color="white", alpha=0.85, label="Real data")
    
    # In-sample 1-step na trainie
    ax.plot(dates_train, pred_train.values, 
            linestyle='--', color="blue", lw=1, label="In-sample 1-step (train)")
    
    # Walk-forward na teście
    ax.plot(dates_test, pred_test.values,
            color="#F029E2", lw=1.2, ls="--",
            label="Walk-forward 1-step (test)")
    
    # Linia oddzielająca train/test
    ax.axvline(dates_test[0], color="black", lw=1.5)
    
    # Tło kategorii
    for i, color in enumerate(COLORS):
        ax.axhspan(i + 0.5, i + 1.5, color=color, alpha=0.75)
    
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels(LABELS)
    ax.set_ylim(0.5, 6.5)
    ax.set_xlabel("Date")
    ax.set_title(f"EAQI Daily Air Quality Index — {model_name}")
    ax.legend(loc='lower left', prop={'size': 12})
    
    plt.tight_layout()
    return fig

def evaluate_eaqi_metrics(res, dataset="test", figsize=(7, 6)):
    obs = res[f"obs_{dataset}"]
    pred = res[f"pred_{dataset}"]
    model_name = res["model"]
    
    mask = obs.notna() & pred.notna()
    yt = obs[mask].astype(int).values
    yp = pred[mask].astype(int).values
    
    exact = (yt == yp).mean() * 100
    adj_1 = (np.abs(yt - yp) <= 1).mean() * 100
    qwk = cohen_kappa_score(yt, yp, weights='quadratic', labels=[1,2,3,4,5,6])
    acc = accuracy_score(yt, yp)
    acc_tol = np.mean(np.abs(np.asarray(yt) - np.asarray(yp)) <= 1)
    
    cm = confusion_matrix(yt, yp, labels=[1,2,3,4,5,6])
    
    print(f"\n{model_name} - {dataset}  (n = {len(yt)} days)")
    print(f"  Exact: {exact:.1f}% - correct category")
    print(f"  ±1:    {adj_1:.1f}% - correct or adjacent category")
    print(f"  QWK:   {qwk:.3f} - agreement with penalty for distance")
    print(f"  Accuracy:   {acc:.3f} - exact accuracy")
    print(f"  Accuracy ±1:   {acc_tol:.3f} - accuracy with toleration")
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EAQI_LABELS, yticklabels=EAQI_LABELS,
                cbar_kws={"label": "Number of days"}, ax=ax)
    ax.set_xlabel("Forecast")
    ax.set_ylabel("Observation")
    ax.set_title(f"{model_name} — {dataset}\n"
                  f"Exact: {exact:.1f}%   ±1: {adj_1:.1f}%   QWK: {qwk:.3f}")
    plt.tight_layout()
    return fig

############ MULTIVARIATE #########################

def get_predictions__ugm3(fit, df_train, df_test, pollutants, lambdas,
                          exog_test=None):
    bc_columns = [f'{p}_bc' for p in pollutants]
    
    # in-sample train
    pred_train_bc = fit.get_prediction(start=0, end=len(df_train) - 1).predicted_mean
    
    # walk-forward 1-step na teście 
    endog_test = df_test[bc_columns]
    if exog_test is not None:
        fit_on_test = fit.apply(endog=endog_test, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=endog_test)
    pred_test_bc = fit_on_test.get_prediction(start=0, end=len(df_test) - 1).predicted_mean
        
    # inverse Box-Cox 
    pred_train = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        lam = lambdas[p]
        pred_train[p] = inv_boxcox(pred_train_bc[f'{p}_bc'].values, lam)
        pred_test[p]  = inv_boxcox(pred_test_bc[f'{p}_bc'].values, lam)
    
    return pred_train, pred_test

def get_predictions_ugm3_sinus(fit, df_train, df_test, pollutants, lambdas,
                                exog_test=None):
    resid_columns = [f'{p}_non_sinus' for p in pollutants]
    
    # in-sample train
    pred_train_resid = fit.get_prediction(start=0, end=len(df_train) - 1).predicted_mean
    
    # walk-forward 1-step na teście
    endog_test = df_test[resid_columns]
    if exog_test is not None:
        fit_on_test = fit.apply(endog=endog_test, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=endog_test)
    pred_test_resid = fit_on_test.get_prediction(start=0, end=len(df_test) - 1).predicted_mean
    
    # dodaj sinus + inverse Box-Cox
    pred_train = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        lam = lambdas[p]
        sinus_train = df_train[f'{p}_sinus'].values
        sinus_test  = df_test[f'{p}_sinus'].values
        
        pred_train[p] = inv_boxcox(
            pred_train_resid[f'{p}_non_sinus'].values + sinus_train, lam
        )
        pred_test[p] = inv_boxcox(
            pred_test_resid[f'{p}_non_sinus'].values + sinus_test, lam
        )
    
    return pred_train, pred_test

def evaluate_index(fit, df_train, df_test, pollutants, lambdas, 
                       sinus=False, THRESHOLDS=EAQI_THRESHOLDS, exog_test=None, model_name="VAR"):
    if sinus:
        pred_train, pred_test = get_predictions_ugm3_sinus(
            fit, df_train, df_test, pollutants, lambdas, exog_test
        )
    else:
        pred_train, pred_test = get_predictions__ugm3(
            fit, df_train, df_test, pollutants, lambdas, exog_test
        )

    eaqi_poll = [p for p in pollutants if p in THRESHOLDS]
    
    eaqi_obs_train  = compute_eaqi(df_train[eaqi_poll])["EAQI"]
    eaqi_pred_train = compute_eaqi(pred_train[eaqi_poll])["EAQI"]
    eaqi_obs_test   = compute_eaqi(df_test[eaqi_poll])["EAQI"]
    eaqi_pred_test  = compute_eaqi(pred_test[eaqi_poll])["EAQI"]
    
    return {
        "model": model_name,
        "obs_train": eaqi_obs_train,
        "pred_train": eaqi_pred_train,
        "obs_test": eaqi_obs_test,
        "pred_test": eaqi_pred_test,
    }
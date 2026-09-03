import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import boxcox, inv_boxcox
import pandas as pd


##################### PLOT FULL PREDICTIONS ###############################
def plot_var(fit, df_train, df_test, pollutants, exog_test, lambdas, 
                title_prefix="VARMAX", figsize=(10, 4), alpha=0.05, burnin=5):

    pred_train = fit.get_prediction(start=0, end=len(df_train) - 1)
    pred_train_mean_resid = pred_train.predicted_mean
    pred_train_ci_resid = pred_train.conf_int(alpha=alpha)
    
    resid_columns = [f'{p}_non_sinus' for p in pollutants]
    endog_test_resid = df_test[resid_columns]
    
    if exog_test is not None:
        fit_on_test = fit.apply(endog=endog_test_resid, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=endog_test_resid)
    
    pred_test = fit_on_test.get_prediction(start=0, end=len(df_test) - 1)
    pred_test_mean_resid = pred_test.predicted_mean
    pred_test_ci_resid = pred_test.conf_int(alpha=alpha)
    
    pred_train = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    pred_train_ci_lower = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_train_ci_upper = pd.DataFrame(index=df_train.index, columns=pollutants, dtype=float)
    pred_test_ci_lower = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    pred_test_ci_upper = pd.DataFrame(index=df_test.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        resid_col = f'{p}_non_sinus'
        lam = lambdas[p]
        
        sinus_train = df_train[f'{p}_sinus'].values
        sinus_test = df_test[f'{p}_sinus'].values
        
        pred_train_bc_mean = pred_train_mean_resid[resid_col].values + sinus_train
        pred_test_bc_mean = pred_test_mean_resid[resid_col].values + sinus_test
        
        pred_train[p] = inv_boxcox(pred_train_bc_mean, lam)
        pred_test[p] = inv_boxcox(pred_test_bc_mean, lam)
        
        ci_lower_name = f'lower {resid_col}'
        ci_upper_name = f'upper {resid_col}'
        
        pred_train_ci_lower[p] = inv_boxcox(
            pred_train_ci_resid[ci_lower_name].values + sinus_train, lam
        )
        pred_train_ci_upper[p] = inv_boxcox(
            pred_train_ci_resid[ci_upper_name].values + sinus_train, lam
        )
        pred_test_ci_lower[p] = inv_boxcox(
            pred_test_ci_resid[ci_lower_name].values + sinus_test, lam
        )
        pred_test_ci_upper[p] = inv_boxcox(
            pred_test_ci_resid[ci_upper_name].values + sinus_test, lam
        )

    
    for col in pollutants:  
        idx_train = df_train.index
        idx_test = df_test.index
        dates_full  = idx_train.append(idx_test)                     
        full_obs = np.concatenate([df_train[col].values, df_test[col].values])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.grid()
        
    
        ax.plot(dates_full, full_obs, 'k.', alpha=0.6, label='Real data')
    
        ax.plot(idx_train, pred_train[col].values, color='C0',
                label='In-sample 1-step (train)')
        ax.fill_between(idx_train[burnin:],
                        pred_train_ci_lower[col].values[burnin:], 
                        pred_train_ci_upper[col].values[burnin:],  
                        color='C0', alpha=0.3)
        

        ax.plot(idx_test, pred_test[col].values, c='C2',
                label='Walk-forward 1-step (test)')
        ax.fill_between(idx_test,
                        pred_test_ci_lower[col].values,            
                        pred_test_ci_upper[col].values,           
                        color="C2", alpha=0.3)
        
        ax.axvline(idx_test[0], color='C3', linewidth=2)
        
    
        y_min = np.nanmin(full_obs)
        y_max = np.nanmax(full_obs)
        margin = (y_max - y_min) * 0.08
        ax.set_ylim(max(0, y_min - margin), y_max + margin)
        
        ax.set_title(f"{title_prefix} — {col}")
        ax.set_ylabel(col)
        ax.legend(loc='upper left',  prop = { "size": 13 })
        
        if exog_test is not None:
            plt.savefig(f'./plots/multi_dim_modeling/var_sinus_{col}_meteo.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'./plots/multi_dim_modeling/var_sinus_{col}.pdf', bbox_inches='tight')
        plt.show()

def plot_varmax(fit, df_train, df_test, pollutants, exog_test, lambdas,
                title_prefix="VARMAX", figsize=(9, 4), alpha=0.05, burnin=5):

    pred_train_bc = fit.get_prediction(start=0, end=len(df_train) - 1)
    pred_train_mean_bc = pred_train_bc.predicted_mean
    pred_train_ci_bc = pred_train_bc.conf_int(alpha=alpha)
    
    bc_columns = [f'{p}_bc' for p in pollutants]
    endog_test_bc = df_test[bc_columns]
    
    if exog_test is not None:
        fit_on_test = fit.apply(endog=endog_test_bc, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=endog_test_bc)
    
    pred_test_bc = fit_on_test.get_prediction(start=0, end=len(df_test) - 1)
    pred_test_mean_bc = pred_test_bc.predicted_mean
    pred_test_ci_bc = pred_test_bc.conf_int(alpha=alpha)
    
    pred_train = pd.DataFrame(index=pred_train_mean_bc.index, columns=pollutants, dtype=float)
    pred_test = pd.DataFrame(index=pred_test_mean_bc.index, columns=pollutants, dtype=float)
    
    pred_train_ci_lower = pd.DataFrame(index=pred_train_ci_bc.index, columns=pollutants, dtype=float)
    pred_train_ci_upper = pd.DataFrame(index=pred_train_ci_bc.index, columns=pollutants, dtype=float)
    pred_test_ci_lower = pd.DataFrame(index=pred_test_ci_bc.index, columns=pollutants, dtype=float)
    pred_test_ci_upper = pd.DataFrame(index=pred_test_ci_bc.index, columns=pollutants, dtype=float)
    
    for p in pollutants:
        bc_col = f'{p}_bc'
        lam_p = lambdas[p]                   
        
        pred_train[p] = inv_boxcox(pred_train_mean_bc[bc_col].values, lam_p)
        pred_test[p] = inv_boxcox(pred_test_mean_bc[bc_col].values, lam_p)
        
        ci_lower_name = f'lower {bc_col}'
        ci_upper_name = f'upper {bc_col}'
        
        pred_train_ci_lower[p] = inv_boxcox(pred_train_ci_bc[ci_lower_name].values, lam_p)
        pred_train_ci_upper[p] = inv_boxcox(pred_train_ci_bc[ci_upper_name].values, lam_p)
        pred_test_ci_lower[p] = inv_boxcox(pred_test_ci_bc[ci_lower_name].values, lam_p)
        pred_test_ci_upper[p] = inv_boxcox(pred_test_ci_bc[ci_upper_name].values, lam_p)
    
    
    for col in pollutants: 
        idx_train = df_train.index
        idx_test = df_test.index
        dates_full  = idx_train.append(idx_test)              
        full_obs = np.concatenate([df_train[col].values, df_test[col].values])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.grid()
        
    
        ax.plot(dates_full, full_obs, 'k.', alpha=0.6, label='Real data')
    
        ax.plot(idx_train, pred_train[col].values, color='C0',
                label='In-sample 1-step (train)')
        ax.fill_between(idx_train[burnin:],
                        pred_train_ci_lower[col].values[burnin:], 
                        pred_train_ci_upper[col].values[burnin:],  
                        color='C0', alpha=0.3)
        

        ax.plot(idx_test, pred_test[col].values, c='C2',
                label='Walk-forward 1-step (test)')
        ax.fill_between(idx_test,
                        pred_test_ci_lower[col].values,            
                        pred_test_ci_upper[col].values,           
                        color="C2", alpha=0.3)
        
        ax.axvline(idx_test[0], color='C3', linewidth=2)
        
    
        y_min = np.nanmin(full_obs)
        y_max = np.nanmax(full_obs)
        margin = (y_max - y_min) * 0.08
        ax.set_ylim(max(0, y_min - margin), y_max + margin)
        
        ax.set_title(f"{title_prefix} — {col}")
        ax.set_ylabel(col)
        ax.legend(loc='upper left',  prop = { "size": 13 })
        
        plt.tight_layout()
        
        if exog_test is not None and exog_test.shape[1] > 4:
            plt.savefig(f'./plots/multi_dim_modeling/varx_dumm_{col}_meteo.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'./plots/multi_dim_modeling/varx_dumm_{col}.pdf', bbox_inches='tight')
        plt.show()

##################### ERROR METRICS ON TEST SET ###############################
def evaluate_var(fit, df_test, pollutants, exog_test, lambdas):
    resid_columns = [f'{p}_non_sinus' for p in pollutants]
    endog_test_resid = df_test[resid_columns]
    
    if exog_test is not None:
        fit_on_test = fit.apply(endog=endog_test_resid, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=endog_test_resid)
    
    pred_resid = fit_on_test.fittedvalues   # DataFrame (n_test, n_polutants)
    
    rows = []
    for col in pollutants:
        resid_col = f'{col}_non_sinus'
        lam = lambdas[col]
        sinus_test = df_test[f'{col}_sinus'].values
        
        pred_bc = pred_resid[resid_col].values + sinus_test
        
        pred_orig = pd.Series(
            inv_boxcox(pred_bc, lam),
            index=pred_resid.index
        )
        

        y_obs = df_test[col]
        yp = pred_orig
        
        err = y_obs - yp
        ss_res = (err ** 2).sum()
        ss_tot = ((y_obs - y_obs.mean()) ** 2).sum()
        
        rows.append({
            "pollutant": col,
            "lambda": lam,
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAE":  err.abs().mean(),
            "R2":   1 - ss_res / ss_tot,
            "MAPE": (err / y_obs.replace(0, np.nan)).abs().mean() * 100,
        })
    
    return pd.DataFrame(rows).round(2)

def evaluate_varmax(fit, df_test, pollutants, exog_test, lambdas):
    bc_columns = [f'{p}_bc' for p in pollutants]
    Y_test_bc = df_test[bc_columns]
    
    if exog_test is not None:
        fit_on_test = fit.apply(endog=Y_test_bc, exog=exog_test)
    else:
        fit_on_test = fit.apply(endog=Y_test_bc)
    
    pred_test_bc = fit_on_test.fittedvalues
    
    rows = []
    for col in pollutants:
        bc_col = f'{col}_bc'
        lam = lambdas[col]
        
        pred_orig = pd.Series(
            inv_boxcox(pred_test_bc[bc_col].values, lam),
            index=pred_test_bc.index
        )
        

        y_obs = df_test[col]
        yp = pred_orig
        
        err = y_obs - yp
        ss_res = (err ** 2).sum()
        ss_tot = ((y_obs - y_obs.mean()) ** 2).sum()
        
        rows.append({
            "pollutant": col,
            "lambda": lam,
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAE":  err.abs().mean(),
            "R2":   1 - ss_res / ss_tot,
            "MAPE": (err / y_obs.replace(0, np.nan)).abs().mean() * 100,
        })
    
    return pd.DataFrame(rows).round(2)

##################### GRANGER ###############################
def granger_table(var_fit, pollutants):
    k = len(pollutants)
    pval = np.full((k, k), np.nan)
    for i, caused in enumerate(pollutants):
        for j, causing in enumerate(pollutants):
            if i == j:
                continue
            test = var_fit.test_causality(caused, [causing], kind="f")
            pval[i, j] = test.pvalue
    return pd.DataFrame(pval, index=pollutants, columns=pollutants)
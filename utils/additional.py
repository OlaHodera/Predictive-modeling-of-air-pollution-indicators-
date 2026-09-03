import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

##################### VIF ###############################

def vif_res(df, vars):
    X = df[vars]
    X_with_const = add_constant(X)   

    vif = pd.DataFrame({
        'variable': X_with_const.columns,
        'VIF': [variance_inflation_factor(X_with_const.values, i) 
                for i in range(X_with_const.shape[1])]
    })

    vif = vif[vif['variable'] != 'const'].sort_values('VIF', ascending=False)
    return vif


##################### LB TEST AND ACF ON RESIDUALS ###############################

def ljung_box_test(fitted_models, pollutants, model, lags=30):

    summary_results_lb = []
    for col in pollutants:
        
        residuals = fitted_models[col].resid
        
        lb_test_results = acorr_ljungbox(residuals, lags=lags, return_df=True)
        
        lb_test_results = lb_test_results.rename(columns={
            'lb_stat': 'test_statistic',
            'lb_pvalue': 'p_value'
        })
        
        # Passed means p_value > 0.05 (no significant autocorrelation)
        lb_test_results['passed_test'] = (lb_test_results['p_value'] > 0.05).astype(int)

        plt.figure(figsize=(8, 4))
        plt.plot(lb_test_results['p_value'], 'o')
        plt.axhline(0.05, color='red', linestyle='--', label='Significance level')
        plt.title(f'P-value from Ljung-Box test - {col}')
        plt.xlabel('Lag')
        plt.ylabel('P-Value')
        plt.savefig(f'./plots/one_dim_modeling/lb_{col}_{model}.pdf', bbox_inches='tight')
        plt.show()

        total_passed = lb_test_results['passed_test'].sum()
        total_lags = len(lb_test_results)
        
        all_lags_passed = int(total_passed == total_lags)
        
        summary_results_lb.append({
            'variable_name': col,
            'passed_lags_count': total_passed,
            'total_lags_tested': total_lags,
            'is_perfect_white_noise': all_lags_passed
        })
        
        plt.figure(figsize=(8, 4))
        ax = residuals.plot()
        ax.set_title(f'{col} - residuals')
        plt.savefig(f'./plots/one_dim_modeling/residuals_{col}_{model}.pdf', bbox_inches='tight')
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        plot_acf(residuals, lags=30, ax=axes, title=f'ACF — {col} residuals')
        # plot_pacf(residuals, lags=30, ax=axes[1], title=f'PACF — {col} residuals', method='ywm')
        plt.tight_layout()
        plt.savefig(f'./plots/one_dim_modeling/acf_residuals_{col}_{model}.pdf', bbox_inches='tight')
        plt.show()
    
    return pd.DataFrame(summary_results_lb)


################## additional stationarity check #####################

def check_stationarity(data, columns, sig=0.05):
    """
    ADF:  H_0 = niestacjonarny.  p < 0.05 -> stacjonarny
    KPSS: H_0 = stacjonarny.     p < 0.05 -> niestacjonarny

    - Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    - Case 2: Both tests conclude that the series is stationary - The series is stationary
    - Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    - Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
    -biblioteka stats

    """
    results = []
    for col in columns:
        residuals = data[col].resid
        name = col
        res_adf = adfuller(residuals, autolag='AIC')
        adf_stat, adf_p = round(res_adf[0], 3), round(res_adf[1], 3)
        adf_result = 'Stationary' if adf_p <= sig else 'Non-stationary'
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res_kpss = kpss(residuals, regression='c', nlags='auto')
        kpss_stat, kpss_p = round(res_kpss[0], 3), round(res_kpss[1], 3)
        kpss_result = 'Non-stationary' if kpss_p <= sig else 'Stationary'
        
        if adf_result == 'Stationary' and kpss_result == 'Stationary':
            conclusion = 'Stationary'
        elif adf_result == 'Non-stationary' and kpss_result == 'Non-stationary':
            conclusion = 'Non-stationary (trend)'
        elif adf_result == 'Stationary' and kpss_result == 'Non-stationary':
            conclusion = 'Seasonal/trend-stationary'
        else:
            conclusion = 'Difference-stationary'
        
        results.append({
            'variable':    name,
            'ADF_stat':    adf_stat,
            'ADF_p':       adf_p,
            'ADF_result':  adf_result,
            'KPSS_stat':   kpss_stat,
            'KPSS_p':      kpss_p,
            'KPSS_result': kpss_result,
            'conclusion':  conclusion
        })

    return pd.DataFrame(results)
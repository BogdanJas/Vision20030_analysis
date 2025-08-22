import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
import gc
from typing import Dict, Tuple, Optional, Any, List
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class ForecastingConfig:
    VISION_START_DATE = '2017-01-01'
    FORECAST_END_DATE = '2030-12-31'
    MIN_DATA_POINTS = 10
    SECTORS = ['gdp', 'technology', 'tourism']
    MODEL_TYPES = ['arima', 'ets', 'linear']
    VISION_YEARS = 13
    CONFIDENCE_INTERVAL = 1.96
    FORECAST_FREQUENCY = 'Q'
    
    GDP_MULTIPLIER_BASE = 1.05
    GDP_MULTIPLIER_MAX = 0.10
    TECH_MULTIPLIER_BASE = 1.10
    TECH_MULTIPLIER_MAX = 0.15
    TOURISM_MULTIPLIER_BASE = 1.08
    TOURISM_MULTIPLIER_MAX = 0.12

class Constants:
    VISION_YEARS = 13
    CONFIDENCE_INTERVAL = 1.96
    MIN_DATA_POINTS = 10
    FORECAST_FREQUENCY = 'Q'

def validate_input_data(main_df: pd.DataFrame) -> bool:
    required_columns = ['DateTime', 'GDP', 'Technology_Index', 'Tourism_Index', 'Oil_Price', 'Time_Trend']
    
    missing_columns = [col for col in required_columns if col not in main_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if main_df.empty:
        raise ValueError("Input dataframe is empty")
    
    if not pd.api.types.is_datetime64_any_dtype(main_df['DateTime']):
        raise ValueError("DateTime column must be datetime type")
    
    return True

def prepare_alternative_data(main_df: pd.DataFrame, cutoff_date: str = '2016-12-31', scenario: str = 'scenario1') -> Dict[str, pd.DataFrame]:
    if scenario == 'scenario1':
        pre_vision_data = main_df[main_df['DateTime'] <= cutoff_date].copy()
    else:
        pre_vision_data = main_df.copy()
    
    datasets = {}

    gdp_data = pre_vision_data[['DateTime', 'GDP', 'Oil_Price', 'Time_Trend']].dropna()
    if len(gdp_data) > ForecastingConfig.MIN_DATA_POINTS:
        datasets['gdp'] = gdp_data.copy()

    tech_data = pre_vision_data[['DateTime', 'Technology_Index', 'Oil_Price', 'Time_Trend']].dropna()
    if len(tech_data) > ForecastingConfig.MIN_DATA_POINTS:
        datasets['technology'] = tech_data.copy()

    tourism_data = pre_vision_data[['DateTime', 'Tourism_Index', 'Oil_Price', 'Time_Trend']].dropna()
    if len(tourism_data) > ForecastingConfig.MIN_DATA_POINTS:
        datasets['tourism'] = tourism_data.copy()
    
    return datasets

def apply_vision_2030_effects_vectorized(forecast_df: pd.DataFrame, sector: str, scenario: str = 'scenario2') -> pd.DataFrame:
    if scenario != 'scenario2':
        return forecast_df
    
    vision_start = pd.to_datetime(ForecastingConfig.VISION_START_DATE)
    vision_mask = forecast_df['ds'] >= vision_start
    
    if not vision_mask.any():
        return forecast_df
    
    years_since_vision = (forecast_df.loc[vision_mask, 'ds'] - vision_start).dt.days / 365.25
    
    if sector == 'gdp':
        multiplier = ForecastingConfig.GDP_MULTIPLIER_BASE + (ForecastingConfig.GDP_MULTIPLIER_MAX * years_since_vision / ForecastingConfig.VISION_YEARS)
    elif sector == 'technology':
        multiplier = ForecastingConfig.TECH_MULTIPLIER_BASE + (ForecastingConfig.TECH_MULTIPLIER_MAX * years_since_vision / ForecastingConfig.VISION_YEARS)
    elif sector == 'tourism':
        multiplier = ForecastingConfig.TOURISM_MULTIPLIER_BASE + (ForecastingConfig.TOURISM_MULTIPLIER_MAX * years_since_vision / ForecastingConfig.VISION_YEARS)
    
    forecast_df.loc[vision_mask, 'yhat'] *= multiplier
    forecast_df.loc[vision_mask, 'yhat_lower'] *= multiplier * 0.95
    forecast_df.loc[vision_mask, 'yhat_upper'] *= multiplier * 1.05
    
    return forecast_df

def extract_target_variable(sector: str, data: pd.DataFrame) -> np.ndarray:
    if sector == 'gdp':
        return data['GDP'].values
    elif sector == 'technology':
        return data['Technology_Index'].values
    else:
        return data['Tourism_Index'].values

def create_arima_models(datasets: Dict[str, pd.DataFrame], scenario: str = 'scenario1', test_mode: bool = False) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    arima_models = {}
    arima_forecasts = {}
    
    for sector, data in datasets.items():
        try:
            y = extract_target_variable(sector, data)
            
            if len(y) < ForecastingConfig.MIN_DATA_POINTS:
                continue
            
            y = y[~np.isnan(y)]
            if len(y) < ForecastingConfig.MIN_DATA_POINTS:
                continue
            
            best_aic = np.inf
            best_model = None
            
            if test_mode:
                p_range, d_range, q_range = range(2), range(1), range(2)
            else:
                if sector == 'gdp':
                    p_range, d_range, q_range = range(4), range(2), range(4)
                elif sector == 'technology':
                    p_range, d_range, q_range = range(3), range(2), range(3)
                else:
                    p_range, d_range, q_range = range(2), range(2), range(2)
            
            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                        except (ValueError, np.linalg.LinAlgError):
                            continue
            
            if best_model is not None:
                arima_models[sector] = best_model
                
                try:
                    forecast_periods = pd.date_range(
                        start=data['DateTime'].max() + pd.DateOffset(days=1),
                        end=ForecastingConfig.FORECAST_END_DATE,
                        freq=ForecastingConfig.FORECAST_FREQUENCY
                    )
                    
                    forecast_steps = len(forecast_periods)
                    forecast_result = best_model.forecast(steps=forecast_steps)
                    forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
                    
                    if hasattr(forecast_ci, 'iloc'):
                        forecast_lower = forecast_ci.iloc[:, 0]
                        forecast_upper = forecast_ci.iloc[:, 1]
                    else:
                        forecast_lower = forecast_ci[:, 0]
                        forecast_upper = forecast_ci[:, 1]
                    
                    forecast_df = pd.DataFrame({
                        'ds': forecast_periods,
                        'yhat': forecast_result,
                        'yhat_lower': forecast_lower,
                        'yhat_upper': forecast_upper
                    })
                    
                    forecast_df = apply_vision_2030_effects_vectorized(forecast_df, sector, scenario)
                    arima_forecasts[sector] = forecast_df
                    
                except Exception as e:
                    logger.warning(f"Failed to generate ARIMA forecast for {sector}: {e}")
                    arima_models[sector] = best_model
                
        except Exception as e:
            logger.error(f"Unexpected error creating ARIMA model for {sector}: {e}")
    
    return arima_models, arima_forecasts

def create_exponential_smoothing_models(datasets: Dict[str, pd.DataFrame], scenario: str = 'scenario1') -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    ets_models = {}
    ets_forecasts = {}
    
    for sector, data in datasets.items():
        try:
            y = extract_target_variable(sector, data)
            
            best_aic = np.inf
            best_model = None

            trends = ['add', 'mul', None]
            seasonals = ['add', 'mul', None]
            
            for trend in trends:
                for seasonal in seasonals:
                    try:
                        seasonal_periods = 4 if seasonal is not None else None
                        
                        if seasonal is not None and len(y) < 2 * seasonal_periods:
                            continue
                        
                        model = ExponentialSmoothing(
                            y, 
                            trend=trend, 
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        )
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except (ValueError, np.linalg.LinAlgError):
                        continue
            
            if best_model is not None:
                ets_models[sector] = best_model
                forecast_periods = pd.date_range(
                    start=data['DateTime'].max() + pd.DateOffset(days=1),
                    end=ForecastingConfig.FORECAST_END_DATE,
                    freq=ForecastingConfig.FORECAST_FREQUENCY
                )
                
                forecast_steps = len(forecast_periods)
                forecast_result = best_model.forecast(steps=forecast_steps)
                
                forecast_std = np.std(best_model.resid)
                forecast_lower = forecast_result - ForecastingConfig.CONFIDENCE_INTERVAL * forecast_std
                forecast_upper = forecast_result + ForecastingConfig.CONFIDENCE_INTERVAL * forecast_std
                
                forecast_df = pd.DataFrame({
                    'ds': forecast_periods,
                    'yhat': forecast_result,
                    'yhat_lower': forecast_lower,
                    'yhat_upper': forecast_upper
                })
                
                forecast_df = apply_vision_2030_effects_vectorized(forecast_df, sector, scenario)
                ets_forecasts[sector] = forecast_df
                
        except Exception as e:
            logger.error(f"Unexpected error creating ETS model for {sector}: {e}")
    
    return ets_models, ets_forecasts

def create_linear_trend_models(datasets: Dict[str, pd.DataFrame], scenario: str = 'scenario1') -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    linear_models = {}
    linear_forecasts = {}
    
    for sector, data in datasets.items():
        try:
            y = extract_target_variable(sector, data)
            X = data[['Time_Trend', 'Oil_Price']].values

            best_r2 = -np.inf
            best_model = None
            
            if sector == 'gdp':
                degree_range = [1, 2, 3, 4]
            elif sector == 'tourism':
                degree_range = [1, 2]
            else:
                degree_range = [1, 2, 3]
            
            for degree in degree_range:
                try:
                    poly_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                    
                    poly_model.fit(X, y)
                    y_pred = poly_model.predict(X)
                    r2 = r2_score(y, y_pred)
                    
                    if len(y) > ForecastingConfig.MIN_DATA_POINTS:
                        split_point = len(y) // 2
                        X_train, X_test = X[:split_point], X[split_point:]
                        y_train, y_test = y[:split_point], y[split_point:]
                        
                        if len(X_test) > 0:
                            poly_model_train = Pipeline([
                                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                                ('linear', LinearRegression())
                            ])
                            poly_model_train.fit(X_train, y_train)
                            y_pred_test = poly_model_train.predict(X_test)
                            r2_test = r2_score(y_test, y_pred_test)
                            adjusted_r2 = (r2 + r2_test) / 2
                        else:
                            adjusted_r2 = r2
                    else:
                        adjusted_r2 = r2
                    
                    if adjusted_r2 > best_r2:
                        best_r2 = adjusted_r2
                        best_model = poly_model
                except (ValueError, np.linalg.LinAlgError):
                    continue
            
            if best_model is not None:
                linear_models[sector] = best_model
            
                forecast_periods = pd.date_range(
                    start=data['DateTime'].max() + pd.DateOffset(days=1),
                    end=ForecastingConfig.FORECAST_END_DATE,
                    freq=ForecastingConfig.FORECAST_FREQUENCY
                )
                
                last_time_trend = data['Time_Trend'].max()
                future_time_trends = np.arange(
                    last_time_trend + 1, 
                    last_time_trend + 1 + len(forecast_periods)
                )
                
                last_oil_price = data['Oil_Price'].iloc[-1]
                future_oil_prices = np.full(len(forecast_periods), last_oil_price)
                
                X_future = np.column_stack([future_time_trends, future_oil_prices])
                forecast_result = best_model.predict(X_future)
                
                if sector == 'gdp':
                    last_actual_gdp = y[-1]
                    for i in range(len(forecast_result)):
                        max_growth = last_actual_gdp * 1.15
                        min_growth = last_actual_gdp * 0.90
                        forecast_result[i] = np.clip(forecast_result[i], min_growth, max_growth)
                        last_actual_gdp = forecast_result[i]
                
                elif sector == 'tourism':
                    forecast_result = np.clip(forecast_result, 0, 100)
                
                y_pred_train = best_model.predict(X)
                residuals = y - y_pred_train
                forecast_std = np.std(residuals)
                
                forecast_lower = forecast_result - ForecastingConfig.CONFIDENCE_INTERVAL * forecast_std
                forecast_upper = forecast_result + ForecastingConfig.CONFIDENCE_INTERVAL * forecast_std
                
                forecast_df = pd.DataFrame({
                    'ds': forecast_periods,
                    'yhat': forecast_result,
                    'yhat_lower': forecast_lower,
                    'yhat_upper': forecast_upper
                })
                
                forecast_df = apply_vision_2030_effects_vectorized(forecast_df, sector, scenario)
                linear_forecasts[sector] = forecast_df
                
        except Exception as e:
            logger.error(f"Unexpected error creating linear model for {sector}: {e}")
    
    return linear_models, linear_forecasts

def compare_model_performance(datasets: Dict[str, pd.DataFrame], arima_models: Dict[str, Any], ets_models: Dict[str, Any], linear_models: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    performance_comparison = {}
    
    for sector, data in datasets.items():
        y_actual = extract_target_variable(sector, data)
        sector_performance = {}
        
        if sector in arima_models:
            try:
                y_pred_arima = arima_models[sector].fittedvalues
                min_len = min(len(y_actual), len(y_pred_arima))
                mse_arima = mean_squared_error(y_actual[:min_len], y_pred_arima[:min_len])
                mae_arima = mean_absolute_error(y_actual[:min_len], y_pred_arima[:min_len])
                r2_arima = r2_score(y_actual[:min_len], y_pred_arima[:min_len])
                
                sector_performance['ARIMA'] = {
                    'MSE': mse_arima,
                    'MAE': mae_arima,
                    'R2': r2_arima,
                    'RMSE': np.sqrt(mse_arima)
                }
            except Exception as e:
                logger.warning(f"Failed to calculate ARIMA performance for {sector}: {e}")
        
        if sector in ets_models:
            try:
                y_pred_ets = ets_models[sector].fittedvalues
                min_len = min(len(y_actual), len(y_pred_ets))
                mse_ets = mean_squared_error(y_actual[:min_len], y_pred_ets[:min_len])
                mae_ets = mean_absolute_error(y_actual[:min_len], y_pred_ets[:min_len])
                r2_ets = r2_score(y_actual[:min_len], y_pred_ets[:min_len])
                
                sector_performance['ETS'] = {
                    'MSE': mse_ets,
                    'MAE': mae_ets,
                    'R2': r2_ets,
                    'RMSE': np.sqrt(mse_ets)
                }
            except Exception as e:
                logger.warning(f"Failed to calculate ETS performance for {sector}: {e}")
        
        if sector in linear_models:
            try:
                X = data[['Time_Trend', 'Oil_Price']].values
                y_pred_linear = linear_models[sector].predict(X)
                mse_linear = mean_squared_error(y_actual, y_pred_linear)
                mae_linear = mean_absolute_error(y_actual, y_pred_linear)
                r2_linear = r2_score(y_actual, y_pred_linear)
                
                sector_performance['Linear'] = {
                    'MSE': mse_linear,
                    'MAE': mae_linear,
                    'R2': r2_linear,
                    'RMSE': np.sqrt(mse_linear)
                }
            except Exception as e:
                logger.warning(f"Failed to calculate linear model performance for {sector}: {e}")
        
        performance_comparison[sector] = sector_performance

    return performance_comparison

def create_scenario_comparison_visualization(scenario1_results: Dict[str, Any], scenario2_results: Dict[str, Any], main_df: pd.DataFrame) -> Optional[plt.Figure]:
    if not scenario1_results or not scenario2_results:
        return None
    
    historical_data = {}
    for sector in ForecastingConfig.SECTORS:
        if sector == 'gdp':
            historical_data[sector] = main_df[['DateTime', 'GDP']].dropna()
        elif sector == 'technology':
            historical_data[sector] = main_df[['DateTime', 'Technology_Index']].dropna()
        else:
            historical_data[sector] = main_df[['DateTime', 'Tourism_Index']].dropna()
    
    sectors = ForecastingConfig.SECTORS
    n_sectors = len(sectors)
    
    fig, axes = plt.subplots(n_sectors, 2, figsize=(20, 6 * n_sectors))
    if n_sectors == 1:
        axes = axes.reshape(1, -1)
    
    colors = {
        'scenario1': '#1f77b4',
        'scenario2': '#ff7f0e',
        'historical': '#2ca02c'
    }
    
    for i, sector in enumerate(sectors):
        ax1 = axes[i, 0]
        
        if sector in historical_data:
            hist_data = historical_data[sector]
            if sector == 'gdp':
                ax1.plot(hist_data['DateTime'], hist_data['GDP'], 
                        color=colors['historical'], linewidth=2, label='Historical', alpha=0.7)
            elif sector == 'technology':
                ax1.plot(hist_data['DateTime'], hist_data['Technology_Index'], 
                        color=colors['historical'], linewidth=2, label='Historical', alpha=0.7)
            else:
                ax1.plot(hist_data['DateTime'], hist_data['Tourism_Index'], 
                        color=colors['historical'], linewidth=2, label='Historical', alpha=0.7)
        
        for scenario_name, scenario_data in [('Scenario 1 (Baseline)', scenario1_results), 
                                           ('Scenario 2 (Vision 2030)', scenario2_results)]:
            if 'arima_forecasts' in scenario_data and sector in scenario_data['arima_forecasts']:
                forecast = scenario_data['arima_forecasts'][sector]
                color = colors['scenario1'] if 'Baseline' in scenario_name else colors['scenario2']
                ax1.plot(forecast['ds'], forecast['yhat'], 
                        color=color, linewidth=2, linestyle='--', 
                        label=f'{scenario_name} - ARIMA')
        
        ax1.axvline(x=pd.to_datetime(ForecastingConfig.VISION_START_DATE), color='red', linestyle='-', 
                   linewidth=2, label='Vision 2030 Start')
        ax1.set_title(f'{sector.upper()} - Historical vs Forecasts')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        
        ax2 = axes[i, 1]
        
        for scenario_name, scenario_data in [('Scenario 1', scenario1_results), 
                                           ('Scenario 2', scenario2_results)]:
            color = colors['scenario1'] if '1' in scenario_name else colors['scenario2']
            
            for model_type in ['arima_forecasts', 'ets_forecasts', 'linear_forecasts']:
                if model_type in scenario_data and sector in scenario_data[model_type]:
                    forecast = scenario_data[model_type][sector]
                    model_name = model_type.replace('_forecasts', '').upper()
                    ax2.plot(forecast['ds'], forecast['yhat'], 
                            color=color, linewidth=2, alpha=0.7,
                            label=f'{scenario_name} - {model_name}')
        
        ax2.axvline(x=pd.to_datetime(ForecastingConfig.VISION_START_DATE), color='red', linestyle='-', 
                   linewidth=2, label='Vision 2030 Start')
        ax2.set_title(f'{sector.upper()} - Model Comparison Across Scenarios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
    
    plt.tight_layout()
    plt.suptitle("Scenario Comparison: Baseline vs Vision 2030", y=1.02, fontsize=16)
    
    import os
    results_dir = 'Results/alternative_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        plt.savefig(f"{results_dir}/scenario_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save scenario comparison plot: {e}")
        try:
            plt.savefig(f"{results_dir}/scenario_comparison_plots.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e2:
            logger.error(f"Failed to save scenario comparison plot with lower DPI: {e2}")
    
    return fig

def create_vision_2030_impact_analysis(scenario1_results: Dict[str, Any], scenario2_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    impact_analysis = {}
    
    for sector in ForecastingConfig.SECTORS:
        sector_impact = {}
        
        for model_type in ['arima_forecasts', 'ets_forecasts', 'linear_forecasts']:
            if (model_type in scenario1_results and sector in scenario1_results[model_type] and
                model_type in scenario2_results and sector in scenario2_results[model_type]):
                
                scenario1_forecast = scenario1_results[model_type][sector]
                scenario2_forecast = scenario2_results[model_type][sector]
                
                baseline_2030 = scenario1_forecast[scenario1_forecast['ds'] == ForecastingConfig.FORECAST_END_DATE]['yhat'].iloc[0]
                vision_2030 = scenario2_forecast[scenario2_forecast['ds'] == ForecastingConfig.FORECAST_END_DATE]['yhat'].iloc[0]
                
                absolute_impact = vision_2030 - baseline_2030
                percentage_impact = ((vision_2030 - baseline_2030) / baseline_2030) * 100
                
                vision_start = pd.to_datetime(ForecastingConfig.VISION_START_DATE)
                scenario1_post_vision = scenario1_forecast[scenario1_forecast['ds'] >= vision_start]
                scenario2_post_vision = scenario2_forecast[scenario2_forecast['ds'] >= vision_start]
                
                cumulative_baseline = scenario1_post_vision['yhat'].sum()
                cumulative_vision = scenario2_post_vision['yhat'].sum()
                cumulative_impact = cumulative_vision - cumulative_baseline
                cumulative_percentage = ((cumulative_vision - cumulative_baseline) / cumulative_baseline) * 100
                
                sector_impact[model_type.replace('_forecasts', '')] = {
                    'baseline_2030': baseline_2030,
                    'vision_2030': vision_2030,
                    'absolute_impact_2030': absolute_impact,
                    'percentage_impact_2030': percentage_impact,
                    'cumulative_impact': cumulative_impact,
                    'cumulative_percentage_impact': cumulative_percentage
                }
        
        impact_analysis[sector] = sector_impact
    
    return impact_analysis

def create_impact_visualization(impact_analysis: Dict[str, Dict[str, Dict[str, float]]]) -> Optional[plt.Figure]:
    if not impact_analysis:
        return None
    
    sectors = list(impact_analysis.keys())
    models = ['arima', 'ets', 'linear']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    x_pos = np.arange(len(sectors))
    width = 0.25
    
    for i, model in enumerate(models):
        impacts = []
        for sector in sectors:
            if model in impact_analysis[sector]:
                impacts.append(impact_analysis[sector][model]['percentage_impact_2030'])
            else:
                impacts.append(0)
        
        ax1.bar(x_pos + i*width, impacts, width, label=model.upper(), alpha=0.8)
    
    ax1.set_xlabel('Sectors')
    ax1.set_ylabel('Percentage Impact (%)')
    ax1.set_title('Vision 2030 Impact by 2030')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels([s.upper() for s in sectors])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        impacts = []
        for sector in sectors:
            if model in impact_analysis[sector]:
                impacts.append(impact_analysis[sector][model]['cumulative_percentage_impact'])
            else:
                impacts.append(0)
        
        ax2.bar(x_pos + i*width, impacts, width, label=model.upper(), alpha=0.8)
    
    ax2.set_xlabel('Sectors')
    ax2.set_ylabel('Cumulative Percentage Impact (%)')
    ax2.set_title('Cumulative Vision 2030 Impact (2017-2030)')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels([s.upper() for s in sectors])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    for i, model in enumerate(models):
        impacts = []
        for sector in sectors:
            if model in impact_analysis[sector]:
                impacts.append(impact_analysis[sector][model]['absolute_impact_2030'])
            else:
                impacts.append(0)
        
        ax3.bar(x_pos + i*width, impacts, width, label=model.upper(), alpha=0.8)
    
    ax3.set_xlabel('Sectors')
    ax3.set_ylabel('Absolute Impact')
    ax3.set_title('Absolute Vision 2030 Impact by 2030')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels([s.upper() for s in sectors])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    model_performance = {}
    for model in models:
        avg_impact = []
        for sector in sectors:
            if model in impact_analysis[sector]:
                avg_impact.append(impact_analysis[sector][model]['percentage_impact_2030'])
        if avg_impact:
            model_performance[model.upper()] = np.mean(avg_impact)
    
    if model_performance:
        models_list = list(model_performance.keys())
        impacts_list = list(model_performance.values())
        ax4.bar(models_list, impacts_list, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_ylabel('Average Percentage Impact (%)')
        ax4.set_title('Average Vision 2030 Impact by Model')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Vision 2030 Impact Analysis", y=1.02, fontsize=16)
    
    import os
    results_dir = 'Results/alternative_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        plt.savefig(f"{results_dir}/vision_2030_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save impact analysis plot: {e}")
        try:
            plt.savefig(f"{results_dir}/vision_2030_impact_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e2:
            logger.error(f"Failed to save impact analysis plot with lower DPI: {e2}")
    
    return fig

def save_alternative_results(arima_forecasts: Dict[str, pd.DataFrame], ets_forecasts: Dict[str, pd.DataFrame], 
                           linear_forecasts: Dict[str, pd.DataFrame], performance_comparison: Dict[str, Dict[str, Dict[str, float]]], 
                           scenario: str = 'scenario1') -> bool:
    import os
    results_dir = 'Results/alternative_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    for model_type, forecasts in [
        ('arima', arima_forecasts),
        ('ets', ets_forecasts), 
        ('linear', linear_forecasts)
    ]:
        for sector, forecast in forecasts.items():
            filename = f'{results_dir}/{scenario}_forecast_{sector}_{model_type}.csv'
            forecast.to_csv(filename, index=False)
    
    perf_df = pd.DataFrame(performance_comparison)
    perf_df.to_csv(f'{results_dir}/{scenario}_model_performance_comparison.csv')
    
    return True

def run_scenario(main_df: pd.DataFrame, scenario: str) -> Optional[Dict[str, Any]]:
    datasets = prepare_alternative_data(main_df, scenario=scenario)
    
    if not datasets:
        return None

    arima_models, arima_forecasts = create_arima_models(datasets, scenario=scenario)
    ets_models, ets_forecasts = create_exponential_smoothing_models(datasets, scenario=scenario)
    linear_models, linear_forecasts = create_linear_trend_models(datasets, scenario=scenario)
    performance_comparison = compare_model_performance(datasets, arima_models, ets_models, linear_models)
    
    return {
        'datasets': datasets,
        'arima_models': arima_models,
        'arima_forecasts': arima_forecasts,
        'ets_models': ets_models,
        'ets_forecasts': ets_forecasts,
        'linear_models': linear_models,
        'linear_forecasts': linear_forecasts,
        'performance_comparison': performance_comparison
    }

def create_alternative_visualizations(scenario1_results: Dict[str, Any], scenario2_results: Dict[str, Any], main_df: pd.DataFrame) -> None:
    create_scenario_comparison_visualization(scenario1_results, scenario2_results, main_df)
    impact_analysis = create_vision_2030_impact_analysis(scenario1_results, scenario2_results)
    create_impact_visualization(impact_analysis)

def save_all_results(scenario1_results: Dict[str, Any], scenario2_results: Dict[str, Any], impact_analysis: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    if scenario1_results:
        save_alternative_results(
            scenario1_results['arima_forecasts'], 
            scenario1_results['ets_forecasts'], 
            scenario1_results['linear_forecasts'], 
            scenario1_results['performance_comparison'], 
            'scenario1'
        )
    
    if scenario2_results:
        save_alternative_results(
            scenario2_results['arima_forecasts'], 
            scenario2_results['ets_forecasts'], 
            scenario2_results['linear_forecasts'], 
            scenario2_results['performance_comparison'], 
            'scenario2'
        )
    
    import os
    results_dir = 'Results/alternative_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    impact_df = pd.DataFrame(impact_analysis)
    impact_df.to_csv(f'{results_dir}/vision_2030_impact_analysis.csv')

def run_alternative_forecasting(main_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        validate_input_data(main_df)
        
        scenario1_results = run_scenario(main_df, 'scenario1')
        scenario2_results = run_scenario(main_df, 'scenario2')
        
        if not scenario2_results:
            return scenario1_results
        
        create_alternative_visualizations(scenario1_results, scenario2_results, main_df)
        impact_analysis = create_vision_2030_impact_analysis(scenario1_results, scenario2_results)
        save_all_results(scenario1_results, scenario2_results, impact_analysis)
        
        return {
            'scenario1_results': scenario1_results,
            'scenario2_results': scenario2_results,
            'impact_analysis': impact_analysis
        }
    finally:
        gc.collect()

if __name__ == "__main__":
    exec(open('data_loading_and_EDA.py').read())
    results_alternative_forecasting = run_alternative_forecasting(main_df)
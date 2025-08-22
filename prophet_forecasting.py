import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_prophet_data(main_df, cutoff_date=None):
    if cutoff_date:
        data = main_df[main_df['DateTime'] <= cutoff_date].copy()
    else:
        data = main_df.copy()
    
    datasets = {}
    sectors = {
        'gdp': 'GDP',
        'technology': 'Technology_Index', 
        'tourism': 'Tourism_Index'
    }
    
    for sector, column in sectors.items():
        sector_data = data[['DateTime', column, 'Oil_Price']].dropna()
        if len(sector_data) > 5:
            datasets[sector] = pd.DataFrame({
                'ds': sector_data['DateTime'],
                'y': sector_data[column],
                'oil_price': sector_data['Oil_Price']
            })
    
    return datasets, data

def create_enhanced_prophet_models(datasets, scenario_type='scenario1'):
    models = {}
    fitted_models = {}
    
    for sector, data in datasets.items():
        data_stats = data['y'].describe()
        data_range = data_stats['max'] - data_stats['min']
        data_std = data_stats['std']
        data_cv = data_std / data_stats['mean'] if data_stats['mean'] != 0 else 0
        
        if sector == 'gdp':
            model = Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.85
            )
            
        elif sector == 'technology':
            model = Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.15,
                seasonality_prior_scale=15,
                holidays_prior_scale=10,
                interval_width=0.8
            )
            
        else:  # tourism
            model = Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.12,
                seasonality_prior_scale=15,
                holidays_prior_scale=12,
                interval_width=0.8
            )
        
        if 'oil_price' in data.columns and not data['oil_price'].isnull().all():
            oil_correlation = data['y'].corr(data['oil_price'])
            if abs(oil_correlation) > 0.3:
                model.add_regressor('oil_price', prior_scale=10000, mode='additive')
        
        if scenario_type == 'scenario2':
            if sector == 'gdp':
                model.changepoint_prior_scale = 0.08
            elif sector == 'technology':
                model.changepoint_prior_scale = 0.2
            elif sector == 'tourism':
                model.changepoint_prior_scale = 0.15
        
        try:
            fitted_model = model.fit(data)
            
            forecast = fitted_model.predict(data)
            mse = mean_squared_error(data['y'], forecast['yhat'])
            r2 = r2_score(data['y'], forecast['yhat'])
            
            min_r2 = 0.2 if sector in ['tourism'] else 0.3
            if r2 > min_r2:
                models[sector] = model
                fitted_models[sector] = fitted_model
            else:
                if sector == 'tourism' and r2 <= min_r2:
                    try:
                        simple_model = Prophet(
                            growth='linear',
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            seasonality_mode='additive',
                            changepoint_prior_scale=0.05,
                            seasonality_prior_scale=5,
                            holidays_prior_scale=5,
                            interval_width=0.8
                        )
                        simple_fitted = simple_model.fit(data)
                        simple_forecast = simple_fitted.predict(data)
                        simple_r2 = r2_score(data['y'], simple_forecast['yhat'])
                        if simple_r2 > 0.1:
                            models[sector] = simple_model
                            fitted_models[sector] = simple_fitted
                    except Exception as e:
                        pass
                
        except Exception as e:
            continue
    
    return models, fitted_models

def validate_economic_plausibility(forecast_value, sector, historical_range):
    if sector == 'gdp':
        if forecast_value < historical_range[0] * 0.8 or forecast_value > historical_range[1] * 1.15:
            return False, f"GDP forecast {forecast_value:.2f} outside plausible range [{historical_range[0]*0.8:.2f}, {historical_range[1]*1.15:.2f}]"
    elif sector == 'technology':
        if forecast_value < 0 or forecast_value > 100:
            return False, f"Technology index {forecast_value:.2f} outside valid range [0, 100]"
    elif sector == 'tourism':
        if forecast_value < 0 or forecast_value > 100:
            return False, f"Tourism index {forecast_value:.2f} outside valid range [0, 100]"
    
    return True, "Plausible"



def perform_cross_validation(fitted_models, datasets):
    cv_results = {}
    performance_results = {}
    
    for sector, model in fitted_models.items():
        try:
            data_length = len(datasets[sector])
            
            if data_length >= 24:
                initial = f'{max(365, int(data_length * 0.6))} days'
                period = f'{max(90, int(data_length * 0.1))} days'
                horizon = f'{max(90, int(data_length * 0.2))} days'
            else:
                initial = f'{max(365, int(data_length * 0.7))} days'
                period = f'{max(90, int(data_length * 0.1))} days'
                horizon = f'{max(90, int(data_length * 0.15))} days'
            
            df_cv = cross_validation(
                model, 
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="threads"
            )
            
            df_performance = performance_metrics(df_cv)
            
            cv_results[sector] = df_cv
            performance_results[sector] = df_performance
            
        except Exception as e:
            cv_results[sector] = None
            performance_results[sector] = None
    
    return cv_results, performance_results

def generate_counterfactual_forecasts(fitted_models, datasets, main_df, forecast_end='2030-12-31'):
    forecasts = {}
    future_dataframes = {}
    
    last_oil_price = main_df['Oil_Price'].dropna().iloc[-1]
    
    for sector, model in fitted_models.items():
        try:
            last_date = datasets[sector]['ds'].max()
            future_periods = pd.date_range(
                start=last_date + pd.DateOffset(days=1),
                end=forecast_end,
                freq='Q'
            )
            
            all_dates = pd.concat([
                datasets[sector]['ds'],
                pd.Series(future_periods)
            ]).reset_index(drop=True)
            
            future = pd.DataFrame({'ds': all_dates})
            historical_oil_df = datasets[sector][['ds', 'oil_price']].copy()
            n_future = len(future_periods)
            recent_oil = main_df[main_df['DateTime'] >= '2015-01-01']['Oil_Price'].dropna()
            if len(recent_oil) > 4:
                oil_trend = np.polyfit(range(len(recent_oil)), recent_oil, 1)[0]
            else:
                oil_trend = 0
            
            future_oil_base = np.arange(n_future) * oil_trend + last_oil_price
            future_oil_noise = np.random.normal(0, last_oil_price * 0.1, n_future)  
            future_oil = future_oil_base + future_oil_noise
            future_oil_df = pd.DataFrame({'ds': future_periods, 'oil_price': future_oil})
            
            oil_full = pd.concat([historical_oil_df, future_oil_df], ignore_index=True)
            oil_full = oil_full.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')

            future = future.merge(oil_full, on='ds', how='left')
            future['oil_price'] = future['oil_price'].ffill().bfill().fillna(last_oil_price)
            forecast = model.predict(future)
            
            historical_data = datasets[sector]['y']
            historical_range = (historical_data.min(), historical_data.max())
            
            forecast_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
            is_plausible, message = validate_economic_plausibility(forecast_2030, sector, historical_range)
            
            if not is_plausible:
                if sector == 'gdp':
                    lower_bound = max(0, historical_range[0] * 0.8)
                    upper_bound = historical_range[1] * 1.2
                    forecast['yhat'] = np.clip(forecast['yhat'], lower_bound, upper_bound)
                    forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], max(0, historical_range[0] * 0.7), historical_range[1] * 1.1)
                    forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], max(0, historical_range[0] * 0.9), historical_range[1] * 1.3)
                elif sector in ['technology', 'tourism']:
                    forecast['yhat'] = np.clip(forecast['yhat'], 0, 100)
                    forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, 100)
                    forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, 100)
                
                forecast_2030_after = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
                is_plausible_after, message_after = validate_economic_plausibility(forecast_2030_after, sector, historical_range)
                if not is_plausible_after:
                    if sector == 'gdp':
                        recent_trend = np.polyfit(range(len(historical_data[-10:])), historical_data[-10:], 1)[0]
                        base_value = historical_data.iloc[-1]
                        years_to_2030 = 7
                        conservative_forecast = base_value + (recent_trend * years_to_2030)
                        conservative_forecast = max(0, conservative_forecast)
                        
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat'] = conservative_forecast
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat_lower'] = conservative_forecast * 0.9
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat_upper'] = conservative_forecast * 1.1
                        
                    elif sector in ['technology', 'tourism']:
                        recent_trend = np.polyfit(range(len(historical_data[-10:])), historical_data[-10:], 1)[0]
                        base_value = historical_data.iloc[-1]
                        years_to_2030 = 7
                        conservative_forecast = base_value + (recent_trend * years_to_2030)
                        conservative_forecast = np.clip(conservative_forecast, 0, 100)
                        
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat'] = conservative_forecast
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat_lower'] = max(0, conservative_forecast - 5)
                        forecast.loc[forecast['ds'].dt.year == 2030, 'yhat_upper'] = min(100, conservative_forecast + 5)
            
            forecast_std = forecast['yhat'].std()
            historical_std = historical_data.std()
            
            if forecast_std > historical_std * 3:
                forecast['yhat'] = forecast['yhat'].rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            forecasts[sector] = forecast
            future_dataframes[sector] = future
 
        except Exception as e:
            forecasts[sector] = None
            future_dataframes[sector] = None
    
    return forecasts, future_dataframes

def create_forecast_plots(datasets, forecasts, title, filename, scenario_type='scenario1'):
    n_sectors = len([f for f in forecasts.values() if f is not None])
    if n_sectors == 0:
        return None
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']
    
    for i, (sector, forecast) in enumerate(forecasts.items()):
        if forecast is None:
            continue
            
        color = colors[i % len(colors)]
        historical = datasets[sector]
        
        ax = fig.add_subplot(gs[0, i] if i < 2 else gs[1, :])
        
        ax.plot(historical['ds'], historical['y'], 
               color=color, linewidth=2, marker='o', markersize=4, 
               label=f'{sector} Historical')
        
        forecast_label = f'{sector} Forecast (with Vision 2030)' if scenario_type == 'scenario2' else f'{sector} Forecast'
        ax.plot(forecast['ds'], forecast['yhat'], 
               color=color, linewidth=2, linestyle='--', label=forecast_label)
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                       alpha=0.2, color=color, label='Confidence Interval')
        
        ax.axvline(x=pd.to_datetime('2017-01-01'), color='red', linestyle='--', 
                  label='Vision 2030 Start')
        
        title_suffix = f'Scenario 2 Forecast (With Vision 2030)' if scenario_type == 'scenario2' else 'Forecast'
        ax.set_title(f'{sector.upper()} - {title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    
    import os
    results_dir = 'Results/prophet_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        plt.savefig(f"{results_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        try:
            plt.savefig(f"{results_dir}/{filename}", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception:
            pass
    
    return fig

def analyze_forecast_results(forecasts, datasets):
    results_summary = {}
    
    for sector, forecast in forecasts.items():
        if forecast is None:
            continue
            
        historical_data = datasets[sector]
        last_historical_date = historical_data['ds'].max()
        
        historical_forecast = forecast[forecast['ds'] <= last_historical_date]
        future_forecast = forecast[forecast['ds'] > last_historical_date]

        if len(historical_forecast) > 0:
            historical_actual = historical_data['y'].values
            historical_predicted = historical_forecast['yhat'].values

            min_len = min(len(historical_actual), len(historical_predicted))
            historical_actual = historical_actual[:min_len]
            historical_predicted = historical_predicted[:min_len]
            
            mse = mean_squared_error(historical_actual, historical_predicted)
            mae = mean_absolute_error(historical_actual, historical_predicted)
            rmse = np.sqrt(mse)
            
            if np.var(historical_actual) > 0:
                r2 = r2_score(historical_actual, historical_predicted)
            else:
                r2 = np.nan

        if len(future_forecast) > 0:
            future_start = future_forecast['yhat'].iloc[0]
            future_end = future_forecast['yhat'].iloc[-1]
            future_growth = ((future_end - future_start) / future_start) * 100

            forecast_2030 = future_forecast[future_forecast['ds'].dt.year == 2030]
            if len(forecast_2030) > 0:
                avg_2030_value = forecast_2030['yhat'].mean()

        results_summary[sector] = {
            'historical_rmse': rmse if 'rmse' in locals() else None,
            'historical_mae': mae if 'mae' in locals() else None,
            'historical_r2': r2 if 'r2' in locals() else None,
            'future_growth': future_growth if 'future_growth' in locals() else None,
            'avg_2030_value': avg_2030_value if 'avg_2030_value' in locals() else None,
            'forecast_data': future_forecast
        }
    
    return results_summary

def save_results(forecasts, results_summary, datasets, scenario='scenario1'):
    import os
    results_dir = 'Results/prophet_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    for sector, forecast in forecasts.items():
        if forecast is not None:
            filename = f'{results_dir}/{scenario}_forecast_{sector}.csv'
            forecast.to_csv(filename, index=False)
    
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv(f'{results_dir}/{scenario}_summary_results.csv')

    return True

def generate_scenario2_forecasts(fitted_models, datasets, main_df, forecast_end='2030-12-31'):
    """Generate forecasts for Scenario 2 - with Vision 2030 implementation"""
    forecasts = {}
    future_dataframes = {}
    
    last_oil_price = main_df['Oil_Price'].dropna().iloc[-1]
    
    for sector, model in fitted_models.items():
        try:
            last_date = datasets[sector]['ds'].max()
            future_periods = pd.date_range(
                start=last_date + pd.DateOffset(days=1),
                end=forecast_end,
                freq='Q'
            )
            
            all_dates = pd.concat([
                datasets[sector]['ds'],
                pd.Series(future_periods)
            ]).reset_index(drop=True)
            
            future = pd.DataFrame({'ds': all_dates})
            historical_oil_df = datasets[sector][['ds', 'oil_price']].copy()
            n_future = len(future_periods)
            recent_oil = main_df[main_df['DateTime'] >= '2015-01-01']['Oil_Price'].dropna()
            if len(recent_oil) > 4:
                oil_trend = np.polyfit(range(len(recent_oil)), recent_oil, 1)[0]
            else:
                oil_trend = 0
            
            future_oil_base = np.arange(n_future) * oil_trend + last_oil_price
            future_oil_noise = np.random.normal(0, last_oil_price * 0.1, n_future)  
            future_oil = future_oil_base + future_oil_noise
            future_oil_df = pd.DataFrame({'ds': future_periods, 'oil_price': future_oil})
            
            oil_full = pd.concat([historical_oil_df, future_oil_df], ignore_index=True)
            oil_full = oil_full.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')

            future = future.merge(oil_full, on='ds', how='left')
            future['oil_price'] = future['oil_price'].ffill().bfill().fillna(last_oil_price)
            forecast = model.predict(future)
            
            historical_data = datasets[sector]['y']
            historical_range = (historical_data.min(), historical_data.max())
            
            if sector == 'gdp':
                vision_effect = 1.02
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat'] *= vision_effect
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_lower'] *= vision_effect * 0.98
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_upper'] *= vision_effect * 1.02
            elif sector == 'technology':
                vision_effect = 1.05
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat'] *= vision_effect
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_lower'] *= vision_effect * 0.95
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_upper'] *= vision_effect * 1.05
            elif sector == 'tourism':
                vision_effect = 1.08
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat'] *= vision_effect
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_lower'] *= vision_effect * 0.92
                forecast.loc[forecast['ds'] >= '2017-01-01', 'yhat_upper'] *= vision_effect * 1.08
            
            forecasts[sector] = forecast
            future_dataframes[sector] = future
 
        except Exception as e:
            forecasts[sector] = None
            future_dataframes[sector] = None
    
    return forecasts, future_dataframes



def create_scenario_comparison_visualizations(scenario1_forecasts, scenario2_forecasts, main_df):
    sectors = ['gdp', 'technology', 'tourism']
    available_sectors = [s for s in sectors if s in scenario1_forecasts and s in scenario2_forecasts 
                        and scenario1_forecasts[s] is not None and scenario2_forecasts[s] is not None]
    
    if not available_sectors:
        return None
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    colors = {'scenario1': '#1f77b4', 'scenario2': '#ff7f0e', 'actual': '#2ca02c'}
    
    for i, sector in enumerate(available_sectors):
        scenario1_forecast = scenario1_forecasts[sector]
        scenario2_forecast = scenario2_forecasts[sector]
        
        if sector == 'gdp':
            actual_col = 'GDP'
        elif sector == 'technology':
            actual_col = 'Technology_Index'
        else:
            actual_col = 'Tourism_Index'
        
        actual_data = main_df[['DateTime', actual_col]].dropna()
        actual_data = actual_data[['DateTime', actual_col]].rename(columns={'DateTime': 'ds', actual_col: 'y'})
        
        if i < 2:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[1, :])
        
        ax.plot(scenario1_forecast['ds'], scenario1_forecast['yhat'], 
               color=colors['scenario1'], linewidth=2, linestyle='--', 
               label='Scenario 1 (No Vision 2030)')
        ax.plot(scenario2_forecast['ds'], scenario2_forecast['yhat'], 
               color=colors['scenario2'], linewidth=2, linestyle='--', 
               label='Scenario 2 (With Vision 2030)')
        
        ax.plot(actual_data['ds'], actual_data['y'], 
               color=colors['actual'], linewidth=2, marker='o', markersize=4, 
               label='Actual Data')
        
        ax.axvline(x=pd.to_datetime('2017-01-01'), color='red', linestyle='--', 
                  label='Vision 2030 Start')
        ax.set_title(f'{sector.upper()} - Scenario Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Comprehensive Scenario Comparison: Vision 2030 Impact Analysis", y=1.02, fontsize=16)
    
    import os
    results_dir = 'Results/prophet_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        plt.savefig(f"{results_dir}/scenario_comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        try:
            plt.savefig(f"{results_dir}/scenario_comparison_plots.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e2:
            pass
    
    return fig

def calculate_vision_2030_impact(scenario1_forecasts, scenario2_forecasts, main_df):
    impact_analysis = {}
    
    for sector in ['gdp', 'technology', 'tourism']:
        if (sector in scenario1_forecasts and sector in scenario2_forecasts and 
            scenario1_forecasts[sector] is not None and scenario2_forecasts[sector] is not None):
            
            sc1_forecast = scenario1_forecasts[sector]
            sc2_forecast = scenario2_forecasts[sector]
            
            merged = sc1_forecast.merge(sc2_forecast[['ds', 'yhat']], 
                                      on='ds', suffixes=('_sc1', '_sc2'))
            
            post_vision = merged[merged['ds'] >= '2017-01-01']
            
            if len(post_vision) > 0:
                avg_impact = (post_vision['yhat_sc2'] - post_vision['yhat_sc1']).mean()
                avg_percentage_impact = ((post_vision['yhat_sc2'] - post_vision['yhat_sc1']) / 
                                       post_vision['yhat_sc1'] * 100).mean()
                
                forecast_2030 = merged[merged['ds'].dt.year == 2030]
                if len(forecast_2030) > 0:
                    impact_2030 = (forecast_2030['yhat_sc2'].mean() - forecast_2030['yhat_sc1'].mean())
                    percentage_impact_2030 = ((forecast_2030['yhat_sc2'].mean() - forecast_2030['yhat_sc1'].mean()) / 
                                            forecast_2030['yhat_sc1'].mean() * 100)
                else:
                    impact_2030 = None
                    percentage_impact_2030 = None
                
                impact_analysis[sector] = {
                    'avg_impact': avg_impact,
                    'avg_percentage_impact': avg_percentage_impact,
                    'impact_2030': impact_2030,
                    'percentage_impact_2030': percentage_impact_2030,
                    'total_periods_analyzed': len(post_vision)
                }
    
    return impact_analysis

def save_comparison_results(scenario1_results, scenario2_results, impact_analysis):
    import os
    import json
    results_dir = 'Results/prophet_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    impact_df = pd.DataFrame(impact_analysis).T
    impact_df.to_csv(f'{results_dir}/vision_2030_impact_analysis.csv')
    comparison_report = {
        'scenario1_summary': scenario1_results['results_summary'],
        'scenario2_summary': scenario2_results['results_summary'],
        'vision_2030_impact': impact_analysis,
        'analysis_metadata': {
            'scenario1_description': 'Counterfactual forecast without Vision 2030 implementation',
            'scenario2_description': 'Forecast with Vision 2030 implementation effects',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vision_2030_start_date': '2017-01-01',
            'forecast_end_date': '2030-12-31'
        }
    }
    
    with open(f'{results_dir}/comprehensive_comparison_report.json', 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    return True

def run_scenario(main_df, scenario_type='scenario1'):
    cutoff_date = '2016-12-31' if scenario_type == 'scenario1' else None
    datasets, data = prepare_prophet_data(main_df, cutoff_date)
    models, fitted_models = create_enhanced_prophet_models(datasets, scenario_type)

    if not fitted_models:
        return None

    cv_results, performance_results = perform_cross_validation(fitted_models, datasets)
    
    if scenario_type == 'scenario1':
        forecasts, future_dataframes = generate_counterfactual_forecasts(fitted_models, datasets, main_df)
        title = "Counterfactual Forecasts - Scenario 1 (No Vision 2030)"
        filename = "prophet_forecasting_scenario1.png"
    else:
        forecasts, future_dataframes = generate_scenario2_forecasts(fitted_models, datasets, main_df)
        title = "Forecasts - Scenario 2 (With Vision 2030 Implementation)"
        filename = "prophet_forecasting_scenario2.png"
    
    if any(f is not None for f in forecasts.values()):
        try:
            forecast_fig = create_forecast_plots(datasets, forecasts, title, filename, scenario_type)
        except Exception:
            forecast_fig = None
    
    results_summary = analyze_forecast_results(forecasts, datasets)
    save_results(forecasts, results_summary, datasets, scenario_type)

    return {
        'datasets': datasets,
        'models': models,
        'fitted_models': fitted_models,
        'forecasts': forecasts,
        'results_summary': results_summary,
        'cv_results': cv_results,
        'performance_results': performance_results
    }

def run_comprehensive_analysis(main_df):
    scenario1_results = run_scenario(main_df, 'scenario1')
    scenario2_results = run_scenario(main_df, 'scenario2')
    
    if scenario1_results is None or scenario2_results is None:
        return None
    
    comparison_fig = create_scenario_comparison_visualizations(
        scenario1_results['forecasts'], 
        scenario2_results['forecasts'], 
        main_df
    )
    
    impact_analysis = calculate_vision_2030_impact(
        scenario1_results['forecasts'], 
        scenario2_results['forecasts'], 
        main_df
    )
    
    save_comparison_results(scenario1_results, scenario2_results, impact_analysis)
    
    return {
        'scenario1_results': scenario1_results,
        'scenario2_results': scenario2_results,
        'impact_analysis': impact_analysis,
        'comparison_fig': comparison_fig
    }

if __name__ == "__main__":
    exec(open('data_loading_and_EDA.py').read())
    comprehensive_results = run_comprehensive_analysis(main_df)
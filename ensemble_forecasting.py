import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import minimize
import json

def load_all_forecasts(scenario='scenario1'):
    forecasts = {}
    prophet_forecasts = {}
    
    try:
        prophet_forecasts['gdp'] = pd.read_csv(f'Results/prophet_forecasting/{scenario}_forecast_gdp.csv')
        prophet_forecasts['gdp']['ds'] = pd.to_datetime(prophet_forecasts['gdp']['ds'])
    except:
        pass
    
    try:
        prophet_forecasts['technology'] = pd.read_csv(f'Results/prophet_forecasting/{scenario}_forecast_technology.csv')
        prophet_forecasts['technology']['ds'] = pd.to_datetime(prophet_forecasts['technology']['ds'])
    except:
        pass
    
    try:
        prophet_forecasts['tourism'] = pd.read_csv(f'Results/prophet_forecasting/{scenario}_forecast_tourism.csv')
        prophet_forecasts['tourism']['ds'] = pd.to_datetime(prophet_forecasts['tourism']['ds'])
    except:
        pass
    
    if prophet_forecasts:
        forecasts['prophet'] = prophet_forecasts
    
    alternative_models = ['arima', 'ets', 'linear']
    
    for model_name in alternative_models:
        model_forecasts = {}
        for sector in ['gdp', 'technology', 'tourism']:
            try:
                filename = f'Results/alternative_forecasting/{scenario}_forecast_{sector}_{model_name}.csv'
                model_forecasts[sector] = pd.read_csv(filename)
                model_forecasts[sector]['ds'] = pd.to_datetime(model_forecasts[sector]['ds'])
            except:
                continue
        
        if model_forecasts:
            forecasts[model_name] = model_forecasts
    
    return forecasts

def calculate_dynamic_ensemble_weights(validation_results):
    dynamic_weights = {}
    
    for sector in ['gdp', 'technology', 'tourism']:
        sector_weights = {}
        total_performance = 0
        
        for model_name, model_validation in validation_results.items():
            if sector in model_validation:
                rmse = model_validation[sector]['RMSE']
                r2 = model_validation[sector]['R2']
                mape = model_validation[sector]['MAPE']
                
                rmse_normalized = 1 / (1 + rmse) if rmse > 0 else 0.001
                r2_normalized = max(r2, 0) if not np.isnan(r2) else 0.001
                mape_normalized = 1 / (1 + mape) if mape > 0 else 0.001
                
                performance = (r2_normalized * 0.4) + (rmse_normalized * 0.3) + (mape_normalized * 0.3)
                sector_weights[model_name] = max(performance, 0.001)
                total_performance += sector_weights[model_name]
        
        if total_performance > 0:
            for model_name in sector_weights:
                sector_weights[model_name] /= total_performance
        
        dynamic_weights[sector] = sector_weights
    
    return dynamic_weights

def calculate_model_uncertainty(forecasts, ensemble_forecasts):
    uncertainty_metrics = {}
    
    for sector in ensemble_forecasts.keys():
        predictions = []
        dates = ensemble_forecasts[sector]['ds']
        
        for model_name, model_forecasts in forecasts.items():
            if sector in model_forecasts:
                model_df = model_forecasts[sector]
                interp_pred = np.interp(
                    dates.astype(np.int64),
                    model_df['ds'].astype(np.int64),
                    model_df['yhat']
                )
                predictions.append(interp_pred)
        
        if len(predictions) > 1:
            predictions_array = np.array(predictions)
            std_across_models = np.std(predictions_array, axis=0)
            mean_disagreement = np.mean(std_across_models)
            max_disagreement = np.max(std_across_models)
            
            uncertainty_metrics[sector] = {
                'mean_model_disagreement': mean_disagreement,
                'max_model_disagreement': max_disagreement,
                'disagreement_coefficient': mean_disagreement / np.mean(ensemble_forecasts[sector]['yhat'])
            }
    
    return uncertainty_metrics

def calculate_ensemble_forecast_advanced(forecasts, validation_results):
    ensemble_forecasts = {}
    dynamic_weights = calculate_dynamic_ensemble_weights(validation_results)
    
    all_sectors = set()
    for model_name, model_forecasts in forecasts.items():
        all_sectors.update(model_forecasts.keys())
    
    for sector in all_sectors:
        sector_forecasts = {}
        
        for model_name, model_forecasts in forecasts.items():
            if sector in model_forecasts:
                sector_forecasts[model_name] = model_forecasts[sector]
        
        if len(sector_forecasts) < 2:
            continue
            
        min_date = max([df['ds'].min() for df in sector_forecasts.values()])
        max_date = min([df['ds'].max() for df in sector_forecasts.values()])
        
        common_dates = pd.date_range(start=min_date, end=max_date, freq='Q')
        model_predictions = {}
        
        for model_name, forecast_df in sector_forecasts.items():
            forecast_interp = np.interp(
                common_dates.astype(np.int64),
                forecast_df['ds'].astype(np.int64),
                forecast_df['yhat']
            )
            model_predictions[model_name] = forecast_interp
        
        sector_weights = dynamic_weights.get(sector, {})
        ensemble_prediction = np.zeros(len(common_dates))
        
        for model_name, prediction in model_predictions.items():
            weight = sector_weights.get(model_name, 1.0 / len(model_predictions))
            ensemble_prediction += weight * prediction
        
        ensemble_lower = np.zeros(len(common_dates))
        ensemble_upper = np.zeros(len(common_dates))
        
        for model_name, forecast_df in sector_forecasts.items():
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                lower_interp = np.interp(
                    common_dates.astype(np.int64),
                    forecast_df['ds'].astype(np.int64),
                    forecast_df['yhat_lower']
                )
                upper_interp = np.interp(
                    common_dates.astype(np.int64),
                    forecast_df['ds'].astype(np.int64),
                    forecast_df['yhat_upper']
                )
                
                weight = sector_weights.get(model_name, 1.0 / len(sector_forecasts))
                ensemble_lower += weight * lower_interp
                ensemble_upper += weight * upper_interp
        
        predictions_array = np.array(list(model_predictions.values()))
        uncertainty_band = np.std(predictions_array, axis=0) * 1.96
        
        ensemble_df = pd.DataFrame({
            'ds': common_dates,
            'yhat': ensemble_prediction,
            'yhat_lower': np.minimum(ensemble_lower, ensemble_prediction - uncertainty_band),
            'yhat_upper': np.maximum(ensemble_upper, ensemble_prediction + uncertainty_band),
            'model_uncertainty': uncertainty_band
        })
        
        ensemble_forecasts[sector] = ensemble_df
    
    return ensemble_forecasts

def perform_enhanced_validation(main_df, forecasts):
    validation_results = {}
    validation_data = main_df[
        (main_df['DateTime'] >= '2017-01-01') & 
        (main_df['DateTime'] <= '2024-12-31')
    ].copy()
    
    if len(validation_data) == 0:
        return {}
    
    for model_name, model_forecasts in forecasts.items():
        model_validation = {}
        
        for sector in model_forecasts.keys():
            try:
                if sector == 'gdp':
                    actual_values = validation_data[['DateTime', 'GDP']].dropna()
                    actual_col = 'GDP'
                elif sector == 'technology':
                    actual_values = validation_data[['DateTime', 'Technology_Index']].dropna()
                    actual_col = 'Technology_Index'
                elif sector == 'tourism':
                    actual_values = validation_data[['DateTime', 'Tourism_Index']].dropna()
                    actual_col = 'Tourism_Index'
                else:
                    continue
                
                if len(actual_values) == 0:
                    continue
                    
                forecast_data = model_forecasts[sector]
                validation_forecast = forecast_data[
                    (forecast_data['ds'] >= validation_data['DateTime'].min()) &
                    (forecast_data['ds'] <= validation_data['DateTime'].max())
                ]
                
                if len(validation_forecast) == 0:
                    continue
                
                merged_data = pd.merge(
                    actual_values.rename(columns={'DateTime': 'ds', actual_col: 'actual'}),
                    validation_forecast[['ds', 'yhat']].rename(columns={'yhat': 'predicted'}),
                    on='ds',
                    how='inner'
                )
                
                if len(merged_data) < 3:
                    continue
                
                actual = merged_data['actual'].values
                predicted = merged_data['predicted'].values
                
                mse = mean_squared_error(actual, predicted)
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mse)
                
                if np.var(actual) > 0:
                    r2 = r2_score(actual, predicted)
                else:
                    r2 = np.nan
                
                mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
                
                if len(actual) > 1:
                    actual_direction = np.sign(np.diff(actual))
                    predicted_direction = np.sign(np.diff(predicted))
                    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
                else:
                    directional_accuracy = np.nan
                
                residuals = actual - predicted
                ljung_box_stat, ljung_box_p = stats.jarque_bera(residuals)
                
                model_validation[sector] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'Directional_Accuracy': directional_accuracy,
                    'N_Points': len(merged_data),
                    'Residual_Normality_P': ljung_box_p,
                    'Mean_Absolute_Residual': np.mean(np.abs(residuals)),
                    'Merged_Data': merged_data
                }
                
            except Exception as e:
                continue
        
        validation_results[model_name] = model_validation
    
    return validation_results

def select_optimal_models(validation_results):
    best_models = {}
    all_sectors = set()
    for model_validation in validation_results.values():
        all_sectors.update(model_validation.keys())
    
    for sector in all_sectors:
        sector_performance = {}
        
        for model_name, model_validation in validation_results.items():
            if sector in model_validation:
                metrics = model_validation[sector]
                
                rmse_score = -np.log(1 + metrics['RMSE']) * 0.25
                r2_score_val = metrics['R2'] * 0.25 if not np.isnan(metrics['R2']) else 0
                mape_score = -np.log(1 + metrics['MAPE']) * 0.2
                direction_score = metrics['Directional_Accuracy'] / 100 * 0.2 if not np.isnan(metrics['Directional_Accuracy']) else 0
                stability_score = (1 / (1 + metrics['Mean_Absolute_Residual'])) * 0.1
                
                composite_score = rmse_score + r2_score_val + mape_score + direction_score + stability_score
                
                sector_performance[model_name] = {
                    'composite_score': composite_score,
                    'metrics': metrics
                }
        
        if not sector_performance:
            continue

        best_model_name = max(sector_performance.keys(), 
                            key=lambda x: sector_performance[x]['composite_score'])
        
        best_models[sector] = {
            'model': best_model_name,
            'score': sector_performance[best_model_name]['composite_score'],
            'metrics': sector_performance[best_model_name]['metrics'],
            'all_scores': sector_performance
        }
    
    return best_models

def create_final_forecasts(forecasts, best_models, ensemble_forecasts, scenario='scenario1'):
    final_forecasts = {}
    
    for sector, best_info in best_models.items():
        best_model = best_info['model']
        
        if sector in ensemble_forecasts:
            final_forecasts[sector] = ensemble_forecasts[sector].copy()
            final_forecasts[sector]['model_used'] = 'ensemble'
        elif best_model in forecasts and sector in forecasts[best_model]:
            final_forecasts[sector] = forecasts[best_model][sector].copy()
            final_forecasts[sector]['model_used'] = best_model
        else:
            continue
        
        final_forecasts[sector]['sector'] = sector
        final_forecasts[sector]['scenario'] = scenario
        final_forecasts[sector]['validation_r2'] = best_info['metrics']['R2']
        final_forecasts[sector]['validation_rmse'] = best_info['metrics']['RMSE']
        final_forecasts[sector]['composite_score'] = best_info['score']
    
    return final_forecasts

def create_enhanced_all_models_visualization(forecasts, main_df, ensemble_forecasts, scenario='scenario1'):
    if not forecasts:
        return None
    
    scenario_title = "Scenario 1 (No Vision 2030)" if scenario == 'scenario1' else "Scenario 2 (With Vision 2030)"
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f'Enhanced All-Models Forecasting Analysis - {scenario_title} (2017-2030)', fontsize=18, fontweight='bold')
    
    sectors = ['gdp', 'technology', 'tourism']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_colors = {model: colors[i] for i, model in enumerate(forecasts.keys())}
    
    for i, sector in enumerate(sectors):
        if sector == 'gdp':
            historical = main_df[['DateTime', 'GDP']].dropna()
            col_name = 'GDP'
            ylabel = 'GDP (Million USD)'
        elif sector == 'technology':
            historical = main_df[['DateTime', 'Technology_Index']].dropna()
            col_name = 'Technology_Index'
            ylabel = 'Technology Index'
        else:
            historical = main_df[['DateTime', 'Tourism_Index']].dropna()
            col_name = 'Tourism_Index'
            ylabel = 'Tourism Index'
        
        ax1 = axes[0, i]
        ax1.plot(historical['DateTime'], historical[col_name], 'k-', linewidth=3, label='Historical Data', alpha=0.8)
        
        for model_name, model_forecasts in forecasts.items():
            if sector in model_forecasts:
                forecast_df = model_forecasts[sector]
                ax1.plot(forecast_df['ds'], forecast_df['yhat'], '--', 
                        color=model_colors[model_name], alpha=0.8, linewidth=2, 
                        label=f'{model_name.upper()}')
        
        if sector in ensemble_forecasts:
            ensemble_df = ensemble_forecasts[sector]
            ax1.plot(ensemble_df['ds'], ensemble_df['yhat'], 'r-', linewidth=4, label='ENSEMBLE')
            
            if 'model_uncertainty' in ensemble_df.columns:
                uncertainty_upper = ensemble_df['yhat'] + ensemble_df['model_uncertainty']
                uncertainty_lower = ensemble_df['yhat'] - ensemble_df['model_uncertainty']
                ax1.fill_between(ensemble_df['ds'], uncertainty_lower, uncertainty_upper, 
                               alpha=0.2, color='red', label='Model Uncertainty')
        
        ax1.set_title(f'{sector.upper()} - All Model Forecasts (2017-2030)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.axvline(x=pd.to_datetime('2017-01-01'), color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        ax2 = axes[1, i]
        model_names = []
        model_values = []
        model_colors_plot = []
        
        for model_name, model_forecasts in forecasts.items():
            if sector in model_forecasts:
                forecast_df = model_forecasts[sector]
                forecast_2030 = forecast_df[forecast_df['ds'].dt.year == 2030]
                if len(forecast_2030) > 0:
                    model_names.append(model_name.upper())
                    model_values.append(forecast_2030['yhat'].mean())
                    model_colors_plot.append(model_colors[model_name])
        
        if sector in ensemble_forecasts:
            ensemble_df = ensemble_forecasts[sector]
            ensemble_2030 = ensemble_df[ensemble_df['ds'].dt.year == 2030]
            if len(ensemble_2030) > 0:
                model_names.append('ENSEMBLE')
                model_values.append(ensemble_2030['yhat'].mean())
                model_colors_plot.append('red')
        
        if model_names:
            bars = ax2.bar(model_names, model_values, color=model_colors_plot, alpha=0.8)
            ax2.set_title(f'{sector.upper()} - 2030 Forecast Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel(ylabel, fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, model_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    import os
    results_dir = 'Results/ensemble_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/enhanced_all_models_forecast_{scenario}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def generate_summary_report(final_forecasts, validation_results, best_models, uncertainty_metrics, scenario='scenario1'):
    if not final_forecasts:
        return {
            'summary_statistics': {}, '2030_projections': {}, 'model_performance': {},
            'growth_rates': {}, 'uncertainty_analysis': {}, 'scenario': scenario,
            'error': 'No forecasts available'
        }
    
    report = {
        'summary_statistics': {}, '2030_projections': {}, 'model_performance': {},
        'growth_rates': {}, 'uncertainty_analysis': uncertainty_metrics, 'scenario': scenario
    }
    
    for sector, forecast in final_forecasts.items():
        forecast_2030 = forecast[forecast['ds'].dt.year == 2030]
        if len(forecast_2030) > 0:
            avg_2030 = forecast_2030['yhat'].mean()
            min_2030 = forecast_2030['yhat'].min()
            max_2030 = forecast_2030['yhat'].max()
            
            if 'yhat_lower' in forecast_2030.columns and 'yhat_upper' in forecast_2030.columns:
                ci_lower_2030 = forecast_2030['yhat_lower'].mean()
                ci_upper_2030 = forecast_2030['yhat_upper'].mean()
            else:
                ci_lower_2030 = min_2030
                ci_upper_2030 = max_2030
            
            report['2030_projections'][sector] = {
                'average': avg_2030, 'min': min_2030, 'max': max_2030,
                'ci_lower': ci_lower_2030, 'ci_upper': ci_upper_2030
            }

        forecast_2017 = forecast[forecast['ds'].dt.year == 2017]
        if len(forecast_2017) > 0 and len(forecast_2030) > 0:
            start_value = forecast_2017['yhat'].mean()
            end_value = avg_2030
            total_growth = ((end_value - start_value) / start_value) * 100
            annual_growth = ((end_value / start_value) ** (1/13)) - 1
            
            report['growth_rates'][sector] = {
                'total_growth_2017_2030': total_growth,
                'annual_growth_rate': annual_growth * 100,
                'start_value_2017': start_value,
                'end_value_2030': end_value
            }

        if sector in best_models:
            model_info = best_models[sector]
            report['model_performance'][sector] = {
                'best_model': model_info['model'],
                'validation_metrics': model_info['metrics'],
                'composite_score': model_info['score'],
                'all_model_scores': {k: v['composite_score'] for k, v in model_info.get('all_scores', {}).items()}
            }

        report['summary_statistics'][sector] = {
            'forecast_mean': forecast['yhat'].mean(),
            'forecast_std': forecast['yhat'].std(),
            'forecast_min': forecast['yhat'].min(),
            'forecast_max': forecast['yhat'].max(),
            'total_periods': len(forecast),
            'model_used': forecast.get('model_used', ['unknown'])[0] if isinstance(forecast.get('model_used'), list) else forecast.get('model_used', 'unknown')
        }
    
    return report

def save_final_results(final_forecasts, validation_results, report, scenario='scenario1'):
    import os
    results_dir = 'Results/ensemble_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    if not final_forecasts:
        return False
    
    for sector, forecast in final_forecasts.items():
        filename = f'{results_dir}/final_{scenario}_forecast_{sector}.csv'
        forecast.to_csv(filename, index=False)

    validation_summary = []
    for model_name, model_validation in validation_results.items():
        for sector, metrics in model_validation.items():
            validation_summary.append({
                'model': model_name, 'sector': sector, 'scenario': scenario,
                **{k: v for k, v in metrics.items() if k != 'Merged_Data'}
            })
    
    validation_df = pd.DataFrame(validation_summary)
    validation_df.to_csv(f'{results_dir}/{scenario}_validation_results.csv', index=False)

    with open(f'{results_dir}/{scenario}_summary_report.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                df_copy = obj.copy()
                for col in df_copy.columns:
                    if df_copy[col].dtype == 'datetime64[ns]' or 'datetime' in str(df_copy[col].dtype):
                        df_copy[col] = df_copy[col].astype(str)
                if df_copy.index.dtype == 'datetime64[ns]' or 'datetime' in str(df_copy.index.dtype):
                    df_copy.index = df_copy.index.astype(str)
                df_dict = df_copy.to_dict('records')
                return convert_types(df_dict)
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'dtype') and 'datetime' in str(obj.dtype):
                return obj.astype(str).tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(report), f, indent=2)
    
    return True

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

def run_single_scenario(main_df, scenario='scenario1'):
    forecasts = load_all_forecasts(scenario)
    
    if not forecasts:
        return None
    
    validation_results = perform_enhanced_validation(main_df, forecasts)
    if not validation_results:
        return None
    
    ensemble_forecasts = calculate_ensemble_forecast_advanced(forecasts, validation_results)
    uncertainty_metrics = calculate_model_uncertainty(forecasts, ensemble_forecasts)
    best_models = select_optimal_models(validation_results)
    final_forecasts = create_final_forecasts(forecasts, best_models, ensemble_forecasts, scenario)
    
    if not final_forecasts:
        return None
    
    enhanced_fig = create_enhanced_all_models_visualization(forecasts, main_df, ensemble_forecasts, scenario)
    report = generate_summary_report(final_forecasts, validation_results, best_models, uncertainty_metrics, scenario)
    save_final_results(final_forecasts, validation_results, report, scenario)
    
    return {
        'forecasts': forecasts, 'ensemble_forecasts': ensemble_forecasts,
        'validation_results': validation_results, 'best_models': best_models,
        'final_forecasts': final_forecasts, 'report': report,
        'uncertainty_metrics': uncertainty_metrics
    }

def run_comprehensive_ensemble_analysis(main_df):
    scenario1_results = run_single_scenario(main_df, 'scenario1')
    scenario2_results = run_single_scenario(main_df, 'scenario2')
    
    if scenario1_results is None or scenario2_results is None:
        return None
    
    impact_analysis = calculate_vision_2030_impact(
        scenario1_results['final_forecasts'], 
        scenario2_results['final_forecasts'], 
        main_df
    )
    
    import os
    results_dir = 'Results/ensemble_forecasting'
    os.makedirs(results_dir, exist_ok=True)
    
    impact_df = pd.DataFrame(impact_analysis).T
    impact_df.to_csv(f'{results_dir}/vision_2030_impact_analysis.csv')
    
    comprehensive_report = {
        'scenario1_summary': scenario1_results['report'],
        'scenario2_summary': scenario2_results['report'],
        'vision_2030_impact': impact_analysis,
        'analysis_metadata': {
            'scenario1_description': 'Counterfactual forecast without Vision 2030 implementation',
            'scenario2_description': 'Forecast with Vision 2030 implementation effects',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vision_2030_start_date': '2017-01-01',
            'forecast_end_date': '2030-12-31',
            'ensemble_method': 'Performance-weighted combination of Prophet, ARIMA, ETS, and Linear models'
        }
    }
    
    with open(f'{results_dir}/comprehensive_ensemble_report.json', 'w') as f:
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                df_copy = obj.copy()
                for col in df_copy.columns:
                    if df_copy[col].dtype == 'datetime64[ns]' or 'datetime' in str(df_copy[col].dtype):
                        df_copy[col] = df_copy[col].astype(str)
                if df_copy.index.dtype == 'datetime64[ns]' or 'datetime' in str(df_copy.index.dtype):
                    df_copy.index = df_copy.index.astype(str)
                return df_copy.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, 'dtype') and 'datetime' in str(obj.dtype):
                return obj.astype(str).tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(comprehensive_report), f, indent=2)
    
    return {
        'scenario1_results': scenario1_results,
        'scenario2_results': scenario2_results,
        'impact_analysis': impact_analysis,
        'comprehensive_report': comprehensive_report
    }

def run_ensemble_forecasting(main_df):
    return run_single_scenario(main_df, 'scenario1')

if __name__ == "__main__":
    exec(open('data_loading_and_EDA.py').read())
    comprehensive_results = run_comprehensive_ensemble_analysis(main_df)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configuration
CONFIG = {
    'vision_2030_start': '2017-01-01',
    'covid_start': '2020-01-01',
    'covid_end': '2022-12-31',
    'min_data_points': 20,
    'sectors': ['GDP', 'Technology_Index', 'Tourism_Index'],
    'results_dir': 'Results/econometric_validation'
}



def load_json_safe(filepath, default=None):
    """Safely load JSON file with fallback"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default

def load_enhanced_data():
    enhanced_data = {}
    
    try:
        enhanced_data['main'] = pd.read_csv('enhanced_final_2030_comparison.csv')
        enhanced_data['main']['DateTime'] = pd.to_datetime(enhanced_data['main']['DateTime'])
    except:
        pass
    
    try:
        enhanced_data['validation'] = pd.read_csv('enhanced_data_validation_results.json')
    except:
        pass
    
    try:
        enhanced_data['preprocessing'] = pd.read_csv('enhanced_preprocessing_summary.json')
    except:
        pass
    
    return enhanced_data

def prepare_advanced_analysis_data(main_df):
    analysis_data = main_df.copy()
    analysis_data['DateTime'] = pd.to_datetime(analysis_data['DateTime'])
    analysis_data = analysis_data.sort_values('DateTime')
    
    analysis_data['Post_Vision'] = (analysis_data['DateTime'] >= CONFIG['vision_2030_start']).astype(int)
    analysis_data['COVID_Period'] = (
        (analysis_data['DateTime'] >= CONFIG['covid_start']) & 
        (analysis_data['DateTime'] <= CONFIG['covid_end'])
    ).astype(int)
    
    analysis_data['Time_Trend'] = range(len(analysis_data))
    analysis_data['Quarter'] = analysis_data['DateTime'].dt.quarter
    analysis_data['Year'] = analysis_data['DateTime'].dt.year
    
    sector_data = {}
    required_cols = ['DateTime', 'Post_Vision', 'COVID_Period', 'Time_Trend', 'Quarter', 'Year']
    
    for sector in CONFIG['sectors']:
        if sector not in analysis_data.columns:
            continue
        
        sector_df = analysis_data[required_cols + [sector]].dropna()
        
        if len(sector_df) >= CONFIG['min_data_points']:
            sector_data[sector.lower().replace('_index', '')] = {
                'data': sector_df,
                'target_col': sector,
                'sector_name': sector
            }
    
    return analysis_data, sector_data

def run_structural_break_analysis(data, target_col):
    try:
        analysis_data = data.copy()
        analysis_data = analysis_data.set_index('DateTime')
        
        target_series = analysis_data[target_col].dropna()
        
        if len(target_series) < CONFIG['min_data_points']:
            return None
        
        from statsmodels.tsa.stattools import cusum_squares_test
        from statsmodels.tsa.regime_switching import MarkovRegression
        
        cusum_test = cusum_squares_test(target_series)
        structural_breaks = []
        
        if cusum_test.pvalue < 0.05:
            structural_breaks.append({
                'test': 'CUSUM_Squares',
                'p_value': cusum_test.pvalue,
                'significant': True,
                'break_date': 'Detected'
            })
        
        try:
            markov_model = MarkovRegression(target_series, k_regimes=2)
            markov_results = markov_model.fit(disp=False)
            
            if markov_results.aic < 0:
                structural_breaks.append({
                    'test': 'Markov_Regime_Switching',
                    'aic': markov_results.aic,
                    'regimes': 2,
                    'significant': True
                })
        except:
            pass
        
        return {
            'target_variable': target_col,
            'structural_breaks': structural_breaks,
            'cusum_test': cusum_test,
            'data_points': len(target_series)
        }
        
    except Exception:
        return None

def run_granger_causality_analysis(data, target_col):
    try:
        analysis_data = data.copy()
        analysis_data = analysis_data.set_index('DateTime')
        
        target_series = analysis_data[target_col].dropna()
        
        if len(target_series) < 20:
            return None
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        control_variables = ['Oil_Price', 'Time_Trend']
        available_controls = [col for col in control_variables if col in analysis_data.columns]
        
        granger_results = {}
        
        for control_var in available_controls:
            control_series = analysis_data[control_var].dropna()
            
            if len(control_series) >= len(target_series) * 0.8:
                combined_data = pd.concat([target_series, control_series], axis=1).dropna()
                
                if len(combined_data) >= 20:
                    try:
                        gc_result = grangercausalitytests(combined_data, maxlag=4, verbose=False)
                        
                        best_lag = None
                        best_p_value = 1.0
                        
                        for lag, result in gc_result.items():
                            p_value = result[0]['ssr_chi2test'][1]
                            if p_value < best_p_value:
                                best_p_value = p_value
                                best_lag = lag
                        
                        granger_results[control_var] = {
                            'best_lag': best_lag,
                            'best_p_value': best_p_value,
                            'significant': best_p_value < 0.05,
                            'causality_direction': f'{control_var} -> {target_col}'
                        }
                    except:
                        pass
        
        return {
            'target_variable': target_col,
            'granger_results': granger_results,
            'data_points': len(target_series)
        }
        
    except Exception as e:
        return None

def run_cointegration_analysis(data, target_col):
    try:
        analysis_data = data.copy()
        analysis_data = analysis_data.set_index('DateTime')
        
        target_series = analysis_data[target_col].dropna()
        
        if len(target_series) < 20:
            return None
        
        from statsmodels.tsa.stattools import coint
        
        control_variables = ['Oil_Price', 'Time_Trend']
        available_controls = [col for col in control_variables if col in analysis_data.columns]
        
        cointegration_results = {}
        
        for control_var in available_controls:
            control_series = analysis_data[control_var].dropna()
            
            if len(control_series) >= len(target_series) * 0.8:
                combined_data = pd.concat([target_series, control_series], axis=1).dropna()
                
                if len(combined_data) >= 20:
                    try:
                        score, p_value, _ = coint(combined_data.iloc[:, 0], combined_data.iloc[:, 1])
                        
                        cointegration_results[control_var] = {
                            'score': score,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'cointegrated': p_value < 0.05
                        }
                    except:
                        pass
        
        return {
            'target_variable': target_col,
            'cointegration_results': cointegration_results,
            'data_points': len(target_series)
        }
        
    except Exception as e:
        return None

def run_volatility_analysis(data, target_col):
    try:
        analysis_data = data.copy()
        analysis_data = analysis_data.set_index('DateTime')
        
        target_series = analysis_data[target_col].dropna()
        
        if len(target_series) < 20:
            return None
        
        returns = target_series.pct_change().dropna()
        
        if len(returns) < 10:
            return None
        
        volatility_pre = returns[returns.index < '2017-01-01'].std()
        volatility_post = returns[returns.index >= '2017-01-01'].std()
        
        volatility_change = ((volatility_post - volatility_pre) / volatility_pre) * 100 if volatility_pre != 0 else 0
        
        from scipy import stats
        f_stat, p_value = stats.f_oneway(
            returns[returns.index < '2017-01-01'],
            returns[returns.index >= '2017-01-01']
        )
        
        return {
            'target_variable': target_col,
            'volatility_pre_vision': volatility_pre,
            'volatility_post_vision': volatility_post,
            'volatility_change_percent': volatility_change,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'data_points': len(returns)
        }
        
    except Exception as e:
        return None

def run_regime_switching_analysis(data, target_col):
    try:
        analysis_data = data.copy()
        analysis_data = analysis_data.set_index('DateTime')
        
        target_series = analysis_data[target_col].dropna()
        
        if len(target_series) < 30:
            return None
        
        from statsmodels.tsa.regime_switching import MarkovRegression
        
        try:
            markov_model = MarkovRegression(target_series, k_regimes=2)
            markov_results = markov_model.fit(disp=False)
            
            regime_probabilities = markov_results.smoothed_marginal_probabilities
            
            pre_vision_regime = regime_probabilities[regime_probabilities.index < '2017-01-01'].mean()
            post_vision_regime = regime_probabilities[regime_probabilities.index >= '2017-01-01'].mean()
            
            regime_change = post_vision_regime - pre_vision_regime
            
            return {
                'target_variable': target_col,
                'aic': markov_results.aic,
                'bic': markov_results.bic,
                'regime_probabilities': regime_probabilities,
                'pre_vision_regime': pre_vision_regime,
                'post_vision_regime': post_vision_regime,
                'regime_change': regime_change,
                'data_points': len(target_series)
            }
            
        except:
            return None
        
    except Exception as e:
        return None

def run_advanced_analysis(sector_data):
    advanced_results = {}
    
    for sector_key, sector_info in sector_data.items():
        data = sector_info['data']
        target_col = sector_info['target_col']
        
        sector_results = {
            'data': data,
            'target_col': target_col,
            'analyses': {}
        }
        
        structural_result = run_structural_break_analysis(data, target_col)
        if structural_result:
            sector_results['analyses']['structural_breaks'] = structural_result
        
        granger_result = run_granger_causality_analysis(data, target_col)
        if granger_result:
            sector_results['analyses']['granger_causality'] = granger_result
        
        cointegration_result = run_cointegration_analysis(data, target_col)
        if cointegration_result:
            sector_results['analyses']['cointegration'] = cointegration_result
        
        volatility_result = run_volatility_analysis(data, target_col)
        if volatility_result:
            sector_results['analyses']['volatility'] = volatility_result
        
        regime_result = run_regime_switching_analysis(data, target_col)
        if regime_result:
            sector_results['analyses']['regime_switching'] = regime_result
        
        if sector_results['analyses']:
            advanced_results[sector_key] = sector_results
    
    return advanced_results







def generate_advanced_analysis_summary(advanced_results):
    summary_report = {
        'overall_findings': {},
        'sector_analyses': {},
        'method_comparison': {}
    }
    
    total_sectors = len(advanced_results)
    sectors_with_breaks = []
    sectors_with_causality = []
    sectors_with_volatility = []
    sectors_with_regimes = []
    
    for sector_key, results in advanced_results.items():
        sector_summary = {
            'analyses_performed': list(results['analyses'].keys()),
            'findings': {}
        }
        
        if 'structural_breaks' in results['analyses']:
            structural_result = results['analyses']['structural_breaks']
            if structural_result['structural_breaks']:
                sectors_with_breaks.append(sector_key)
                sector_summary['findings']['structural_breaks'] = {
                    'significant_breaks': len([b for b in structural_result['structural_breaks'] if b['significant']]),
                    'total_tests': len(structural_result['structural_breaks'])
                }
        
        if 'granger_causality' in results['analyses']:
            granger_result = results['analyses']['granger_causality']
            if granger_result['granger_results']:
                significant_causality = [var for var, result in granger_result['granger_results'].items() if result['significant']]
                if significant_causality:
                    sectors_with_causality.append(sector_key)
                    sector_summary['findings']['granger_causality'] = {
                        'significant_variables': significant_causality,
                        'total_variables': len(granger_result['granger_results'])
                    }
        
        if 'volatility' in results['analyses']:
            volatility_result = results['analyses']['volatility']
            if volatility_result['significant']:
                sectors_with_volatility.append(sector_key)
                sector_summary['findings']['volatility'] = {
                    'volatility_change_percent': volatility_result['volatility_change_percent'],
                    'significant': volatility_result['significant']
                }
        
        if 'regime_switching' in results['analyses']:
            regime_result = results['analyses']['regime_switching']
            sectors_with_regimes.append(sector_key)
            sector_summary['findings']['regime_switching'] = {
                'aic': regime_result['aic'],
                'bic': regime_result['bic'],
                'regime_change': regime_result['regime_change'].tolist() if hasattr(regime_result['regime_change'], 'tolist') else regime_result['regime_change']
            }
        
        summary_report['sector_analyses'][sector_key] = sector_summary
    
    summary_report['overall_findings'] = {
        'total_sectors_analyzed': total_sectors,
        'sectors_with_structural_breaks': len(sectors_with_breaks),
        'sectors_with_granger_causality': len(sectors_with_causality),
        'sectors_with_volatility_changes': len(sectors_with_volatility),
        'sectors_with_regime_switching': len(sectors_with_regimes),
        'significant_sectors_list': {
            'structural_breaks': sectors_with_breaks,
            'granger_causality': sectors_with_causality,
            'volatility': sectors_with_volatility,
            'regime_switching': sectors_with_regimes
        }
    }
    
    return summary_report

def save_advanced_analysis_results(advanced_results, summary_report):
    for sector_key, results in advanced_results.items():
        sector_data = []
        
        for analysis_type, analysis_result in results['analyses'].items():
            if analysis_type == 'structural_breaks':
                for break_info in analysis_result['structural_breaks']:
                    sector_data.append({
                        'sector': sector_key,
                        'analysis_type': analysis_type,
                        'test': break_info['test'],
                        'p_value': break_info.get('p_value', None),
                        'significant': break_info['significant']
                    })
            
            elif analysis_type == 'granger_causality':
                for var, var_result in analysis_result['granger_results'].items():
                    sector_data.append({
                        'sector': sector_key,
                        'analysis_type': analysis_type,
                        'variable': var,
                        'best_lag': var_result['best_lag'],
                        'p_value': var_result['best_p_value'],
                        'significant': var_result['significant']
                    })
            
            elif analysis_type == 'volatility':
                sector_data.append({
                    'sector': sector_key,
                    'analysis_type': analysis_type,
                    'volatility_pre': analysis_result['volatility_pre_vision'],
                    'volatility_post': analysis_result['volatility_post_vision'],
                    'change_percent': analysis_result['volatility_change_percent'],
                    'significant': analysis_result['significant']
                })
            
            elif analysis_type == 'regime_switching':
                sector_data.append({
                    'sector': sector_key,
                    'analysis_type': analysis_type,
                    'aic': analysis_result['aic'],
                    'bic': analysis_result['bic'],
                    'regime_change': analysis_result['regime_change']
                })
        
        if sector_data:
            import os
            results_dir = 'Results/econometric_validation'
            os.makedirs(results_dir, exist_ok=True)
            
            sector_df = pd.DataFrame(sector_data)
            sector_df.to_csv(f'{results_dir}/advanced_analysis_{sector_key}.csv', index=False)
    
    import os
    results_dir = 'Results/econometric_validation'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/advanced_analysis_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    return True

def create_enhanced_visualizations(advanced_results, causal_results, summary_report):
    """
    Create enhanced visualizations for econometric validation and causal impact analysis
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Causal Impact Effect Size Comparison
    ax1 = plt.subplot(2, 3, 1)
    create_causal_effect_comparison(ax1, causal_results)
    
    # 2. Statistical Significance Heatmap
    ax2 = plt.subplot(2, 3, 2)
    create_significance_heatmap(ax2, advanced_results, causal_results)
    
    # 3. Pre-Post Treatment Comparison
    ax3 = plt.subplot(2, 3, 3)
    create_pre_post_comparison(ax3, causal_results)
    
    # 4. Volatility Analysis
    ax4 = plt.subplot(2, 3, 4)
    create_volatility_analysis(ax4, advanced_results)
    
    # 5. Method Performance Comparison
    ax5 = plt.subplot(2, 3, 5)
    create_method_comparison(ax5, causal_results)
    
    # 6. Sector Performance Dashboard
    ax6 = plt.subplot(2, 3, 6)
    create_sector_dashboard(ax6, summary_report)
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    import os
    results_dir = 'Results/econometric_validation'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/enhanced_econometric_visualizations.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return True

def create_individual_visualizations(advanced_results, causal_results, summary_report):
    """
    Create individual focused visualizations for specific insights
    """
    import os
    results_dir = 'Results/econometric_validation'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Causal Impact Effect Size Comparison (Standalone)
    plt.figure(figsize=(12, 8))
    create_causal_effect_comparison(plt.gca(), causal_results)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/causal_effect_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Statistical Significance Heatmap (Standalone)
    plt.figure(figsize=(14, 8))
    create_significance_heatmap(plt.gca(), advanced_results, causal_results)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/statistical_significance_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Pre-Post Treatment Comparison (Standalone)
    plt.figure(figsize=(12, 8))
    create_pre_post_comparison(plt.gca(), causal_results)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/pre_post_treatment_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Individual visualizations created successfully!")
    return True

def create_causal_effect_comparison(ax, causal_results):
    """Create horizontal bar chart comparing causal effects across sectors"""
    if not causal_results:
        ax.text(0.5, 0.5, 'No causal results available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Causal Impact Effect Sizes', fontsize=14, fontweight='bold')
        return
    
    sectors = []
    effects = []
    colors = []
    
    for sector, results in causal_results.items():
        if 'methods' in results and 'difference_in_differences' in results['methods']:
            did_result = results['methods']['difference_in_differences']
            if 'relative_effect_percent' in did_result:
                sectors.append(sector.upper())
                effects.append(did_result['relative_effect_percent'])
                colors.append('green' if did_result['relative_effect_percent'] > 0 else 'red')
    
    if not effects:
        ax.text(0.5, 0.5, 'No DiD results available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Causal Impact Effect Sizes', fontsize=14, fontweight='bold')
        return
    
    bars = ax.barh(sectors, effects, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, effect) in enumerate(zip(bars, effects)):
        width = bar.get_width()
        ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2, 
                f'{effect:.1f}%', ha='left' if width > 0 else 'right', va='center', 
                fontweight='bold', fontsize=10)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Relative Effect (%)', fontsize=12)
    ax.set_title('Vision 2030 Causal Impact Effects\n(Difference-in-Differences)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    positive_patch = mpatches.Patch(color='green', alpha=0.7, label='Positive Effect')
    negative_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative Effect')
    ax.legend(handles=[positive_patch, negative_patch], loc='lower right')

def create_significance_heatmap(ax, advanced_results, causal_results):
    """Create heatmap showing statistical significance across sectors and methods"""
    # Prepare data for heatmap
    sectors = ['GDP', 'Technology', 'Tourism']
    methods = ['Granger Causality', 'Volatility', 'DiD Causal', 'Robust Causal']
    
    # Initialize significance matrix
    significance_matrix = np.zeros((len(sectors), len(methods)))
    p_value_matrix = np.full((len(sectors), len(methods)), np.nan)
    
    # Fill econometric validation results
    for i, sector in enumerate(['gdp', 'technology', 'tourism']):
        if sector in advanced_results:
            # Granger Causality
            if 'granger_causality' in advanced_results[sector]['analyses']:
                granger_result = advanced_results[sector]['analyses']['granger_causality']
                if granger_result['granger_results']:
                    best_p_value = min([result['best_p_value'] for result in granger_result['granger_results'].values()])
                    p_value_matrix[i, 0] = best_p_value
                    significance_matrix[i, 0] = 1 if best_p_value < 0.05 else 0
            
            # Volatility
            if 'volatility' in advanced_results[sector]['analyses']:
                volatility_result = advanced_results[sector]['analyses']['volatility']
                p_value_matrix[i, 1] = volatility_result['p_value']
                significance_matrix[i, 1] = 1 if volatility_result['significant'] else 0
    
    # Fill causal results
    for i, sector in enumerate(['gdp', 'technology', 'tourism']):
        if sector in causal_results:
            # DiD Causal
            if 'methods' in causal_results[sector] and 'difference_in_differences' in causal_results[sector]['methods']:
                did_result = causal_results[sector]['methods']['difference_in_differences']
                p_value_matrix[i, 2] = did_result['p_value']
                significance_matrix[i, 2] = 1 if did_result['significant'] else 0
            
            # Robust Causal
            if 'methods' in causal_results[sector] and 'robust_alternative' in causal_results[sector]['methods']:
                robust_result = causal_results[sector]['methods']['robust_alternative']
                p_value_matrix[i, 3] = robust_result['summary']['p_value']
                significance_matrix[i, 3] = 1 if robust_result['summary']['significant'] else 0
    
    # Create heatmap
    im = ax.imshow(significance_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations
    for i in range(len(sectors)):
        for j in range(len(methods)):
            if not np.isnan(p_value_matrix[i, j]):
                p_val = p_value_matrix[i, j]
                text = f'{p_val:.3f}' if p_val >= 0.001 else f'{p_val:.2e}'
                color = 'white' if significance_matrix[i, j] == 1 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=8, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(sectors)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(sectors)
    ax.set_title('Statistical Significance Heatmap\n(p-values < 0.05 = Significant)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Significant', 'Significant'])

def create_pre_post_comparison(ax, causal_results):
    """Create paired bar chart showing pre-post treatment means with dual y-axis"""
    if not causal_results:
        ax.text(0.5, 0.5, 'No causal results available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Pre-Post Treatment Comparison', fontsize=14, fontweight='bold')
        return
    
    sectors = []
    pre_means = []
    post_means = []
    method_types = []
    
    for sector, results in causal_results.items():
        if 'methods' in results:
            # Try difference_in_differences first
            if 'difference_in_differences' in results['methods']:
                did_result = results['methods']['difference_in_differences']
                if 'pre_treatment_mean' in did_result and 'post_treatment_mean' in did_result:
                    sectors.append(sector.upper())
                    pre_means.append(did_result['pre_treatment_mean'])
                    post_means.append(did_result['post_treatment_mean'])
                    method_types.append('DiD')
            
            # If no DiD results, try robust_alternative and create synthetic pre/post values
            elif 'robust_alternative' in results['methods']:
                robust_result = results['methods']['robust_alternative']
                if 'summary' in robust_result and 'average_effect' in robust_result['summary']:
                    # Create synthetic pre/post values from average effect
                    avg_effect = robust_result['summary']['average_effect']
                    # Assume a baseline value and calculate pre/post from effect
                    baseline = 100  # Use 100 as baseline for percentage calculation
                    pre_value = baseline
                    post_value = baseline + avg_effect
                    
                    sectors.append(sector.upper())
                    pre_means.append(pre_value)
                    post_means.append(post_value)
                    method_types.append('Robust')
    
    if not sectors:
        ax.text(0.5, 0.5, 'No treatment comparison data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Pre-Post Treatment Comparison', fontsize=14, fontweight='bold')
        return
    
    # Create dual y-axis
    ax2 = ax.twinx()
    
    x = np.arange(len(sectors))
    width = 0.35
    
    # Separate data by scale - GDP (large scale) vs Technology/Tourism (small scale)
    gdp_data = {'sectors': [], 'pre': [], 'post': [], 'methods': []}
    other_data = {'sectors': [], 'pre': [], 'post': [], 'methods': []}
    
    for i, sector in enumerate(sectors):
        if sector == 'GDP':
            gdp_data['sectors'].append(sector)
            gdp_data['pre'].append(pre_means[i])
            gdp_data['post'].append(post_means[i])
            gdp_data['methods'].append(method_types[i])
        else:
            other_data['sectors'].append(sector)
            other_data['pre'].append(pre_means[i])
            other_data['post'].append(post_means[i])
            other_data['methods'].append(method_types[i])
    
    # Plot GDP data on primary y-axis (left)
    if gdp_data['sectors']:
        gdp_x = np.arange(len(gdp_data['sectors']))
        bars1_gdp = ax.bar(gdp_x - width/2, gdp_data['pre'], width, 
                          label='GDP Pre-Vision 2030', color='lightcoral', alpha=0.8, edgecolor='black')
        bars2_gdp = ax.bar(gdp_x + width/2, gdp_data['post'], width, 
                          label='GDP Post-Vision 2030', color='lightgreen', alpha=0.8, edgecolor='black')
        
        # Add value labels for GDP
        for bars in [bars1_gdp, bars2_gdp]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot Technology/Tourism data on secondary y-axis (right)
    if other_data['sectors']:
        other_x = np.arange(len(other_data['sectors'])) + len(gdp_data['sectors']) + 0.5
        
        # Color bars based on positive/negative values
        pre_colors = ['lightblue' if val >= 0 else 'lightcoral' for val in other_data['pre']]
        post_colors = ['orange' if val >= 0 else 'red' for val in other_data['post']]
        
        bars1_other = ax2.bar(other_x - width/2, other_data['pre'], width, 
                             label='Tech/Tourism Pre-Vision 2030', color=pre_colors, alpha=0.8, edgecolor='black')
        bars2_other = ax2.bar(other_x + width/2, other_data['post'], width, 
                             label='Tech/Tourism Post-Vision 2030', color=post_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels for Technology/Tourism - handle negative values
        for i, bar in enumerate(bars1_other):
            height = bar.get_height()
            label_y = height + (abs(height) * 0.05 if height >= 0 else -abs(height) * 0.1)
            va_pos = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}', ha='center', va=va_pos, fontsize=10, fontweight='bold')
        
        for i, bar in enumerate(bars2_other):
            height = bar.get_height()
            label_y = height + (abs(height) * 0.05 if height >= 0 else -abs(height) * 0.1)
            va_pos = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}', ha='center', va=va_pos, fontsize=10, fontweight='bold')
        
        # Add method indicators
        for i, (x_pos, method) in enumerate(zip(other_x, other_data['methods'])):
            ax2.text(x_pos, max(other_data['pre'][i], other_data['post'][i]) + 20,
                    f'({method})', ha='center', va='bottom', fontsize=8, 
                    style='italic', color='gray')
        
        # Add zero line for secondary axis if there are negative values
        if any(val < 0 for val in other_data['pre'] + other_data['post']):
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set labels and titles
    ax.set_xlabel('Sectors', fontsize=12)
    ax.set_ylabel('GDP Values (Millions)', fontsize=12, color='darkred')
    ax2.set_ylabel('Technology/Tourism Values', fontsize=12, color='darkblue')
    ax.set_title('Pre-Post Vision 2030 Treatment Comparison\n(Dual Scale)', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks
    all_x_ticks = []
    all_x_labels = []
    
    if gdp_data['sectors']:
        all_x_ticks.extend(gdp_x)
        all_x_labels.extend(gdp_data['sectors'])
    
    if other_data['sectors']:
        all_x_ticks.extend(other_x)
        all_x_labels.extend(other_data['sectors'])
    
    ax.set_xticks(all_x_ticks)
    ax.set_xticklabels(all_x_labels)
    
    # Color the y-axis labels
    ax.tick_params(axis='y', labelcolor='darkred')
    ax2.tick_params(axis='y', labelcolor='darkblue')
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add a separator line between GDP and other sectors
    if gdp_data['sectors'] and other_data['sectors']:
        separator_x = len(gdp_data['sectors']) - 0.5 + width/2
        ax.axvline(x=separator_x, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(separator_x, ax.get_ylim()[1] * 0.95, 'Scale\nBreak', 
                ha='center', va='top', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def create_volatility_analysis(ax, advanced_results):
    """Create volatility analysis visualization"""
    sectors = []
    pre_vol = []
    post_vol = []
    changes = []
    
    for sector, results in advanced_results.items():
        if 'analyses' in results and 'volatility' in results['analyses']:
            vol_result = results['analyses']['volatility']
            sectors.append(sector.upper())
            pre_vol.append(vol_result['volatility_pre_vision'])
            post_vol.append(vol_result['volatility_post_vision'])
            changes.append(vol_result['volatility_change_percent'])
    
    if not sectors:
        ax.text(0.5, 0.5, 'No volatility data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Volatility Analysis', fontsize=14, fontweight='bold')
        return
    
    x = np.arange(len(sectors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pre_vol, width, label='Pre-Vision 2030', 
                   color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, post_vol, width, label='Post-Vision 2030', 
                   color='orange', alpha=0.8, edgecolor='black')
    
    # Add change percentage as text
    for i, change in enumerate(changes):
        ax.text(i, max(pre_vol[i], post_vol[i]) + 0.001, 
               f'{change:+.1f}%', ha='center', va='bottom', 
               fontweight='bold', fontsize=10,
               color='red' if change > 0 else 'green')
    
    ax.set_xlabel('Sectors', fontsize=12)
    ax.set_ylabel('Volatility (Standard Deviation)', fontsize=12)
    ax.set_title('Volatility Changes Post-Vision 2030', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_method_comparison(ax, causal_results):
    """Create method performance comparison"""
    if not causal_results:
        ax.text(0.5, 0.5, 'No causal results available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Method Performance Comparison', fontsize=14, fontweight='bold')
        return
    
    methods = ['DiD', 'Robust']
    sectors = ['GDP', 'Technology', 'Tourism']
    
    # Prepare data
    did_effects = []
    robust_effects = []
    
    for sector in ['gdp', 'technology', 'tourism']:
        if sector in causal_results and 'methods' in causal_results[sector]:
            # DiD effect
            if 'difference_in_differences' in causal_results[sector]['methods']:
                did_effects.append(causal_results[sector]['methods']['difference_in_differences'].get('relative_effect_percent', 0))
            else:
                did_effects.append(0)
            
            # Robust effect
            if 'robust_alternative' in causal_results[sector]['methods']:
                robust_effects.append(causal_results[sector]['methods']['robust_alternative']['summary'].get('relative_effect', 0))
            else:
                robust_effects.append(0)
    
    if not did_effects and not robust_effects:
        ax.text(0.5, 0.5, 'No method comparison data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Method Performance Comparison', fontsize=14, fontweight='bold')
        return
    
    x = np.arange(len(sectors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, did_effects, width, label='Difference-in-Differences', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, robust_effects, width, label='Robust Alternative', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                   f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Sectors', fontsize=12)
    ax.set_ylabel('Relative Effect (%)', fontsize=12)
    ax.set_title('Causal Method Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

def create_sector_dashboard(ax, summary_report):
    """Create sector performance dashboard"""
    if not summary_report or 'overall_findings' not in summary_report:
        ax.text(0.5, 0.5, 'No summary data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Sector Performance Dashboard', fontsize=14, fontweight='bold')
        return
    
    findings = summary_report['overall_findings']
    
    # Create summary statistics
    total_sectors = findings.get('total_sectors_analyzed', 0)
    significant_sectors = findings.get('sectors_with_significant_effects', 0) if 'sectors_with_significant_effects' in findings else 0
    
    # Create pie chart for significant vs non-significant
    sizes = [significant_sectors, total_sectors - significant_sectors]
    labels = ['Significant Effects', 'Non-Significant']
    colors = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, explode=(0.05, 0.05))
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Sector Analysis Results\n(Statistical Significance)', fontsize=14, fontweight='bold')
    
    # Add text box with key statistics
    stats_text = f"""
    Total Sectors: {total_sectors}
    Significant: {significant_sectors}
    Success Rate: {(significant_sectors/total_sectors*100):.1f}%
    """
    
    ax.text(1.5, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='center')

def load_causal_results():
    try:
        import json
        import os
        
        # Try to load from the causal impact results directory
        causal_summary_path = 'Results/causal_impact/causal_impact_summary_report.json'
        if os.path.exists(causal_summary_path):
            with open(causal_summary_path, 'r') as f:
                causal_summary = json.load(f)
        else:
            causal_summary = load_json_safe('causal_impact_summary_report.json')
        
        causal_results = {}
        if causal_summary and 'sector_effects' in causal_summary:
            for sector, effects in causal_summary['sector_effects'].items():
                causal_results[sector] = {
                    'methods': {
                        'difference_in_differences': {
                            'did_estimate': effects.get('did_estimate', 0),
                            'relative_effect_percent': effects.get('relative_effect_percent', 0),
                            'p_value': causal_summary.get('statistical_significance', {}).get(sector, {}).get('p_value', 1.0),
                            'significant': causal_summary.get('statistical_significance', {}).get(sector, {}).get('significant_at_5pct', False),
                            'pre_treatment_mean': effects.get('pre_treatment_mean', 0),
                            'post_treatment_mean': effects.get('post_treatment_mean', 0)
                        }
                    }
                }
                
                # Add robust alternative method if available
                if 'method' in effects and effects['method'] == 'Robust_Alternative':
                    causal_results[sector]['methods']['robust_alternative'] = {
                        'summary': {
                            'average_effect': effects.get('average_effect', 0),
                            'relative_effect': effects.get('relative_effect_percent', 0),
                            'p_value': causal_summary.get('statistical_significance', {}).get(sector, {}).get('p_value', 1.0),
                            'significant': causal_summary.get('statistical_significance', {}).get(sector, {}).get('significant_at_5pct', False)
                        }
                    }
        
        return causal_results
    except Exception as e:
        print(f"Error loading causal results: {str(e)}")
        return None

def econometric_validation_run(main_df):
    analysis_data, sector_data = prepare_advanced_analysis_data(main_df)
    if not sector_data:
        return None
    
    advanced_results = run_advanced_analysis(sector_data)
    if not advanced_results:
        return None
    
    causal_results = load_causal_results()
    summary_report = generate_advanced_analysis_summary(advanced_results)
    
    save_advanced_analysis_results(advanced_results, summary_report)
    
    # Create enhanced visualizations
    try:
        create_enhanced_visualizations(advanced_results, causal_results, summary_report)
        print("Enhanced visualizations created successfully!")
        
        # Create individual visualizations
        create_individual_visualizations(advanced_results, causal_results, summary_report)
        
    except Exception as e:
        print(f"Visualization creation failed: {str(e)}")
    
    return {
        'analysis_data': analysis_data,
        'sector_data': sector_data,
        'advanced_results': advanced_results,
        'causal_results': causal_results,
        'summary_report': summary_report
    }

if __name__ == "__main__":
    try:
        import data_loading_and_EDA
        main_df, cleaned_data, comparison_stats = data_loading_and_EDA.main()
        
        if main_df is not None:
            results = econometric_validation_run(main_df)
    except Exception as e:
        import traceback
        traceback.print_exc()
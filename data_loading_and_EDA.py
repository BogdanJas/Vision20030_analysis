import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan
import json

def load_and_inspect_data():
    datasets = {}
    data_path = 'Data/Saudi Arabia/'
    
    try:
        datasets['gdp'] = pd.read_csv(data_path + 'SA_GDP.csv')
        datasets['technology'] = pd.read_csv(data_path + 'SA_Technology.csv')
        datasets['tourism'] = pd.read_csv(data_path + 'SA_Tourism.csv')
        datasets['exports'] = pd.read_csv(data_path + 'SA_Exports.csv')
        datasets['non_oil_exports'] = pd.read_csv(data_path + 'SA_Non_Oil_Exports.csv')
        datasets['oil_exports'] = pd.read_csv(data_path + 'SA_Oil_Exports.csv')
        datasets['fdi'] = pd.read_csv(data_path + 'SA_Foreign_Direct_Investment.csv')
        datasets['oil_prices'] = pd.read_csv(data_path + 'POILBREUSDM.csv')
        datasets['fiscal'] = pd.read_csv(data_path + 'SA_Fiscal_Expenditure.csv')
        datasets['inflation'] = pd.read_csv(data_path + 'SA_Inflation_Rate_MoM.csv')
        
    except Exception as e:
        return None
    
    return datasets

def clean_and_preprocess_data(datasets):
    cleaned_data = {}
    
    gdp_clean = datasets['gdp'].copy()
    gdp_clean['DateTime'] = pd.to_datetime(gdp_clean['DateTime'])
    gdp_clean = gdp_clean.sort_values('DateTime')
    gdp_clean['Year'] = gdp_clean['DateTime'].dt.year
    gdp_clean['Quarter'] = gdp_clean['DateTime'].dt.quarter
    cleaned_data['gdp'] = gdp_clean
    
    tech_clean = datasets['technology'].copy()
    tech_long = pd.melt(tech_clean, 
                       id_vars=['Indicator'], 
                       var_name='Year', 
                       value_name='Value')
    tech_long['Year'] = pd.to_numeric(tech_long['Year'], errors='coerce')
    tech_long = tech_long.dropna(subset=['Year', 'Value'])
    tech_long['DateTime'] = pd.to_datetime(tech_long['Year'].astype(int).astype(str) + '-01-01')
    cleaned_data['technology'] = tech_long
    
    tourism_clean = datasets['tourism'].copy()
    tourism_long = pd.melt(tourism_clean, 
                          id_vars=['Indicator'], 
                          var_name='Year', 
                          value_name='Value')
    tourism_long['Year'] = pd.to_numeric(tourism_long['Year'], errors='coerce')
    tourism_long = tourism_long.dropna(subset=['Year', 'Value'])
    tourism_long['DateTime'] = pd.to_datetime(tourism_long['Year'].astype(int).astype(str) + '-01-01')
    cleaned_data['tourism'] = tourism_long
    
    exports_clean = datasets['exports'].copy()
    exports_clean['DateTime'] = pd.to_datetime(exports_clean['DateTime'])
    exports_clean = exports_clean.sort_values('DateTime')
    cleaned_data['exports'] = exports_clean
    
    oil_clean = datasets['oil_prices'].copy()
    oil_clean['observation_date'] = pd.to_datetime(oil_clean['observation_date'])
    oil_clean = oil_clean.sort_values('observation_date')
    oil_clean = oil_clean.rename(columns={'observation_date': 'DateTime', 'POILBREUSDM': 'Oil_Price'})
    cleaned_data['oil_prices'] = oil_clean
    
    for key in ['non_oil_exports', 'oil_exports', 'fdi', 'fiscal', 'inflation']:
        if key in datasets:
            temp_clean = datasets[key].copy()
            temp_clean['DateTime'] = pd.to_datetime(temp_clean['DateTime'])
            temp_clean = temp_clean.sort_values('DateTime')
            cleaned_data[key] = temp_clean
    
    return cleaned_data

def create_composite_indices(cleaned_data):
    tech_data = cleaned_data['technology']
    tech_indicators = tech_data['Indicator'].unique()
    
    tech_data['Value'] = pd.to_numeric(tech_data['Value'], errors='coerce')
    tech_data = tech_data.dropna(subset=['Value'])
    
    tech_pivot = tech_data.pivot(index='DateTime', columns='Indicator', values='Value')
    
    percentage_cols = ['ICT goods exports (% of total goods exports)', 
                      'ICT goods imports (% total goods imports)',
                      'Individuals Using The Internet (% Of Population)',
                      'Households with a computer']
    
    for col in percentage_cols:
        if col in tech_pivot.columns:
            tech_pivot[col] = tech_pivot[col].astype(str).str.replace('%', '').astype(float)
    
    tech_pivot = tech_pivot.fillna(method='ffill').fillna(method='bfill')
    tech_pivot = tech_pivot.dropna(axis=1, thresh=len(tech_pivot)*0.5)
    
    scaler = StandardScaler()
    tech_scaled = pd.DataFrame(
        scaler.fit_transform(tech_pivot),
        index=tech_pivot.index,
        columns=tech_pivot.columns
    )
    
    pca = PCA(n_components=1)
    tech_index = pca.fit_transform(tech_scaled)
    tech_composite_df = pd.DataFrame({
        'DateTime': tech_pivot.index,
        'Technology_Index': tech_index.flatten()
    })
    
    tech_composite_df['Technology_Index'] = (
        (tech_composite_df['Technology_Index'] - tech_composite_df['Technology_Index'].min()) /
        (tech_composite_df['Technology_Index'].max() - tech_composite_df['Technology_Index'].min()) * 100
    )
    
    tourism_data = cleaned_data['tourism']
    tourism_indicators = tourism_data['Indicator'].unique()
    
    tourism_data['Value'] = pd.to_numeric(tourism_data['Value'], errors='coerce')
    tourism_data = tourism_data.dropna(subset=['Value'])
    
    tourism_pivot = tourism_data.pivot(index='DateTime', columns='Indicator', values='Value')
    tourism_pivot = tourism_pivot.fillna(method='ffill').fillna(method='bfill')
    tourism_pivot = tourism_pivot.dropna(axis=1, thresh=len(tourism_pivot)*0.5)
    
    tourism_scaled = pd.DataFrame(
        scaler.fit_transform(tourism_pivot),
        index=tourism_pivot.index,
        columns=tourism_pivot.columns
    )
    
    pca_tourism = PCA(n_components=1)
    tourism_index = pca_tourism.fit_transform(tourism_scaled)
    tourism_composite_df = pd.DataFrame({
        'DateTime': tourism_pivot.index,
        'Tourism_Index': tourism_index.flatten()
    })
    
    tourism_composite_df['Tourism_Index'] = (
        (tourism_composite_df['Tourism_Index'] - tourism_composite_df['Tourism_Index'].min()) /
        (tourism_composite_df['Tourism_Index'].max() - tourism_composite_df['Tourism_Index'].min()) * 100
    )
    
    return tech_composite_df, tourism_composite_df

def comprehensive_data_validation(main_df):
    validation_results = {}
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column not in main_df.columns:
            continue
            
        data = main_df[column].dropna()
        
        stats_summary = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25)
        }
        
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        normality_test = stats.shapiro(data)
        
        try:
            adf_test = adfuller(data)
            stationarity_test = {'adf_statistic': adf_test[0], 'p_value': adf_test[1]}
        except:
            stationarity_test = {'adf_statistic': np.nan, 'p_value': np.nan}
        
        validation_results[column] = {
            'statistics': stats_summary,
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(data) * 100,
            'outlier_values': outliers.tolist(),
            'normality_test': {'statistic': normality_test[0], 'p_value': normality_test[1]},
            'stationarity_test': stationarity_test,
            'data_quality_score': 100 - (len(outliers) / len(data) * 100)
        }
    
    return validation_results

def handle_outliers_and_missing_data(main_df, validation_results):
    cleaned_df = main_df.copy()
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column not in main_df.columns:
            continue
            
        data = main_df[column].dropna()
        
        if validation_results[column]['outliers_count'] > 0:
            q1, q99 = data.quantile([0.01, 0.99])
            cleaned_df[column] = cleaned_df[column].clip(lower=q1, upper=q99)
            
            q05, q95 = data.quantile([0.05, 0.95])
            extreme_outliers = (cleaned_df[column] < q05) | (cleaned_df[column] > q95)
            if extreme_outliers.sum() > 0:
                median_val = data.median()
                cleaned_df.loc[extreme_outliers, column] = median_val
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column in cleaned_df.columns:
            missing_count = cleaned_df[column].isna().sum()
            if missing_count > 0:
                cleaned_df[column] = cleaned_df[column].interpolate(method='cubic')
    
    return cleaned_df

def create_robust_variables(cleaned_df):
    enhanced_df = cleaned_df.copy()
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column in enhanced_df.columns:
            enhanced_df[f'{column}_YoY_Growth'] = enhanced_df[column].pct_change(4) * 100
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column in enhanced_df.columns:
            enhanced_df[f'{column}_MA4'] = enhanced_df[column].rolling(window=4).mean()
            enhanced_df[f'{column}_MA8'] = enhanced_df[column].rolling(window=8).mean()
    
    for column in ['GDP', 'Technology_Index', 'Tourism_Index']:
        if column in enhanced_df.columns:
            enhanced_df[f'{column}_Volatility'] = enhanced_df[column].rolling(window=8).std()
    
    if 'GDP' in enhanced_df.columns and 'Technology_Index' in enhanced_df.columns:
        enhanced_df['Tech_GDP_Ratio'] = enhanced_df['Technology_Index'] / enhanced_df['GDP'] * 1000
    
    if 'GDP' in enhanced_df.columns and 'Tourism_Index' in enhanced_df.columns:
        enhanced_df['Tourism_GDP_Ratio'] = enhanced_df['Tourism_Index'] / enhanced_df['GDP'] * 1000
    
    return enhanced_df

def exploratory_data_analysis(cleaned_data, tech_composite, tourism_composite):
    import os
    results_dir = 'Results/data_loading_and_EDA'
    os.makedirs(results_dir, exist_ok=True)
    
    gdp_data = cleaned_data['gdp']
    gdp_constant = gdp_data[gdp_data['Category'] == 'GDP Constant Prices'].copy()
    main_df = gdp_constant[['DateTime', 'Value']].copy()
    main_df = main_df.rename(columns={'Value': 'GDP'})
    
    oil_quarterly = cleaned_data['oil_prices'].copy()
    oil_quarterly['Year'] = oil_quarterly['DateTime'].dt.year
    oil_quarterly['Quarter'] = oil_quarterly['DateTime'].dt.quarter
    oil_quarterly = oil_quarterly.groupby(['Year', 'Quarter'])['Oil_Price'].mean().reset_index()
    oil_quarterly['DateTime'] = pd.to_datetime(oil_quarterly['Year'].astype(str) + '-' + 
                                              ((oil_quarterly['Quarter']-1)*3+1).astype(str) + '-01')
    
    main_df = pd.merge(main_df, oil_quarterly[['DateTime', 'Oil_Price']], on='DateTime', how='outer')
    
    main_df = main_df.sort_values('DateTime')
    main_df['Oil_Price'] = main_df['Oil_Price'].fillna(method='ffill').fillna(method='bfill')
    
    tech_quarterly = tech_composite.copy()
    tech_quarterly = tech_quarterly.set_index('DateTime').resample('Q').ffill().reset_index()
    main_df = pd.merge(main_df, tech_quarterly, on='DateTime', how='left')
    
    tourism_quarterly = tourism_composite.copy()
    tourism_quarterly = tourism_quarterly.set_index('DateTime').resample('Q').ffill().reset_index()
    main_df = pd.merge(main_df, tourism_quarterly, on='DateTime', how='left')
    
    main_df['Vision_2030_Implemented'] = (main_df['DateTime'] >= '2017-01-01').astype(int)
    main_df['COVID_Period'] = ((main_df['DateTime'] >= '2020-03-01') & 
                              (main_df['DateTime'] <= '2022-12-31')).astype(int)
    
    main_df['Year'] = main_df['DateTime'].dt.year
    main_df['Quarter'] = main_df['DateTime'].dt.quarter
    main_df['Time_Trend'] = range(len(main_df))
    
    validation_results = comprehensive_data_validation(main_df)
    cleaned_df = handle_outliers_and_missing_data(main_df, validation_results)
    enhanced_df = create_robust_variables(cleaned_df)
    
    with open(f'{results_dir}/enhanced_data_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    return enhanced_df, validation_results

def create_visualizations(main_df):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Enhanced EDA: Saudi Arabia Economic Indicators', fontsize=16, fontweight='bold')
    
    for i, column in enumerate(['GDP', 'Technology_Index', 'Tourism_Index']):
        if column not in main_df.columns:
            continue
            
        axes[0, i].plot(main_df['DateTime'], main_df[column], linewidth=2)
        axes[0, i].axvline(x=pd.to_datetime('2017-01-01'), color='red', linestyle='--', 
                          label='Vision 2030 Start')
        axes[0, i].axvline(x=pd.to_datetime('2020-03-01'), color='orange', linestyle='--', 
                          label='COVID-19 Start')
        axes[0, i].set_title(f'{column} Time Series')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].tick_params(axis='x', rotation=45)
    
    for i, column in enumerate(['GDP', 'Technology_Index', 'Tourism_Index']):
        if column not in main_df.columns:
            continue
            
        axes[1, i].hist(main_df[column].dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[1, i].set_title(f'{column} Distribution')
        axes[1, i].grid(True, alpha=0.3)
    
    for i, column in enumerate(['GDP', 'Technology_Index', 'Tourism_Index']):
        if column not in main_df.columns:
            continue
            
        growth_col = f'{column}_YoY_Growth'
        if growth_col in main_df.columns:
            axes[2, i].plot(main_df['DateTime'], main_df[growth_col], linewidth=2)
            axes[2, i].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2, i].axvline(x=pd.to_datetime('2017-01-01'), color='red', linestyle='--')
            axes[2, i].set_title(f'{column} YoY Growth Rate')
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].tick_params(axis='x', rotation=45)
    
    import os
    results_dir = 'Results/data_loading_and_EDA'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/eda_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    numeric_cols = main_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = main_df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix of Economic Indicators')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def generate_summary_statistics(main_df, validation_results):
    pre_vision = main_df[main_df['Vision_2030_Implemented'] == 0]
    post_vision = main_df[main_df['Vision_2030_Implemented'] == 1]
    
    comparison_stats = pd.DataFrame({
        'Pre_Vision_Mean': pre_vision[['GDP', 'Technology_Index', 'Tourism_Index', 'Oil_Price']].mean(),
        'Post_Vision_Mean': post_vision[['GDP', 'Technology_Index', 'Tourism_Index', 'Oil_Price']].mean(),
    })
    comparison_stats['Percentage_Change'] = ((comparison_stats['Post_Vision_Mean'] - 
                                            comparison_stats['Pre_Vision_Mean']) / 
                                           comparison_stats['Pre_Vision_Mean'] * 100)
    
    availability = pd.DataFrame({
        'Total_Observations': main_df.count(),
        'Missing_Values': main_df.isnull().sum(),
        'Availability_Percentage': (main_df.count() / len(main_df) * 100).round(2)
    })
    
    enhanced_summary = {
        'dataset_info': {
            'total_observations': len(main_df),
            'date_range': {
                'start': main_df['DateTime'].min().strftime('%Y-%m-%d'),
                'end': main_df['DateTime'].max().strftime('%Y-%m-%d')
            },
            'variables_created': list(main_df.columns)
        },
        'data_quality_summary': {
            column: {
                'quality_score': validation_results[column]['data_quality_score'],
                'outliers_handled': validation_results[column]['outliers_count'],
                'missing_values_filled': main_df[column].isna().sum()
            }
            for column in validation_results.keys()
        },
        'preprocessing_steps': [
            'Composite indices created using PCA',
            'Outliers handled with winsorization',
            'Missing data interpolated with cubic method',
            'Derived variables created (growth rates, moving averages)',
            'Vision 2030 and COVID-19 indicators added'
        ]
    }
    
    import os
    results_dir = 'Results/data_loading_and_EDA'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/enhanced_preprocessing_summary.json', 'w') as f:
        json.dump(enhanced_summary, f, indent=2, default=str)
    
    return comparison_stats, availability, enhanced_summary

def main():
    datasets = load_and_inspect_data()
    if datasets is None:
        return None, None, None
    
    cleaned_data = clean_and_preprocess_data(datasets)
    tech_composite, tourism_composite = create_composite_indices(cleaned_data)
    main_df, validation_results = exploratory_data_analysis(cleaned_data, tech_composite, tourism_composite)
    fig = create_visualizations(main_df)
    comparison_stats, availability, enhanced_summary = generate_summary_statistics(main_df, validation_results)
    
    return main_df, cleaned_data, comparison_stats
    
if __name__ == "__main__":
    main_df, cleaned_data, comparison_stats = main()
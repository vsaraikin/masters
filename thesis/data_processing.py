"""
Data processing and feature engineering for Bitcoin prediction.
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
import shap

from config import CONFIG


class DataProcessor:
    """handles loading and preprocessing of glassnode data"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['data']
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """load and merge csv files from glassnode"""
        try:
            csv_files = [f for f in os.listdir(self.config.folder_path) 
                        if f.endswith('.csv')]
            print(f"found {len(csv_files)} csv files")
            
            dataframes = []
            for file in csv_files:
                df = pd.read_csv(os.path.join(self.config.folder_path, file))
                suffix = file[4:].replace('.csv', '')
                df.rename(columns={col: f"{col}_{suffix}" 
                         for col in df.columns if col != 'timestamp'}, 
                         inplace=True)
                df.set_index('timestamp', inplace=True)
                dataframes.append(df)
            
            combined_df = dataframes[0]
            for df in dataframes[1:]:
                combined_df = combined_df.join(df, how='outer')
                
            self.raw_data = combined_df.loc[
                self.config.start_date:self.config.end_date
            ]
            
        except FileNotFoundError:
            print("creating sample data...")
            self.raw_data = self._create_sample_data()
            
        return self.raw_data
    
    def _create_sample_data(self) -> pd.DataFrame:
        """create synthetic data for testing"""
        dates = pd.date_range(
            self.config.start_date, 
            self.config.end_date, 
            freq='D'
        )
        np.random.seed(self.config.random_seed)
        
        # realistic bitcoin price simulation
        price_returns = np.random.normal(0.001, 0.05, len(dates))
        price = 10000 * np.exp(np.cumsum(price_returns))
        
        features = {'v_v1_metrics_market_price_usd_close': price}
        
        # generate correlated features
        for i in range(40):
            correlation = np.random.uniform(-0.7, 0.7)
            noise = np.random.normal(0, 1, len(dates))
            feature = correlation * np.log(price) + (1-abs(correlation)) * noise
            features[f'feature_{i:02d}'] = feature
        
        return pd.DataFrame(features, index=dates)
    
    def preprocess_data(self) -> pd.DataFrame:
        """clean and filter data"""
        if self.raw_data is None:
            self.load_data()
            
        # rename price column and select numeric data
        float_data = self.raw_data.rename(
            columns={'v_v1_metrics_market_price_usd_close': 'price'}
        )
        float_data = float_data.select_dtypes(include=['float'])
        
        # remove highly correlated features
        float_data = self._remove_correlated_features(float_data)
        
        # create log returns
        float_data['log_ret'] = np.log(
            float_data['price'] / float_data['price'].shift(1)
        )
        
        self.processed_data = float_data
        return self.processed_data
    
    def _remove_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """remove features with high correlation"""
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for column in upper_tri.columns:
            if column != 'price':
                high_corr = upper_tri.index[
                    upper_tri[column] >= self.config.correlation_threshold
                ].tolist()
                if high_corr and column not in to_drop:
                    to_drop.append(column)
        
        data.drop(columns=to_drop, inplace=True)
        print(f"removed {len(to_drop)} highly correlated features")
        
        return data


class CorrelationAnalyzer:
    """performs comprehensive correlation analysis"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.correlation_matrix = None
        
    def compute_correlations(self) -> pd.DataFrame:
        """compute correlation matrix"""
        self.correlation_matrix = self.data.corr()
        return self.correlation_matrix
    
    def plot_correlation_heatmap(self, top_features: int = 20):
        """plot correlation heatmap for top features"""
        if self.correlation_matrix is None:
            self.compute_correlations()
            
        # select top features by variance
        feature_var = self.data.var().sort_values(ascending=False)
        top_features_list = feature_var.head(top_features).index
        
        corr_subset = self.correlation_matrix.loc[
            top_features_list, top_features_list
        ]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_subset, 
            annot=False, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            linewidths=0.1
        )
        plt.title('correlation matrix - top features by variance')
        plt.tight_layout()
        plt.show()
        
    def find_highly_correlated_pairs(self, threshold: float = 0.8) -> pd.DataFrame:
        """find pairs of features with high correlation"""
        if self.correlation_matrix is None:
            self.compute_correlations()
            
        # get upper triangle of correlation matrix
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # find high correlations
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if abs(upper_tri.loc[idx, col]) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': idx,
                        'feature_2': col,
                        'correlation': upper_tri.loc[idx, col]
                    })
        
        return pd.DataFrame(high_corr_pairs).sort_values(
            'correlation', key=abs, ascending=False
        )
    
    def analyze_target_correlations(self, target_col: str) -> pd.DataFrame:
        """analyze correlations with target variable"""
        if target_col not in self.data.columns:
            raise ValueError(f"target column '{target_col}' not found")
            
        target_corrs = self.data.corrwith(self.data[target_col]).abs()
        target_corrs = target_corrs.drop(target_col).sort_values(ascending=False)
        
        return target_corrs.to_frame('correlation')


class FeatureSelector:
    """multi-criteria feature selection"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['feature_selection']
        self.selected_features = None
        self.feature_scores = None
        
    def select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """perform multi-criteria feature selection"""
        print(f"multi-criteria feature selection")
        print(f"top-{self.config.top_k_per_method} per method")
        
        # train temporary model for shap analysis
        temp_model = CatBoostRegressor(
            iterations=300, 
            verbose=False, 
            random_seed=42
        )
        temp_model.fit(X_train, y_train)
        
        # compute importance scores
        shap_df = self._compute_shap_importance(temp_model, X_train)
        catboost_df = self._compute_catboost_importance(X_train, y_train)
        spearman_df = self._compute_spearman_correlation(X_train, y_train)
        
        # merge all scores
        self.feature_scores = (shap_df
                              .merge(catboost_df, on='feature')
                              .merge(spearman_df, on='feature'))
        
        # select top features from each method
        top_shap = set(shap_df.head(self.config.top_k_per_method)['feature'])
        top_catboost = set(catboost_df.head(self.config.top_k_per_method)['feature'])
        top_spearman = set(spearman_df.head(self.config.top_k_per_method)['feature'])
        
        # union of top features
        self.selected_features = list(
            top_shap.union(top_catboost).union(top_spearman)
        )[:self.config.max_total_features]
        
        print(f"selected {len(self.selected_features)} unique features")
        
        return X_train[self.selected_features], X_test[self.selected_features]
    
    def _compute_shap_importance(self, model, X_sample: pd.DataFrame) -> pd.DataFrame:
        """compute shap-based feature importance"""
        print("computing shap importance...")
        
        # subsample for computational efficiency
        sample_size = min(self.config.shap_max_evals, len(X_sample))
        X_shap = X_sample.sample(n=sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # mean absolute shap values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        return pd.DataFrame({
            'feature': X_sample.columns,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
    
    def _compute_catboost_importance(self, X_train: pd.DataFrame, 
                                   y_train: pd.Series) -> pd.DataFrame:
        """compute catboost feature importance"""
        print("computing catboost importance...")
        
        temp_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_seed=42
        )
        
        temp_model.fit(X_train, y_train)
        importance = temp_model.get_feature_importance()
        
        return pd.DataFrame({
            'feature': X_train.columns,
            'catboost_importance': importance
        }).sort_values('catboost_importance', ascending=False)
    
    def _compute_spearman_correlation(self, X_train: pd.DataFrame, 
                                    y_train: pd.Series) -> pd.DataFrame:
        """compute spearman rank correlation with target"""
        print("computing spearman correlations...")
        
        correlations = []
        for col in X_train.columns:
            corr, _ = spearmanr(X_train[col], y_train)
            correlations.append(abs(corr))
        
        return pd.DataFrame({
            'feature': X_train.columns,
            'spearman_corr': correlations
        }).sort_values('spearman_corr', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 20):
        """plot feature importance comparison"""
        if self.feature_scores is None:
            print("run feature selection first")
            return
        
        top_features = self.feature_scores.head(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # shap importance
        axes[0].barh(range(len(top_features)), top_features['shap_importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'], fontsize=8)
        axes[0].set_title('shap importance')
        axes[0].set_xlabel('mean |shap value|')
        
        # catboost importance  
        axes[1].barh(range(len(top_features)), top_features['catboost_importance'])
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['feature'], fontsize=8)
        axes[1].set_title('catboost importance')
        axes[1].set_xlabel('feature importance')
        
        # spearman correlation
        axes[2].barh(range(len(top_features)), top_features['spearman_corr'])
        axes[2].set_yticks(range(len(top_features)))
        axes[2].set_yticklabels(top_features['feature'], fontsize=8)
        axes[2].set_title('spearman correlation')
        axes[2].set_xlabel('|correlation|')
        
        plt.tight_layout()
        plt.show()


class DataPipeline:
    """complete data processing pipeline"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['data']
        self.processor = DataProcessor(config)
        self.imputer = None
        self.scaler = None
        
    def prepare_data(self) -> pd.DataFrame:
        """prepare clean data for modeling"""
        # load and preprocess
        self.processor.preprocess_data()
        data = self.processor.processed_data.copy()
        
        return data
    
    def create_train_test_split(self, data: pd.DataFrame, 
                               target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                        pd.Series, pd.Series]:
        """create train/test split"""
        # remove target from features
        exclude_cols = ['price', 'log_ret', target_col]
        X = data.drop(columns=exclude_cols, errors='ignore')
        y = data[target_col]
        
        # time-based split
        split_idx = int(len(data) * self.config.train_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_features(self, X_train: pd.DataFrame, 
                           X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """impute and scale features"""
        # remove empty and constant features
        X_train_clean, dropped = self._drop_empty_const_features(X_train)
        X_test_clean = X_test.drop(columns=dropped, errors='ignore')
        
        # imputation
        self.imputer = IterativeImputer(
            random_state=self.config.random_seed, 
            max_iter=2
        )
        X_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_train_clean), 
            columns=X_train_clean.columns, 
            index=X_train_clean.index
        )
        X_test_imputed = pd.DataFrame(
            self.imputer.transform(X_test_clean), 
            columns=X_test_clean.columns, 
            index=X_test_clean.index
        )
        
        # scaling
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_imputed), 
            columns=X_train_imputed.columns, 
            index=X_train_imputed.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_imputed), 
            columns=X_test_imputed.columns, 
            index=X_test_imputed.index
        )
        
        return X_train_scaled, X_test_scaled
    
    def _drop_empty_const_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """remove empty and constant features"""
        empty = df.columns[df.isnull().all()]
        const = df.columns[df.nunique(dropna=True) == 1]
        dropped_features = list(empty.union(const))
        cleaned_df = df.drop(columns=dropped_features)
        return cleaned_df, dropped_features
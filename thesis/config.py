"""
Configuration settings for Bitcoin prediction models.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataConfig:
    """data processing configuration"""
    folder_path: str = 'glassnode_data_btc'
    start_date: str = '2020-01-01'
    end_date: str = '2024-06-01'
    correlation_threshold: float = 0.95
    train_split: float = 0.8
    target_window: int = 10
    random_seed: int = 42


@dataclass
class ModelConfig:
    """base model configuration"""
    random_seed: int = 42
    early_stopping_rounds: int = 100
    n_trials: int = 50


@dataclass
class CatBoostConfig(ModelConfig):
    """catboost specific configuration"""
    
    def get_base_params(self, model_type: str = 'regression') -> Dict[str, Any]:
        """get base parameters for specific model type"""
        if model_type == 'regression':
            return {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'random_seed': self.random_seed,
                'verbose': False,
                'early_stopping_rounds': self.early_stopping_rounds
            }
        else:  # classification
            return {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'random_seed': self.random_seed,
                'verbose': False,
                'early_stopping_rounds': self.early_stopping_rounds
            }
    
    @property
    def optuna_search_space(self) -> Dict[str, tuple]:
        """optuna hyperparameter search space"""
        return {
            'iterations': (500, 2000),
            'learning_rate': (0.01, 0.3),
            'depth': (4, 10),
            'l2_leaf_reg': (1.0, 10.0),
            'border_count': (32, 255),
            'bagging_temperature': (0.0, 1.0)
        }


@dataclass
class TransformerConfig(ModelConfig):
    """transformer specific configuration"""
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 5
    dropout: float = 0.1
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3


@dataclass
class FeatureSelectionConfig:
    """feature selection configuration"""
    top_k_per_method: int = 30
    max_total_features: int = 50
    shap_max_evals: int = 1000


@dataclass
class BacktestConfig:
    """backtesting configuration"""
    threshold: float = 0.1
    transaction_cost: float = 0.001
    holding_period: int = 1


@dataclass
class VisualizationConfig:
    """visualization configuration"""
    style: str = 'seaborn-v0_8'
    palette: str = 'viridis'
    figure_size: tuple = (12, 8)
    dpi: int = 100
    font_scale: float = 1.1


# global configuration instance
CONFIG = {
    'data': DataConfig(),
    'catboost': CatBoostConfig(),
    'transformer': TransformerConfig(),
    'feature_selection': FeatureSelectionConfig(),
    'backtest': BacktestConfig(),
    'visualization': VisualizationConfig()
}
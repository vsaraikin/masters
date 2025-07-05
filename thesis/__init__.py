"""
Bitcoin Prediction Framework
"""

__version__ = "1.0.0"

from .config import CONFIG
from .data_processing import DataProcessor, CorrelationAnalyzer, FeatureSelector, DataPipeline
from .models import CatBoostModel, iTransformer, TransformerTrainer
from .analysis import SharpeSignalGenerator, VolatilityTargetCreator, BacktestFramework, VisualizationUtils, PerformanceAnalyzer

__all__ = [
    'CONFIG',
    'DataProcessor', 'CorrelationAnalyzer', 'FeatureSelector', 'DataPipeline',
    'CatBoostModel', 'iTransformer', 'TransformerTrainer', 
    'SharpeSignalGenerator', 'VolatilityTargetCreator', 'BacktestFramework', 
    'VisualizationUtils', 'PerformanceAnalyzer'
]
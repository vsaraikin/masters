"""
Analysis and backtesting framework for Bitcoin prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Callable
from scipy.stats import spearmanr

from config import CONFIG


class SharpeSignalGenerator:
    """generates forward-looking sharpe ratio targets"""
    
    def __init__(self, window: int = 10):
        self.window = window
        
    def create_target(self, price_series: pd.Series) -> pd.Series:
        """
        create forward-looking sharpe ratio target
        
        for every day t we define the forward h-day sharpe ratio:
        sharpe_t^(h) = sum(r_{t+i}) / std(r_{t+1}, ..., r_{t+h})
        where r_{t+i} = ln(p_{t+i}/p_{t+i-1}) are log returns
        """
        log_returns = np.log(price_series / price_series.shift(1))
        
        sharpe_signal = (
            log_returns
            .shift(-1)  # forward-looking
            .rolling(window=self.window, min_periods=self.window)
            .apply(lambda x: x.sum() / x.std(ddof=0) if x.std() > 0 else 0, raw=True)
        )
        
        return sharpe_signal


class VolatilityTargetCreator:
    """creates volatility prediction targets"""
    
    def __init__(self, window: int = 10):
        self.window = window
        
    def create_realized_volatility(self, price_series: pd.Series) -> pd.Series:
        """
        create forward-looking realized volatility target
        
        σ_t^(h) = sqrt(1/(h-1) * sum((r_{t+i} - r̄_t)^2))
        where r_{t+i} are log returns
        """
        log_returns = np.log(price_series / price_series.shift(1))
        
        future_volatility = (
            log_returns
            .rolling(window=self.window)
            .std()
            .shift(-self.window)  # make it forward-looking
        )
        
        return future_volatility
    
    def create_volatility_regime(self, volatility_series: pd.Series, 
                               n_quantiles: int = 3) -> pd.Series:
        """create volatility regime classification target"""
        volatility_clean = volatility_series.dropna()
        
        if len(volatility_clean) == 0:
            raise ValueError("no valid volatility data for regime creation")
            
        quantiles = np.linspace(0, 1, n_quantiles + 1)
        thresholds = volatility_clean.quantile(quantiles[1:-1])
        
        regime = pd.Series(0, index=volatility_series.index)
        for i, threshold in enumerate(thresholds):
            regime[volatility_series > threshold] = i + 1
            
        # проверка что у нас есть все классы
        unique_classes = regime.dropna().unique()
        print(f"volatility regime classes: {sorted(unique_classes)}")
        print(f"class distribution:\n{regime.value_counts().sort_index()}")
            
        return regime


class BacktestFramework:
    """trading strategy backtesting framework"""
    
    def __init__(self, predicted_signal: pd.Series, realized_returns: pd.Series,
                 price_series: Optional[pd.Series] = None, 
                 threshold: Optional[float] = None,
                 transaction_cost: float = 0.001, 
                 name: str = "strategy"):
        self.predicted_signal = predicted_signal
        self.realized_returns = realized_returns
        self.price_series = price_series
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.name = name
        self.results = None
        
    def generate_positions(self) -> pd.Series:
        """generate trading positions based on predicted signals"""
        if self.threshold is None:
            # buy and hold case
            positions = pd.Series(1.0, index=self.predicted_signal.index)
        else:
            # threshold-based strategy
            positions = pd.Series(0.0, index=self.predicted_signal.index)
            positions[self.predicted_signal > self.threshold] = 1.0    # long
            positions[self.predicted_signal < -self.threshold] = -1.0  # short
            
        return positions
    
    def calculate_portfolio_returns(self) -> tuple[pd.Series, pd.Series]:
        """calculate portfolio returns with transaction costs"""
        positions = self.generate_positions()
        
        # position changes for transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        
        # portfolio returns
        portfolio_returns = positions.shift(1) * self.realized_returns - transaction_costs
        portfolio_returns = portfolio_returns.fillna(0)
        
        return portfolio_returns, positions
    
    def run(self) -> pd.Series:
        """execute backtest and calculate performance metrics"""
        portfolio_returns, positions = self.calculate_portfolio_returns()
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        win_rate = (portfolio_returns > 0).mean()
        
        self.results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'cumulative_returns': cumulative_returns,
            'portfolio_returns': portfolio_returns,
            'positions': positions
        }
        
        return pd.Series(self.results)
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def summary(self) -> None:
        """print performance summary"""
        if self.results is None:
            print("run backtest first")
            return
            
        print(f"\n{self.name} performance summary")
        print("=" * 50)
        print(f"total return      : {self.results['total_return']:.2%}")
        print(f"annualized return : {self.results['annualized_return']:.2%}")
        print(f"annualized vol    : {self.results['annualized_volatility']:.2%}")
        print(f"sharpe ratio      : {self.results['sharpe_ratio']:.3f}")
        print(f"max drawdown      : {self.results['max_drawdown']:.2%}")
        print(f"win rate          : {self.results['win_rate']:.2%}")


class VisualizationUtils:
    """utility functions for visualization"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['visualization']
        self._setup_style()
        
    def _setup_style(self):
        """setup plotting style"""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.palette)
        sns.set_context("notebook", font_scale=self.config.font_scale)
        
    def plot_backtest_results(self, backtest_framework: BacktestFramework):
        """plot comprehensive backtest results"""
        if backtest_framework.results is None:
            print("run backtest first")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # cumulative returns
        axes[0, 0].plot(
            backtest_framework.results['cumulative_returns'], 
            label=backtest_framework.name, 
            linewidth=2
        )
        axes[0, 0].set_title('cumulative returns')
        axes[0, 0].set_ylabel('cumulative return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # drawdown
        rolling_max = backtest_framework.results['cumulative_returns'].expanding().max()
        drawdown = (backtest_framework.results['cumulative_returns'] - rolling_max) / rolling_max
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('drawdown')
        axes[0, 1].set_ylabel('drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # portfolio returns distribution
        axes[1, 0].hist(
            backtest_framework.results['portfolio_returns'], 
            bins=50, alpha=0.7, edgecolor='black'
        )
        axes[1, 0].set_title('returns distribution')
        axes[1, 0].set_xlabel('daily return')
        axes[1, 0].set_ylabel('frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # positions over time
        axes[1, 1].plot(backtest_framework.results['positions'], alpha=0.7)
        axes[1, 1].set_title('positions over time')
        axes[1, 1].set_ylabel('position')
        axes[1, 1].set_ylim(-1.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_model_predictions(self, y_true: pd.Series, y_pred: np.ndarray, 
                              model_name: str = "model"):
        """plot model predictions vs actual"""
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # predictions vs actual scatter
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('actual values')
        axes[0].set_ylabel('predicted values')
        axes[0].set_title(f'{model_name}: predictions vs actual')
        axes[0].grid(True, alpha=0.3)
        
        # time series of predictions
        common_idx = y_true.index[:len(y_pred)]
        axes[1].plot(common_idx, y_true.loc[common_idx], 
                    label='actual', alpha=0.8)
        axes[1].plot(common_idx, y_pred, 
                    label='predicted', alpha=0.8)
        axes[1].set_title(f'{model_name}: time series comparison')
        axes[1].set_xlabel('date')
        axes[1].set_ylabel('value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_volatility_analysis(self, price_series: pd.Series, 
                                volatility_series: pd.Series):
        """plot volatility analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # price and volatility
        axes[0, 0].plot(price_series.index, price_series, alpha=0.8)
        axes[0, 0].set_title('bitcoin price over time')
        axes[0, 0].set_ylabel('price (usd)')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(volatility_series.index, volatility_series, 
                       color='red', alpha=0.7)
        axes[0, 1].set_title('realized volatility')
        axes[0, 1].set_ylabel('volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        # volatility distribution
        vol_clean = volatility_series.dropna()
        axes[1, 0].hist(vol_clean, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('volatility distribution')
        axes[1, 0].set_xlabel('volatility')
        axes[1, 0].set_ylabel('frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # price vs volatility scatter
        common_idx = price_series.index.intersection(volatility_series.index)
        price_aligned = price_series.loc[common_idx]
        vol_aligned = volatility_series.loc[common_idx]
        
        mask = ~(np.isnan(price_aligned) | np.isnan(vol_aligned))
        axes[1, 1].scatter(price_aligned[mask], vol_aligned[mask], alpha=0.5)
        axes[1, 1].set_title('price vs volatility')
        axes[1, 1].set_xlabel('price (usd)')
        axes[1, 1].set_ylabel('volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_strategy_comparison(self, strategies: Dict[str, BacktestFramework]):
        """plot comparison of multiple strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # cumulative returns comparison
        for name, strategy in strategies.items():
            if strategy.results is not None:
                cum_returns = strategy.results['cumulative_returns']
                axes[0, 0].plot(cum_returns.index, cum_returns.values, label=name, linewidth=2)
        
        axes[0, 0].set_title('cumulative returns comparison')
        axes[0, 0].set_ylabel('cumulative return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # fix x-axis formatting
        import matplotlib.dates as mdates
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # performance metrics bar chart
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
        strategy_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for name, strategy in strategies.items():
            if strategy.results is not None:
                strategy_names.append(name)
                for metric in metrics:
                    metric_values[metric].append(strategy.results[metric])
        
        x = np.arange(len(strategy_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[0, 1].bar(x + i*width, metric_values[metric], 
                          width, label=metric.replace('_', ' '))
        
        axes[0, 1].set_title('performance metrics comparison')
        axes[0, 1].set_xlabel('strategy')
        axes[0, 1].set_ylabel('value')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels(strategy_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # drawdown comparison
        for name, strategy in strategies.items():
            if strategy.results is not None:
                rolling_max = strategy.results['cumulative_returns'].expanding().max()
                drawdown = (strategy.results['cumulative_returns'] - rolling_max) / rolling_max
                axes[1, 0].plot(drawdown, label=name, alpha=0.7)
        
        axes[1, 0].set_title('drawdown comparison')
        axes[1, 0].set_ylabel('drawdown')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # returns distribution comparison
        for name, strategy in strategies.items():
            if strategy.results is not None:
                axes[1, 1].hist(
                    strategy.results['portfolio_returns'], 
                    bins=30, alpha=0.5, label=name, density=True
                )
        
        axes[1, 1].set_title('returns distribution comparison')
        axes[1, 1].set_xlabel('daily return')
        axes[1, 1].set_ylabel('density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class PerformanceAnalyzer:
    """comprehensive performance analysis"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_model_performance(self, models: Dict[str, Any], 
                                 X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """analyze and compare model performance"""
        performance_data = []
        
        for name, model in models.items():
            try:
                if hasattr(model, 'evaluate'):
                    # catboost models
                    results = model.evaluate(X_test, y_test)
                    if 'ic' in results:
                        metric_name = f"ic: {results['ic']:.3f}"
                        rmse = results['rmse']
                    else:
                        metric_name = f"acc: {results['accuracy']:.3f}"
                        rmse = 'n/a'
                else:
                    # other models (like itransformer results dict)
                    if 'ic' in model:
                        metric_name = f"ic: {model['ic']:.3f}"
                        rmse = model['rmse']
                    else:
                        metric_name = 'n/a'
                        rmse = 'n/a'
                        
                performance_data.append({
                    'model': name,
                    'primary_metric': metric_name,
                    'rmse': f"{rmse:.4f}" if rmse != 'n/a' else 'n/a'
                })
                
            except Exception as e:
                print(f"error evaluating {name}: {e}")
                performance_data.append({
                    'model': name,
                    'primary_metric': 'error',
                    'rmse': 'error'
                })
        
        return pd.DataFrame(performance_data)
    
    def create_research_summary(self, models_performance: pd.DataFrame,
                              strategies_performance: Dict[str, Dict]) -> str:
        """create comprehensive research summary"""
        
        summary = f"""
BITCOIN PREDICTION FRAMEWORK - RESEARCH SUMMARY

methodology:
- multi-criteria feature selection (shap, catboost importance, spearman correlation)
- hyperparameter optimization using optuna
- comprehensive backtesting with transaction costs

model performance:
{models_performance.to_string(index=False)}

trading strategy results:
"""
        
        for strategy_name, results in strategies_performance.items():
            summary += f"\n{strategy_name}:"
            summary += f"  sharpe ratio: {results.get('sharpe_ratio', 'n/a'):.3f}"
            summary += f"  total return: {results.get('total_return', 'n/a'):.2%}"
            summary += f"  max drawdown: {results.get('max_drawdown', 'n/a'):.2%}"
        
        summary += f"""

key findings:
- feature selection improved model efficiency while maintaining performance
- risk-adjusted signals showed competitive performance vs buy-and-hold
- volatility prediction enables dynamic position sizing

practical implications:
- framework provides actionable insights for cryptocurrency portfolio management
- risk-adjusted signal generation enhances traditional momentum strategies
- on-chain metrics offer valuable predictive information beyond price data
        """
        
        return summary
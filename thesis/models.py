"""
Machine learning models for Bitcoin prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Dict, Any, Tuple, Optional

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna

from config import CONFIG


class OptunaTuner:
    """hyperparameter optimization using optuna"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['catboost']
        self.study = None
        self.best_params = None
        
    def optimize_catboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series,
                         model_type: str = 'regression') -> Dict[str, Any]:
        """optimize catboost hyperparameters"""
        
        def objective(trial):
            search_space = self.config.optuna_search_space
            
            # get base parameters for model type
            base_params = self.config.get_base_params(model_type)
            
            # optimize only hyperparameters, not loss function
            params = {
                **base_params,
                'iterations': trial.suggest_int('iterations', *search_space['iterations']),
                'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
                'depth': trial.suggest_int('depth', *search_space['depth']),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *search_space['l2_leaf_reg']),
                'border_count': trial.suggest_int('border_count', *search_space['border_count']),
                'bagging_temperature': trial.suggest_float('bagging_temperature', *search_space['bagging_temperature']),
            }
            
            if model_type == 'regression':
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
                y_pred = model.predict(X_test)
                ic = spearmanr(y_test, y_pred).correlation
                return ic if not np.isnan(ic) else -1.0
            else:
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
                y_pred = model.predict(X_test)
                return accuracy_score(y_test, y_pred)
        
        print(f"optimizing {model_type} model ({self.config.n_trials} trials)...")
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed)
        )
        
        self.study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        self.best_params = self.study.best_params
        print(f"best score: {self.study.best_value:.4f}")
        
        return self.best_params

class CatBoostModel:
    """catboost model wrapper with optimization"""
    
    def __init__(self, model_type: str = 'regression', config=None):
        self.model_type = model_type
        self.config = config or CONFIG['catboost']
        self.model = None
        self.tuner = OptunaTuner(config)
        
    def train(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
              y_train: pd.Series, y_test: pd.Series,
              optimize: bool = True) -> None:
        """train catboost model"""
        
        # get model-specific base parameters
        base_params = self.config.get_base_params(self.model_type)
        
        if optimize:
            # hyperparameter optimization
            best_params = self.tuner.optimize_catboost(
                X_train, X_test, y_train, y_test, self.model_type
            )
            # merge base params with optimized params, excluding loss_function
            optimized_params = {k: v for k, v in best_params.items() 
                              if k not in ['loss_function', 'eval_metric']}
            params = {**base_params, **optimized_params}
        else:
            params = base_params
            
        # train final model
        if self.model_type == 'regression':
            self.model = CatBoostRegressor(**params)
        else:
            self.model = CatBoostClassifier(**params)
            
        self.model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """make predictions"""
        if self.model is None:
            raise ValueError("model not trained")
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """evaluate model performance"""
        y_pred = self.predict(X_test)
        
        if self.model_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            ic = spearmanr(y_test, y_pred).correlation
            
            return {
                'rmse': rmse,
                'r2': r2,
                'ic': ic,
                'predictions': y_pred
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            
            # precision@k for classification
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, -1]
                k = min(50, len(y_test))
                top_k_indices = np.argsort(y_prob)[-k:]
                precision_at_k = (y_test.iloc[top_k_indices] == (len(np.unique(y_test)) - 1)).mean()
            else:
                precision_at_k = 0.0
                
            return {
                'accuracy': accuracy,
                'precision_at_k': precision_at_k,
                'predictions': y_pred
            }


class PositionalEncoding(nn.Module):
    """positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class iTransformer(nn.Module):
    """inverted transformer for time series forecasting"""
    
    def __init__(self, input_dim: int, config=None):
        super().__init__()
        self.config = config or CONFIG['transformer']
        
        self.embedding = nn.Linear(input_dim, self.config.d_model)
        self.pos_encoder = PositionalEncoding(
            self.config.d_model, 
            max_len=self.config.max_seq_len
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.n_layers
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        
    def forward(self, x):
        # handle nan inputs
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        
        # check for nan
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        output = self.head(x)
        
        # final nan check
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class TimeSeriesDataset(Dataset):
    """pytorch dataset for time series data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # handle nan values
        self.X = torch.nan_to_num(self.X, nan=0.0)
        self.y = torch.nan_to_num(self.y, nan=0.0)
        
        # clamp extreme values
        self.X = torch.clamp(self.X, min=-1e6, max=1e6)
        self.y = torch.clamp(self.y, min=-1e6, max=1e6)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerTrainer:
    """trainer for transformer models"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG['transformer']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """create sequential datasets for time series modeling"""
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.config.max_seq_len, len(X_clean)):
            X_seq = X_clean.iloc[i - self.config.max_seq_len:i].values
            y_target = y_clean.iloc[i]
            
            # skip if any nan or infinite values
            if (np.isnan(X_seq).any() or np.isnan(y_target) or 
                not np.isfinite(X_seq).all() or not np.isfinite(y_target)):
                continue
                
            X_sequences.append(X_seq)
            y_sequences.append(y_target)
        
        if len(X_sequences) == 0:
            raise ValueError("no valid sequences created")
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_data_loaders(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Tuple[DataLoader, DataLoader]:
        """prepare pytorch data loaders"""
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        print(f"sequence shapes: train {X_train_seq.shape}, test {X_test_seq.shape}")
        
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, model: iTransformer, train_loader: DataLoader,
                   test_loader: DataLoader) -> iTransformer:
        """train transformer model"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=1e-5
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        
        best_ic = -np.inf
        patience_counter = 0
        patience = 10
        
        print(f"training itransformer on {self.device}")
        
        for epoch in range(self.config.epochs):
            # training phase
            model.train()
            train_loss = 0
            valid_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                # check for nan loss
                if torch.isnan(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
            
            if valid_batches == 0:
                print(f"no valid batches at epoch {epoch+1}")
                continue
            
            # validation phase
            model.eval()
            val_predictions, val_targets = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    predictions = model(X_batch).cpu().numpy().flatten()
                    val_predictions.extend(predictions)
                    val_targets.extend(y_batch.numpy().flatten())
            
            # handle nan values
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            
            mask = ~np.isnan(val_predictions) & ~np.isnan(val_targets)
            
            if np.sum(mask) < 5:
                print(f"warning: only {np.sum(mask)} valid samples at epoch {epoch+1}")
                rmse = 999.0
                ic = 0.0
            else:
                val_predictions_clean = val_predictions[mask]
                val_targets_clean = val_targets[mask]
                
                try:
                    rmse = np.sqrt(mean_squared_error(val_targets_clean, val_predictions_clean))
                    ic_result = spearmanr(val_targets_clean, val_predictions_clean)
                    ic = ic_result.correlation if not np.isnan(ic_result.correlation) else 0.0
                except:
                    rmse = 999.0
                    ic = 0.0
            
            scheduler.step(rmse)
            
            if epoch % 10 == 0:
                print(f"epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"loss: {train_loss/max(valid_batches,1):.4f} | "
                      f"rmse: {rmse:.4f} | ic: {ic:.4f}")
            
            # early stopping
            if np.sum(mask) >= 5 and ic > best_ic:
                best_ic = ic
                patience_counter = 0
                torch.save(model.state_dict(), 'best_itransformer.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"early stopping at epoch {epoch+1}")
                    break
        
        # load best model
        try:
            model.load_state_dict(torch.load('best_itransformer.pth'))
            print(f"best spearman ic: {best_ic:.4f}")
        except:
            print("no best model saved, using current")
        
        return model
    
    def evaluate_model(self, model: iTransformer, test_loader: DataLoader) -> Dict[str, Any]:
        """evaluate trained model"""
        model.eval()
        predictions, targets = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch).cpu().numpy().flatten()
                predictions.extend(pred)
                targets.extend(y_batch.numpy().flatten())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # handle nan values
        mask = (~np.isnan(predictions) & ~np.isnan(targets) & 
                np.isfinite(predictions) & np.isfinite(targets))
        
        kept = np.sum(mask)
        print(f"evaluation: {kept}/{len(predictions)} valid samples")
        
        if kept < 5:
            print("too few valid samples for evaluation")
            return {
                'rmse': 999.0,
                'r2': -999.0, 
                'ic': 0.0,
                'predictions': predictions,
                'targets': targets
            }
        
        predictions_clean = predictions[mask]
        targets_clean = targets[mask]
        
        try:
            rmse = np.sqrt(mean_squared_error(targets_clean, predictions_clean))
            r2 = r2_score(targets_clean, predictions_clean)
            ic_result = spearmanr(targets_clean, predictions_clean)
            ic = ic_result.correlation if not np.isnan(ic_result.correlation) else 0.0
        except:
            rmse = 999.0
            r2 = -999.0
            ic = 0.0
        
        print(f"itransformer results:")
        print(f"rmse: {rmse:.4f}")
        print(f"r² score: {r2:.4f}")
        print(f"spearman ic: {ic:.4f}")
        
        return {
            'rmse': rmse, 
            'r2': r2, 
            'ic': ic, 
            'predictions': predictions, 
            'targets': targets
        }
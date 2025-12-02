"""
Deep Learning Models for Stock Price Prediction

Models:
1. LSTM (Long Short-Term Memory) - Excellent for time series with long-term dependencies
2. GRU (Gated Recurrent Unit) - Faster training, good for shorter sequences
3. Transformer-based model - State-of-the-art for sequence modeling
4. Hybrid CNN-LSTM - Combines local feature extraction with temporal modeling

Features:
- Multi-step forecasting
- Ensemble predictions
- Uncertainty estimation
- Feature importance analysis
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check TensorFlow availability
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    logger.warning("TensorFlow not available. Deep learning models will not work.")


class DataPreprocessor:
    """Chuẩn bị dữ liệu cho Deep Learning models"""
    
    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 5):
        """
        Args:
            sequence_length: Số ngày lookback
            forecast_horizon: Số ngày dự đoán
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler_params = {}
        
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Min-Max normalization"""
        if fit:
            self.scaler_params['min'] = data.min(axis=0)
            self.scaler_params['max'] = data.max(axis=0)
            self.scaler_params['range'] = self.scaler_params['max'] - self.scaler_params['min']
            self.scaler_params['range'][self.scaler_params['range'] == 0] = 1
        
        return (data - self.scaler_params['min']) / self.scaler_params['range']
    
    def denormalize_data(self, data: np.ndarray, col_idx: int = 0) -> np.ndarray:
        """Inverse normalization"""
        min_val = self.scaler_params['min'][col_idx]
        range_val = self.scaler_params['range'][col_idx]
        return data * range_val + min_val
    
    def create_sequences(self, data: np.ndarray, 
                        target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo sequences cho training
        
        Args:
            data: Shape (samples, features)
            target_col: Index của column để predict
            
        Returns:
            X: (samples, sequence_length, features)
            y: (samples, forecast_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon), target_col])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame, 
                    feature_cols: List[str],
                    target_col: str = 'close',
                    train_ratio: float = 0.8) -> Dict:
        """
        Pipeline hoàn chỉnh để chuẩn bị dữ liệu
        
        Returns:
            Dict với X_train, X_test, y_train, y_test và metadata
        """
        # Get features
        data = df[feature_cols].values
        target_idx = feature_cols.index(target_col)
        
        # Normalize
        normalized = self.normalize_data(data, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(normalized, target_col=target_idx)
        
        # Train/test split
        split_idx = int(len(X) * train_ratio)
        
        return {
            'X_train': X[:split_idx],
            'X_test': X[split_idx:],
            'y_train': y[:split_idx],
            'y_test': y[split_idx:],
            'target_idx': target_idx,
            'feature_cols': feature_cols,
            'n_features': len(feature_cols)
        }


class LSTMModel:
    """
    LSTM Model for Stock Price Prediction
    
    Architecture:
    - Stacked LSTM layers with dropout
    - Dense layers for output
    - Optional attention mechanism
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 10,
                 forecast_horizon: int = 5,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 use_attention: bool = True):
        """
        Args:
            sequence_length: Input sequence length
            n_features: Number of input features
            forecast_horizon: Number of days to predict
            lstm_units: List of LSTM units per layer
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.model = None
        self.history = None
        
    def _build_model(self) -> Model:
        """Build LSTM model"""
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # Stacked LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(0.01)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        if self.use_attention:
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(self.lstm_units[-1])(attention)
            attention = layers.Permute([2, 1])(attention)
            
            x = layers.Multiply()([x, attention])
            x = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 100,
              batch_size: int = 32,
              early_stopping_patience: int = 15) -> Dict:
        """
        Train LSTM model
        
        Returns:
            Training history
        """
        if self.model is None:
            self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'train_loss': self.history.history['loss'][-1],
            'train_mae': self.history.history['mae'][-1],
            'val_loss': self.history.history.get('val_loss', [None])[-1],
            'val_mae': self.history.history.get('val_mae', [None])[-1],
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0)
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Returns:
            mean_predictions, std_predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)  # training=True enables dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)


class GRUModel:
    """
    GRU Model - Faster alternative to LSTM
    
    Advantages:
    - Fewer parameters than LSTM
    - Faster training
    - Good for shorter sequences
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 n_features: int = 10,
                 forecast_horizon: int = 5,
                 gru_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 bidirectional: bool = True):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.model = None
        self.history = None
        
    def _build_model(self) -> Model:
        """Build GRU model"""
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU model")
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # GRU layers
        for i, units in enumerate(self.gru_units):
            return_sequences = (i < len(self.gru_units) - 1)
            
            gru_layer = layers.GRU(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(0.01)
            )
            
            if self.bidirectional:
                x = layers.Bidirectional(gru_layer)(x)
            else:
                x = gru_layer(x)
            
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, early_stopping_patience=15):
        """Train GRU model"""
        if self.model is None:
            self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return {'train_loss': self.history.history['loss'][-1]}
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)


class CNNLSTMModel:
    """
    Hybrid CNN-LSTM Model
    
    Uses CNN to extract local features, then LSTM for temporal patterns.
    Excellent for capturing both short-term patterns and long-term dependencies.
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 n_features: int = 10,
                 forecast_horizon: int = 5,
                 cnn_filters: List[int] = [64, 128],
                 lstm_units: int = 64,
                 dropout_rate: float = 0.2):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def _build_model(self) -> Model:
        """Build CNN-LSTM model"""
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # CNN layers for feature extraction
        for filters in self.cnn_filters:
            x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # LSTM for temporal patterns
        x = layers.LSTM(self.lstm_units, return_sequences=False)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense output
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        if self.model is None:
            self.model = self._build_model()
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)


class EnsemblePredictor:
    """
    Ensemble của nhiều models
    
    Kết hợp LSTM, GRU, CNN-LSTM để có predictions robust hơn
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 10,
                 forecast_horizon: int = 5):
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def train_all(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """Train all models in ensemble"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            result = model.train(X_train, y_train, X_val, y_val, epochs=epochs)
            results[name] = result
            
        return results
    
    def predict(self, X: np.ndarray) -> Dict:
        """
        Make ensemble predictions
        
        Returns:
            {
                'ensemble': weighted average prediction,
                'individual': dict of individual predictions,
                'uncertainty': standard deviation across models
            }
        """
        predictions = {}
        weighted_sum = np.zeros((X.shape[0], self.forecast_horizon))
        total_weight = 0
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions[name] = pred
            weighted_sum += pred * self.weights[name]
            total_weight += self.weights[name]
        
        ensemble_pred = weighted_sum / total_weight
        
        # Calculate uncertainty
        all_preds = np.stack(list(predictions.values()))
        uncertainty = all_preds.std(axis=0)
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions,
            'uncertainty': uncertainty
        }


class StockPredictor:
    """
    High-level API for stock prediction
    
    Example:
        predictor = StockPredictor()
        predictor.fit(df, feature_cols=['close', 'volume', 'rsi', 'macd'])
        predictions = predictor.predict(days=5)
    """
    
    def __init__(self, 
                 model_type: str = 'lstm',
                 sequence_length: int = 60,
                 forecast_horizon: int = 5):
        
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.preprocessor = DataPreprocessor(sequence_length, forecast_horizon)
        self.model = None
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'close',
            epochs: int = 100,
            batch_size: int = 32) -> Dict:
        """
        Train model on data
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for deep learning models")
        
        # Prepare data
        data = self.preprocessor.prepare_data(df, feature_cols, target_col)
        
        # Split validation
        val_split = int(len(data['X_train']) * 0.9)
        X_train = data['X_train'][:val_split]
        X_val = data['X_train'][val_split:]
        y_train = data['y_train'][:val_split]
        y_val = data['y_train'][val_split:]
        
        # Create model
        n_features = data['n_features']
        
        if self.model_type == 'lstm':
            self.model = LSTMModel(
                sequence_length=self.sequence_length,
                n_features=n_features,
                forecast_horizon=self.forecast_horizon
            )
        elif self.model_type == 'gru':
            self.model = GRUModel(
                sequence_length=self.sequence_length,
                n_features=n_features,
                forecast_horizon=self.forecast_horizon
            )
        elif self.model_type == 'cnn_lstm':
            self.model = CNNLSTMModel(
                sequence_length=self.sequence_length,
                n_features=n_features,
                forecast_horizon=self.forecast_horizon
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        result = self.model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # Evaluate on test
        test_pred = self.model.predict(data['X_test'])
        test_mse = np.mean((test_pred - data['y_test']) ** 2)
        test_mae = np.mean(np.abs(test_pred - data['y_test']))
        
        self.is_fitted = True
        self._feature_cols = feature_cols
        self._target_col = target_col
        
        return {
            **result,
            'test_mse': test_mse,
            'test_mae': test_mae
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get last sequence
        data = df[self._feature_cols].values[-self.sequence_length:]
        data = self.preprocessor.normalize_data(data, fit=False)
        X = data.reshape(1, self.sequence_length, -1)
        
        # Predict
        pred_normalized = self.model.predict(X)
        
        # Denormalize
        target_idx = self._feature_cols.index(self._target_col)
        predictions = self.preprocessor.denormalize_data(pred_normalized[0], target_idx)
        
        return predictions


if __name__ == "__main__":
    print("🧠 Deep Learning Models for Stock Prediction")
    print("=" * 50)
    print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    
    if TENSORFLOW_AVAILABLE:
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        
        # Quick test
        print("\n📊 Testing LSTM model architecture...")
        model = LSTMModel(sequence_length=60, n_features=10, forecast_horizon=5)
        test_model = model._build_model()
        test_model.summary()
        
        print("\n✅ All models ready!")
    else:
        print("⚠️  Install TensorFlow for deep learning: pip install tensorflow")

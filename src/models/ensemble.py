"""
Ensemble Model với Meta-Learning
Kết hợp nhiều mô hình để cải thiện độ chính xác dự đoán
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnsemble:
    """Base class cho Ensemble methods"""
    
    def __init__(self, models: Dict):
        """
        Args:
            models: Dictionary {model_name: model_instance}
        """
        self.models = models
        self.weights = None
        self.meta_model = None
    
    def get_predictions(self, data: pd.DataFrame, 
                       target_col: str = 'close') -> Dict[str, np.ndarray]:
        """
        Lấy predictions từ tất cả models
        
        Returns:
            Dictionary {model_name: predictions}
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    preds = model.predict(data, target_col)
                    predictions[name] = preds
                    logger.info(f"Got predictions from {name}: {len(preds)} values")
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {str(e)}")
        
        return predictions


class SimpleAverageEnsemble(BaseEnsemble):
    """
    Simple Average Ensemble
    Lấy trung bình của tất cả predictions
    
    Ưu điểm:
    - Đơn giản, dễ implement
    - Robust với outliers
    - Không cần training
    
    Nhược điểm:
    - Không tận dụng được strength của từng model
    - Tất cả models có trọng số bằng nhau
    """
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """Dự đoán bằng simple average"""
        predictions_dict = self.get_predictions(data, target_col)
        
        if not predictions_dict:
            raise ValueError("No predictions from models")
        
        # Stack predictions
        all_predictions = np.array(list(predictions_dict.values()))
        
        # Take mean
        ensemble_prediction = np.mean(all_predictions, axis=0)
        
        logger.info(f"Ensemble prediction using {len(predictions_dict)} models")
        return ensemble_prediction


class WeightedAverageEnsemble(BaseEnsemble):
    """
    Weighted Average Ensemble
    Mỗi model có trọng số khác nhau dựa trên performance
    
    Weights được xác định bằng:
    1. Inverse of error metrics (MAE, RMSE)
    2. Optimization algorithms
    3. Manual assignment
    """
    
    def __init__(self, models: Dict, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            models: Dictionary of models
            weights: Dictionary {model_name: weight}. Nếu None, sẽ tự động tính
        """
        super().__init__(models)
        self.weights = weights
    
    def calculate_weights(self, val_data: pd.DataFrame, 
                         target_col: str = 'close') -> Dict[str, float]:
        """
        Tính weights dựa trên validation performance
        
        Sử dụng inverse of MAE:
        weight_i = (1 / MAE_i) / sum(1 / MAE_j)
        """
        predictions_dict = self.get_predictions(val_data, target_col)
        
        if not predictions_dict:
            raise ValueError("No predictions from models")
        
        # Get actual values
        actual = val_data[target_col].values
        
        # Calculate MAE for each model
        errors = {}
        for name, preds in predictions_dict.items():
            # Align lengths
            min_len = min(len(actual), len(preds))
            mae = np.mean(np.abs(actual[:min_len] - preds[:min_len]))
            errors[name] = mae
        
        # Calculate weights (inverse of MAE)
        inverse_errors = {name: 1.0 / error for name, error in errors.items()}
        total = sum(inverse_errors.values())
        weights = {name: inv_err / total for name, inv_err in inverse_errors.items()}
        
        logger.info("Calculated weights:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight:.4f} (MAE: {errors[name]:.4f})")
        
        return weights
    
    def fit(self, val_data: pd.DataFrame, target_col: str = 'close'):
        """Tính weights từ validation data"""
        self.weights = self.calculate_weights(val_data, target_col)
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """Dự đoán bằng weighted average"""
        if self.weights is None:
            raise ValueError("Weights chưa được set. Gọi fit() hoặc truyền weights vào constructor")
        
        predictions_dict = self.get_predictions(data, target_col)
        
        # Weighted sum
        ensemble_prediction = np.zeros(len(list(predictions_dict.values())[0]))
        
        for name, preds in predictions_dict.items():
            if name in self.weights:
                ensemble_prediction += self.weights[name] * preds
        
        return ensemble_prediction


class StackingEnsemble(BaseEnsemble):
    """
    Stacking (Stacked Generalization)
    Sử dụng meta-model để học cách kết hợp predictions
    
    Architecture:
    Level 0: Base models (ARIMA, Prophet, LSTM, GRU)
    Level 1: Meta-model (Linear Regression, Random Forest, etc.)
    
    Quy trình:
    1. Train base models trên training set
    2. Tạo predictions trên validation set
    3. Train meta-model với predictions làm features
    4. Predict: base models -> meta-model -> final prediction
    
    Ưu điểm:
    - Meta-model học được cách tối ưu kết hợp models
    - Thường cho kết quả tốt nhất
    - Linh hoạt với nhiều loại meta-model
    
    Nhược điểm:
    - Phức tạp hơn
    - Cần nhiều dữ liệu hơn
    - Risk of overfitting
    """
    
    def __init__(self, models: Dict, meta_model_type: str = 'linear'):
        """
        Args:
            models: Base models
            meta_model_type: 'linear', 'ridge', 'lasso', 'rf', 'gb', 'mlp'
        """
        super().__init__(models)
        self.meta_model_type = meta_model_type
        self.meta_model = self._create_meta_model(meta_model_type)
    
    def _create_meta_model(self, model_type: str):
        """Tạo meta-model"""
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif model_type == 'lasso':
            return Lasso(alpha=1.0)
        elif model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gb':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'mlp':
            return MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown meta_model_type: {model_type}")
    
    def fit(self, val_data: pd.DataFrame, target_col: str = 'close'):
        """
        Train meta-model
        
        Args:
            val_data: Validation data để train meta-model
            target_col: Target column
        """
        logger.info("Training meta-model for stacking ensemble...")
        
        # Get predictions from base models
        predictions_dict = self.get_predictions(val_data, target_col)
        
        if not predictions_dict:
            raise ValueError("No predictions from base models")
        
        # Create feature matrix (predictions from base models)
        X_meta = []
        for name in sorted(predictions_dict.keys()):
            X_meta.append(predictions_dict[name])
        
        X_meta = np.column_stack(X_meta)
        
        # Get actual values (target for meta-model)
        y_meta = val_data[target_col].values
        
        # Align lengths
        min_len = min(len(y_meta), X_meta.shape[0])
        X_meta = X_meta[:min_len]
        y_meta = y_meta[:min_len]
        
        # Train meta-model
        self.meta_model.fit(X_meta, y_meta)
        
        logger.info(f"Meta-model ({self.meta_model_type}) trained successfully")
        
        # Log feature importances if available
        if hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
            for name, importance in zip(sorted(predictions_dict.keys()), importances):
                logger.info(f"  {name} importance: {importance:.4f}")
        elif hasattr(self.meta_model, 'coef_'):
            coefs = self.meta_model.coef_
            for name, coef in zip(sorted(predictions_dict.keys()), coefs):
                logger.info(f"  {name} coefficient: {coef:.4f}")
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """Dự đoán với stacking ensemble"""
        if self.meta_model is None:
            raise ValueError("Meta-model chưa được train. Gọi fit() trước.")
        
        # Get predictions from base models
        predictions_dict = self.get_predictions(data, target_col)
        
        # Create feature matrix
        X_meta = []
        for name in sorted(predictions_dict.keys()):
            X_meta.append(predictions_dict[name])
        
        X_meta = np.column_stack(X_meta)
        
        # Predict with meta-model
        ensemble_prediction = self.meta_model.predict(X_meta)
        
        return ensemble_prediction


class BlendingEnsemble(BaseEnsemble):
    """
    Blending - Variant của Stacking
    
    Khác với Stacking:
    - Stacking: Sử dụng cross-validation predictions để train meta-model
    - Blending: Sử dụng hold-out validation set
    
    Đơn giản hơn stacking nhưng có thể kém hơn về performance
    """
    
    def __init__(self, models: Dict, meta_model_type: str = 'linear'):
        super().__init__(models)
        self.meta_model_type = meta_model_type
        self.meta_model = self._create_meta_model(meta_model_type)
    
    def _create_meta_model(self, model_type: str):
        """Tạo meta-model (tương tự StackingEnsemble)"""
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif model_type == 'lasso':
            return Lasso(alpha=1.0)
        elif model_type == 'rf':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return LinearRegression()
    
    def fit(self, val_data: pd.DataFrame, target_col: str = 'close'):
        """Train meta-model với blending approach"""
        # Giống StackingEnsemble.fit()
        predictions_dict = self.get_predictions(val_data, target_col)
        
        X_meta = np.column_stack([predictions_dict[name] 
                                  for name in sorted(predictions_dict.keys())])
        y_meta = val_data[target_col].values
        
        min_len = min(len(y_meta), X_meta.shape[0])
        X_meta = X_meta[:min_len]
        y_meta = y_meta[:min_len]
        
        self.meta_model.fit(X_meta, y_meta)
        logger.info(f"Blending meta-model ({self.meta_model_type}) trained")
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """Dự đoán với blending"""
        predictions_dict = self.get_predictions(data, target_col)
        
        X_meta = np.column_stack([predictions_dict[name] 
                                  for name in sorted(predictions_dict.keys())])
        
        ensemble_prediction = self.meta_model.predict(X_meta)
        return ensemble_prediction


class EnsembleFactory:
    """Factory để tạo ensemble models"""
    
    @staticmethod
    def create_ensemble(ensemble_type: str, models: Dict, **kwargs):
        """
        Tạo ensemble model
        
        Args:
            ensemble_type: 'average', 'weighted', 'stacking', 'blending'
            models: Dictionary of base models
            **kwargs: Additional parameters
        
        Returns:
            Ensemble instance
        """
        if ensemble_type == 'average':
            return SimpleAverageEnsemble(models)
        
        elif ensemble_type == 'weighted':
            weights = kwargs.get('weights', None)
            return WeightedAverageEnsemble(models, weights)
        
        elif ensemble_type == 'stacking':
            meta_model_type = kwargs.get('meta_model_type', 'linear')
            return StackingEnsemble(models, meta_model_type)
        
        elif ensemble_type == 'blending':
            meta_model_type = kwargs.get('meta_model_type', 'linear')
            return BlendingEnsemble(models, meta_model_type)
        
        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


def evaluate_ensemble(ensemble, test_data: pd.DataFrame, 
                     target_col: str = 'close') -> Dict[str, float]:
    """
    Đánh giá ensemble model
    
    Returns:
        Dictionary với metrics
    """
    predictions = ensemble.predict(test_data, target_col)
    actual = test_data[target_col].values
    
    # Align lengths
    min_len = min(len(actual), len(predictions))
    actual = actual[:min_len]
    predictions = predictions[:min_len]
    
    # Calculate metrics
    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


if __name__ == "__main__":
    print("Ensemble Methods for Stock Price Prediction")
    print("=" * 60)
    print("\n1. Simple Average:")
    print("   - Equal weights for all models")
    print("   - No training required")
    
    print("\n2. Weighted Average:")
    print("   - Weights based on validation performance")
    print("   - Better models get higher weights")
    
    print("\n3. Stacking (Meta-Learning):")
    print("   - Meta-model learns optimal combination")
    print("   - Usually best performance")
    print("   - More complex, needs more data")
    
    print("\n4. Blending:")
    print("   - Simplified stacking")
    print("   - Uses hold-out validation set")
    
    print("\nRecommended approach:")
    print("1. Train base models: ARIMA, Prophet, LSTM, GRU")
    print("2. Get predictions on validation set")
    print("3. Train stacking ensemble with Ridge/RandomForest meta-model")
    print("4. Evaluate on test set")

"""
ETL Pipeline - Extract, Transform, Load cho Stock Data
Pipeline tự động thu thập, xử lý và lưu trữ dữ liệu
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
import hashlib
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityStatus(Enum):
    """Trạng thái chất lượng dữ liệu"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    MISSING = "missing"


@dataclass
class ValidationResult:
    """Kết quả validation"""
    is_valid: bool
    status: DataQualityStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'status': self.status.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }


@dataclass
class ETLResult:
    """Kết quả của ETL pipeline"""
    success: bool
    symbol: str
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_skipped: int = 0
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0.0
    validation_result: Optional[ValidationResult] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'symbol': self.symbol,
            'records_extracted': self.records_extracted,
            'records_transformed': self.records_transformed,
            'records_loaded': self.records_loaded,
            'records_skipped': self.records_skipped,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': round(self.duration_seconds, 2),
            'validation': self.validation_result.to_dict() if self.validation_result else None,
            'errors': self.errors
        }


class DataValidator:
    """
    Validator cho dữ liệu cổ phiếu
    
    Checks:
    - Missing values
    - Data types
    - Value ranges
    - Duplicates
    - Outliers
    """
    
    REQUIRED_COLUMNS = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self, 
                 max_missing_pct: float = 0.05,
                 min_price: float = 100,
                 max_price: float = 1_000_000,
                 max_change_pct: float = 0.30):  # 30% max daily change
        """
        Args:
            max_missing_pct: Tỷ lệ missing data tối đa cho phép (5%)
            min_price: Giá tối thiểu hợp lệ
            max_price: Giá tối đa hợp lệ
            max_change_pct: % thay đổi giá tối đa trong ngày
        """
        self.max_missing_pct = max_missing_pct
        self.min_price = min_price
        self.max_price = max_price
        self.max_change_pct = max_change_pct
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame
        
        Args:
            df: DataFrame cần validate
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        stats = {}
        
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                status=DataQualityStatus.INVALID,
                errors=["DataFrame is empty or None"]
            )
        
        # 1. Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # 2. Check missing values
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_pct = missing_cells / total_cells if total_cells > 0 else 0
        stats['missing_pct'] = round(missing_pct * 100, 2)
        
        if missing_pct > self.max_missing_pct:
            errors.append(f"Too many missing values: {missing_pct*100:.2f}% > {self.max_missing_pct*100}%")
        elif missing_pct > 0:
            warnings.append(f"Contains {missing_pct*100:.2f}% missing values")
        
        # 3. Check duplicates
        if 'date' in df.columns:
            duplicates = df.duplicated(subset=['date']).sum()
            stats['duplicates'] = duplicates
            if duplicates > 0:
                warnings.append(f"Found {duplicates} duplicate dates")
        
        # 4. Validate price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                # Check range
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val < self.min_price:
                    warnings.append(f"{col} has value {min_val} below minimum {self.min_price}")
                if max_val > self.max_price:
                    warnings.append(f"{col} has value {max_val} above maximum {self.max_price}")
                
                # Check for negative values
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    errors.append(f"{col} contains {negative_count} negative values")
        
        # 5. Validate OHLC relationship
        if all(col in df.columns for col in price_cols):
            invalid_ohlc = (
                (df['High'] < df['Low']) |
                (df['Open'] > df['High']) |
                (df['Open'] < df['Low']) |
                (df['Close'] > df['High']) |
                (df['Close'] < df['Low'])
            ).sum()
            
            stats['invalid_ohlc'] = invalid_ohlc
            if invalid_ohlc > 0:
                warnings.append(f"Found {invalid_ohlc} rows with invalid OHLC relationship")
        
        # 6. Check for outliers (large daily changes)
        if 'Close' in df.columns and len(df) > 1:
            daily_changes = df['Close'].pct_change().abs()
            extreme_changes = (daily_changes > self.max_change_pct).sum()
            stats['extreme_changes'] = extreme_changes
            
            if extreme_changes > 0:
                warnings.append(f"Found {extreme_changes} days with >30% price change (possible data error)")
        
        # 7. Check Volume
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            negative_volume = (df['Volume'] < 0).sum()
            stats['zero_volume_days'] = zero_volume
            
            if negative_volume > 0:
                errors.append(f"Volume contains {negative_volume} negative values")
            if zero_volume > len(df) * 0.1:  # >10% zero volume days
                warnings.append(f"Found {zero_volume} days with zero volume")
        
        # 8. Additional stats
        stats['total_rows'] = len(df)
        stats['date_range'] = {
            'start': df['date'].min().isoformat() if 'date' in df.columns and not df['date'].isna().all() else None,
            'end': df['date'].max().isoformat() if 'date' in df.columns and not df['date'].isna().all() else None
        }
        
        # Determine overall status
        is_valid = len(errors) == 0
        if not is_valid:
            status = DataQualityStatus.INVALID
        elif len(warnings) > 0:
            status = DataQualityStatus.WARNING
        else:
            status = DataQualityStatus.VALID
        
        return ValidationResult(
            is_valid=is_valid,
            status=status,
            errors=errors,
            warnings=warnings,
            stats=stats
        )


class DataTransformer:
    """
    Transformer cho dữ liệu cổ phiếu
    
    Transformations:
    - Clean missing values
    - Remove duplicates
    - Fix OHLC relationships
    - Standardize column names
    - Add derived features
    """
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa tên cột"""
        column_mapping = {
            'DATE': 'date',
            'Date': 'date',
            'OPEN': 'Open',
            'open': 'Open',
            'HIGH': 'High',
            'high': 'High',
            'LOW': 'Low',
            'low': 'Low',
            'CLOSE': 'Close',
            'close': 'Close',
            'VOLUME': 'Volume',
            'volume': 'Volume',
            'VOL': 'Volume',
            'SYMBOL': 'symbol',
            'Symbol': 'symbol'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    @staticmethod
    def clean_missing_values(df: pd.DataFrame, 
                            method: str = 'ffill') -> pd.DataFrame:
        """
        Xử lý missing values
        
        Args:
            df: DataFrame
            method: 'ffill' (forward fill), 'bfill' (backward fill), 
                   'mean', 'median', 'drop'
        """
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        if method == 'ffill':
            df[price_cols] = df[price_cols].ffill()
            df[price_cols] = df[price_cols].bfill()  # Handle first rows
        elif method == 'bfill':
            df[price_cols] = df[price_cols].bfill()
            df[price_cols] = df[price_cols].ffill()
        elif method == 'mean':
            for col in price_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            for col in price_cols:
                df[col] = df[col].fillna(df[col].median())
        elif method == 'drop':
            df = df.dropna(subset=price_cols)
        
        # Fill Volume with 0 if missing
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        return df
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, 
                         keep: str = 'last') -> pd.DataFrame:
        """Loại bỏ duplicate rows"""
        if 'date' in df.columns:
            df = df.drop_duplicates(subset=['date'], keep=keep)
        else:
            df = df.drop_duplicates(keep=keep)
        return df
    
    @staticmethod
    def fix_ohlc_relationship(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sửa OHLC relationship không hợp lệ
        - High phải >= Open, Close, Low
        - Low phải <= Open, Close, High
        """
        df = df.copy()
        
        # Fix High to be max of OHLC
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        
        # Fix Low to be min of OHLC
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return df
    
    @staticmethod
    def convert_date(df: pd.DataFrame) -> pd.DataFrame:
        """Chuyển đổi cột date sang datetime"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    @staticmethod
    def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
        """Sắp xếp theo ngày"""
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        return df
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các features phái sinh cơ bản"""
        df = df.copy()
        
        if 'Close' in df.columns:
            # Daily return
            df['daily_return'] = df['Close'].pct_change()
            
            # Price change
            df['price_change'] = df['Close'].diff()
        
        if all(col in df.columns for col in ['High', 'Low']):
            # Daily range
            df['daily_range'] = df['High'] - df['Low']
            df['daily_range_pct'] = df['daily_range'] / df['Close']
        
        if 'Volume' in df.columns and 'Close' in df.columns:
            # Trading value
            df['trading_value'] = df['Volume'] * df['Close']
        
        return df
    
    @classmethod
    def transform(cls, df: pd.DataFrame, 
                 add_features: bool = True) -> pd.DataFrame:
        """
        Pipeline transform đầy đủ
        
        Args:
            df: DataFrame raw
            add_features: Có thêm derived features không
        
        Returns:
            DataFrame đã transform
        """
        # 1. Standardize columns
        df = cls.standardize_columns(df)
        
        # 2. Convert date
        df = cls.convert_date(df)
        
        # 3. Remove duplicates
        df = cls.remove_duplicates(df)
        
        # 4. Sort by date
        df = cls.sort_by_date(df)
        
        # 5. Clean missing values
        df = cls.clean_missing_values(df)
        
        # 6. Fix OHLC relationship
        df = cls.fix_ohlc_relationship(df)
        
        # 7. Add derived features
        if add_features:
            df = cls.add_derived_features(df)
        
        return df


class Extractor(ABC):
    """Base class cho Extractors"""
    
    @abstractmethod
    def extract(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract data from source"""
        pass


class VNDirectExtractor(Extractor):
    """Extractor cho VNDirect API"""
    
    def __init__(self):
        # Import here to avoid circular dependency
        import sys
        sys.path.append('..')
        try:
            from data_collection import VNDirectAPI
            self.api = VNDirectAPI()
        except:
            self.api = None
    
    def extract(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract data từ VNDirect"""
        if self.api is None:
            logger.error("VNDirectAPI not available")
            return pd.DataFrame()
        
        try:
            df = self.api.get_stock_price(symbol, start_date, end_date)
            logger.info(f"Extracted {len(df)} records for {symbol} from VNDirect")
            return df
        except Exception as e:
            logger.error(f"Error extracting from VNDirect: {str(e)}")
            return pd.DataFrame()


class Loader(ABC):
    """Base class cho Loaders"""
    
    @abstractmethod
    def load(self, df: pd.DataFrame, symbol: str) -> int:
        """Load data to destination, return number of records loaded"""
        pass


class DatabaseLoader(Loader):
    """Loader cho SQLite Database"""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def load(self, df: pd.DataFrame, symbol: str) -> int:
        """Load data vào database"""
        if df.empty:
            return 0
        
        try:
            # Import model
            from database.models import StockPrice
            
            loaded = 0
            for _, row in df.iterrows():
                # Check if exists
                existing = self.db_session.query(StockPrice).filter(
                    StockPrice.symbol == symbol,
                    StockPrice.date == row['date']
                ).first()
                
                if existing:
                    # Update
                    existing.open = row['Open']
                    existing.high = row['High']
                    existing.low = row['Low']
                    existing.close = row['Close']
                    existing.volume = row['Volume']
                else:
                    # Insert
                    price = StockPrice(
                        symbol=symbol,
                        date=row['date'],
                        open=row['Open'],
                        high=row['High'],
                        low=row['Low'],
                        close=row['Close'],
                        volume=row['Volume']
                    )
                    self.db_session.add(price)
                
                loaded += 1
            
            self.db_session.commit()
            logger.info(f"Loaded {loaded} records for {symbol} to database")
            return loaded
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error loading to database: {str(e)}")
            raise


class CSVLoader(Loader):
    """Loader cho CSV file"""
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = output_dir
    
    def load(self, df: pd.DataFrame, symbol: str) -> int:
        """Load data vào CSV file"""
        if df.empty:
            return 0
        
        try:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            filename = f"{self.output_dir}/{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            
            logger.info(f"Loaded {len(df)} records for {symbol} to {filename}")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error loading to CSV: {str(e)}")
            raise


class ETLPipeline:
    """
    ETL Pipeline chính
    
    Flow:
    Extract (from API) -> Validate -> Transform -> Load (to DB/CSV)
    
    Features:
    - Configurable extractors, loaders
    - Data validation với detailed reports
    - Error handling với retry
    - Logging và monitoring
    """
    
    def __init__(self,
                 extractor: Optional[Extractor] = None,
                 loader: Optional[Loader] = None,
                 validator: Optional[DataValidator] = None,
                 transformer: Optional[DataTransformer] = None):
        """
        Args:
            extractor: Data extractor (default: VNDirectExtractor)
            loader: Data loader (default: CSVLoader)
            validator: Data validator
            transformer: Data transformer
        """
        self.extractor = extractor or VNDirectExtractor()
        self.loader = loader or CSVLoader()
        self.validator = validator or DataValidator()
        self.transformer = transformer or DataTransformer()
        
        self.results: List[ETLResult] = []
    
    def run(self, 
            symbol: str, 
            start_date: str, 
            end_date: str,
            validate: bool = True,
            add_features: bool = True,
            retry_count: int = 3,
            retry_delay: float = 5.0) -> ETLResult:
        """
        Chạy ETL pipeline cho một symbol
        
        Args:
            symbol: Mã cổ phiếu
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            validate: Có validate data không
            add_features: Có thêm derived features không
            retry_count: Số lần retry khi lỗi
            retry_delay: Thời gian chờ giữa các lần retry (seconds)
        
        Returns:
            ETLResult
        """
        result = ETLResult(
            success=False,
            symbol=symbol,
            start_time=datetime.now()
        )
        
        logger.info(f"Starting ETL for {symbol}: {start_date} to {end_date}")
        
        try:
            # EXTRACT
            df = None
            for attempt in range(retry_count):
                try:
                    df = self.extractor.extract(symbol, start_date, end_date)
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"Extract attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
            
            if df is None or df.empty:
                result.errors.append("Failed to extract data after all retries")
                return result
            
            result.records_extracted = len(df)
            logger.info(f"Extracted {len(df)} records")
            
            # VALIDATE (pre-transform)
            if validate:
                validation_result = self.validator.validate(df)
                result.validation_result = validation_result
                
                if not validation_result.is_valid:
                    result.errors.extend(validation_result.errors)
                    logger.warning(f"Validation failed: {validation_result.errors}")
                    # Continue with transformation anyway for partial recovery
            
            # TRANSFORM
            df_transformed = self.transformer.transform(df, add_features=add_features)
            result.records_transformed = len(df_transformed)
            result.records_skipped = result.records_extracted - result.records_transformed
            logger.info(f"Transformed {len(df_transformed)} records")
            
            # VALIDATE (post-transform)
            if validate:
                post_validation = self.validator.validate(df_transformed)
                if not post_validation.is_valid:
                    result.errors.extend([f"Post-transform: {e}" for e in post_validation.errors])
            
            # LOAD
            records_loaded = self.loader.load(df_transformed, symbol)
            result.records_loaded = records_loaded
            logger.info(f"Loaded {records_loaded} records")
            
            result.success = True
            
        except Exception as e:
            logger.error(f"ETL pipeline error: {str(e)}")
            result.errors.append(str(e))
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.results.append(result)
        
        return result
    
    def run_batch(self, 
                  symbols: List[str], 
                  start_date: str, 
                  end_date: str,
                  **kwargs) -> List[ETLResult]:
        """
        Chạy ETL cho nhiều symbols
        
        Args:
            symbols: Danh sách mã cổ phiếu
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            **kwargs: Các tham số khác cho run()
        
        Returns:
            List[ETLResult]
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {i+1}/{len(symbols)}: {symbol}")
            result = self.run(symbol, start_date, end_date, **kwargs)
            results.append(result)
            
            # Rate limiting
            if i < len(symbols) - 1:
                time.sleep(1)
        
        # Summary
        success_count = sum(1 for r in results if r.success)
        total_records = sum(r.records_loaded for r in results)
        
        logger.info(f"Batch ETL complete: {success_count}/{len(symbols)} successful, "
                   f"{total_records} total records loaded")
        
        return results
    
    def get_summary(self) -> Dict:
        """Lấy summary của tất cả ETL runs"""
        if not self.results:
            return {'message': 'No ETL runs yet'}
        
        return {
            'total_runs': len(self.results),
            'successful': sum(1 for r in self.results if r.success),
            'failed': sum(1 for r in self.results if not r.success),
            'total_extracted': sum(r.records_extracted for r in self.results),
            'total_transformed': sum(r.records_transformed for r in self.results),
            'total_loaded': sum(r.records_loaded for r in self.results),
            'total_skipped': sum(r.records_skipped for r in self.results),
            'avg_duration_seconds': np.mean([r.duration_seconds for r in self.results]),
            'symbols': [r.symbol for r in self.results]
        }


class IncrementalETL:
    """
    Incremental ETL - Chỉ load dữ liệu mới
    Tối ưu cho cập nhật hàng ngày
    """
    
    def __init__(self, pipeline: ETLPipeline, db_session=None):
        self.pipeline = pipeline
        self.db_session = db_session
    
    def get_last_date(self, symbol: str) -> Optional[datetime]:
        """Lấy ngày cuối cùng có data trong DB"""
        if self.db_session is None:
            return None
        
        try:
            from database.models import StockPrice
            
            result = self.db_session.query(StockPrice.date).filter(
                StockPrice.symbol == symbol
            ).order_by(StockPrice.date.desc()).first()
            
            return result[0] if result else None
        except:
            return None
    
    def run_incremental(self, symbol: str) -> ETLResult:
        """Chỉ load dữ liệu mới từ ngày cuối cùng"""
        last_date = self.get_last_date(symbol)
        
        if last_date:
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Nếu chưa có data, lấy 1 năm
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Incremental ETL for {symbol}: {start_date} to {end_date}")
        
        return self.pipeline.run(symbol, start_date, end_date)


# Quick ETL functions for API
def run_etl_for_symbol(symbol: str, 
                       start_date: str = None, 
                       end_date: str = None,
                       db_session=None) -> Dict:
    """
    Quick ETL function cho một symbol
    
    Args:
        symbol: Mã cổ phiếu
        start_date: Ngày bắt đầu (mặc định 1 năm trước)
        end_date: Ngày kết thúc (mặc định hôm nay)
        db_session: Database session (optional)
    
    Returns:
        Dict kết quả ETL
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Setup loader
    if db_session:
        loader = DatabaseLoader(db_session)
    else:
        loader = CSVLoader()
    
    # Run pipeline
    pipeline = ETLPipeline(loader=loader)
    result = pipeline.run(symbol, start_date, end_date)
    
    return result.to_dict()


def run_batch_etl(symbols: List[str],
                  start_date: str = None,
                  end_date: str = None,
                  db_session=None) -> Dict:
    """
    Run ETL cho nhiều symbols
    
    Returns:
        Dict với summary và chi tiết từng symbol
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Setup loader
    if db_session:
        loader = DatabaseLoader(db_session)
    else:
        loader = CSVLoader()
    
    # Run pipeline
    pipeline = ETLPipeline(loader=loader)
    results = pipeline.run_batch(symbols, start_date, end_date)
    
    return {
        'summary': pipeline.get_summary(),
        'details': [r.to_dict() for r in results]
    }


if __name__ == "__main__":
    print("ETL Pipeline for Stock Data")
    print("=" * 60)
    print("\nPipeline Steps:")
    print("1. EXTRACT: Fetch data from VNDirect API")
    print("2. VALIDATE: Check data quality (missing, duplicates, outliers)")
    print("3. TRANSFORM: Clean, standardize, add features")
    print("4. LOAD: Save to database or CSV")
    print("\nUsage:")
    print("  pipeline = ETLPipeline()")
    print("  result = pipeline.run('VNM', '2024-01-01', '2024-12-01')")
    print("  print(result.to_dict())")

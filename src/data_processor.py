"""
Data Processing Module for Breast Cancer Classification

Handles data loading, cleaning, preprocessing, outlier detection,
and feature engineering for breast cancer diagnosis data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Comprehensive data processor for breast cancer dataset.
    
    Handles all preprocessing steps including:
    - Data loading and initial cleaning
    - Outlier detection and treatment
    - Feature scaling
    - Feature selection based on correlation
    - Train/test splitting
    
    Attributes:
        df: Main DataFrame
        scaler: Fitted scaler object
        label_encoder: Fitted label encoder
        
    Example:
        >>> processor = DataProcessor()
        >>> processor.load_data('data.csv')
        >>> processor.clean_data()
        >>> processor.handle_outliers()
        >>> X_train, X_test, y_train, y_test = processor.prepare_for_training()
    """
    
    # Columns to drop (known useless columns)
    DROP_COLUMNS = ['id', 'Unnamed: 32']
    
    # Feature groups for analysis
    FEATURE_GROUPS = {
        'mean': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'],
        'se': ['radius_se', 'texture_se', 'perimeter_se', 'area_se',
               'smoothness_se', 'compactness_se', 'concavity_se',
               'concave points_se', 'symmetry_se', 'fractal_dimension_se'],
        'worst': ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                  'smoothness_worst', 'compactness_worst', 'concavity_worst',
                  'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    }
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.feature_columns: List[str] = []
        self.target_column: str = 'diagnosis'
    
    # ==================== DATA LOADING ====================
    
    def load_data(self, filepath: Union[str, Path]) -> 'DataProcessor':
        """
        Load breast cancer data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            self for method chaining
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.df = pd.read_csv(filepath)
        self.original_df = self.df.copy()
        
        print(f"[OK] Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        return self
    
    def load_from_sklearn(self) -> 'DataProcessor':
        """
        Load breast cancer dataset from sklearn.
        
        Returns:
            self for method chaining
        """
        from sklearn.datasets import load_breast_cancer
        
        data = load_breast_cancer()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.df['diagnosis'] = data.target
        self.original_df = self.df.copy()
        
        print(f"[OK] Loaded {len(self.df)} samples from sklearn")
        return self
    
    # ==================== DATA CLEANING ====================
    
    def clean_data(self) -> 'DataProcessor':
        """
        Clean the dataset by removing unnecessary columns and handling missing values.
        
        Returns:
            self for method chaining
        """
        initial_cols = len(self.df.columns)
        
        # Drop unnecessary columns
        for col in self.DROP_COLUMNS:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)
        
        # Handle missing values
        missing_before = self.df.isnull().sum().sum()
        self.df.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN
        self.df.dropna(inplace=True)  # Drop rows with any NaN
        
        # Encode target variable if it's categorical
        if self.df[self.target_column].dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.df[self.target_column] = self.label_encoder.fit_transform(self.df[self.target_column])
            print(f"[OK] Encoded target: M=1 (Malignant), B=0 (Benign)")
        
        final_cols = len(self.df.columns)
        print(f"[OK] Cleaned data: {initial_cols} -> {final_cols} columns")
        
        return self
    
    def get_info(self) -> pd.DataFrame:
        """Get detailed information about the dataset."""
        info = []
        for col in self.df.columns:
            info.append({
                'Column': col,
                'Type': str(self.df[col].dtype),
                'Non-Null': self.df[col].notna().sum(),
                'Unique': self.df[col].nunique(),
                'Mean': f"{self.df[col].mean():.4f}" if self.df[col].dtype in ['int64', 'float64'] else 'N/A'
            })
        return pd.DataFrame(info)
    
    def get_summary(self) -> Dict:
        """Get a summary of the dataset."""
        target_counts = self.df[self.target_column].value_counts()
        return {
            'Total Samples': len(self.df),
            'Features': len(self.df.columns) - 1,
            'Malignant (1)': target_counts.get(1, 0),
            'Benign (0)': target_counts.get(0, 0),
            'Class Balance': f"{target_counts.get(1, 0) / len(self.df) * 100:.1f}% Malignant"
        }
    
    # ==================== OUTLIER HANDLING ====================
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in numerical columns.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary mapping column names to lists of outlier indices
        """
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target_column]
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            else:  # zscore
                mean = self.df[col].mean()
                std = self.df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers[col] = self.df[mask].index.tolist()
        
        total_outliers = sum(len(v) for v in outliers.values())
        print(f"[OK] Detected {total_outliers} outlier points across {len(outliers)} columns")
        
        return outliers
    
    def handle_outliers(self, method: str = 'clip') -> 'DataProcessor':
        """
        Handle outliers in the dataset.
        
        Args:
            method: How to handle outliers ('clip', 'remove', or 'median')
            
        Returns:
            self for method chaining
        """
        if not self.outlier_bounds:
            self.detect_outliers()
        
        numeric_cols = [c for c in self.outlier_bounds.keys()]
        
        if method == 'clip':
            # Clip outliers to bounds
            for col in numeric_cols:
                lower, upper = self.outlier_bounds[col]
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            print(f"[OK] Clipped outliers to IQR bounds")
            
        elif method == 'remove':
            # Remove rows with outliers
            mask = pd.Series([True] * len(self.df))
            for col in numeric_cols:
                lower, upper = self.outlier_bounds[col]
                mask &= (self.df[col] >= lower) & (self.df[col] <= upper)
            original_len = len(self.df)
            self.df = self.df[mask].reset_index(drop=True)
            print(f"[OK] Removed {original_len - len(self.df)} rows with outliers")
            
        elif method == 'median':
            # Replace outliers with median
            for col in numeric_cols:
                lower, upper = self.outlier_bounds[col]
                median = self.df[col].median()
                mask = (self.df[col] < lower) | (self.df[col] > upper)
                self.df.loc[mask, col] = median
            print(f"[OK] Replaced outliers with median values")
        
        return self
    
    # ==================== FEATURE ENGINEERING ====================
    
    def scale_features(self, method: str = 'minmax', 
                       exclude_target: bool = True) -> 'DataProcessor':
        """
        Scale numerical features.
        
        Args:
            method: Scaling method ('minmax' or 'standard')
            exclude_target: Whether to exclude target column from scaling
            
        Returns:
            self for method chaining
        """
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        # Get columns to scale
        cols_to_scale = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if exclude_target and self.target_column in cols_to_scale:
            cols_to_scale.remove(self.target_column)
        
        # Scale
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        
        print(f"[OK] Scaled {len(cols_to_scale)} features using {method}")
        return self
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for all features."""
        return self.df.corr()
    
    def get_highly_correlated_features(self, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features.
        
        Args:
            threshold: Correlation threshold
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix = self.df.corr().abs()
        pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    def remove_correlated_features(self, threshold: float = 0.9, 
                                    keep_first: bool = True) -> 'DataProcessor':
        """
        Remove highly correlated features.
        
        Args:
            threshold: Correlation threshold
            keep_first: Keep first feature of correlated pair
            
        Returns:
            self for method chaining
        """
        pairs = self.get_highly_correlated_features(threshold)
        to_drop = set()
        
        for f1, f2, corr in pairs:
            if f1 not in to_drop and f2 not in to_drop:
                drop_col = f2 if keep_first else f1
                if drop_col != self.target_column:
                    to_drop.add(drop_col)
        
        if to_drop:
            self.df.drop(columns=list(to_drop), inplace=True)
            print(f"[OK] Removed {len(to_drop)} highly correlated features")
        
        return self
    
    def select_features(self, columns_to_drop: List[str]) -> 'DataProcessor':
        """
        Manually select features by dropping specified columns.
        
        Args:
            columns_to_drop: List of column names to drop
            
        Returns:
            self for method chaining
        """
        existing = [c for c in columns_to_drop if c in self.df.columns]
        self.df.drop(columns=existing, inplace=True)
        print(f"[OK] Dropped {len(existing)} columns")
        return self
    
    # ==================== TRAIN/TEST PREPARATION ====================
    
    def prepare_for_training(self, test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                               np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        self.feature_columns = [c for c in self.df.columns if c != self.target_column]
        
        X = self.df[self.feature_columns].values
        y = self.df[self.target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"[OK] Split data: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns if self.feature_columns else \
               [c for c in self.df.columns if c != self.target_column]
    
    # ==================== UTILITY ====================
    
    def reset(self) -> 'DataProcessor':
        """Reset to original data."""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.outlier_bounds = {}
            print("[OK] Reset to original data")
        return self
    
    def __repr__(self) -> str:
        if self.df is None:
            return "DataProcessor(no data loaded)"
        return f"DataProcessor({len(self.df)} samples, {len(self.df.columns)} features)"
    
    def __len__(self) -> int:
        return len(self.df) if self.df is not None else 0

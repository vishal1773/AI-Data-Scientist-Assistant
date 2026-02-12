

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           mean_squared_error, mean_absolute_error, r2_score, classification_report)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistics
from scipy import stats
import joblib
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure page
st.set_page_config(
    page_title="AI Data Scientist Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedDataAnalyzer:
    """Comprehensive data analysis and profiling system"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        
    def create_comprehensive_profile(self, target_col: str = None) -> Dict[str, Any]:
        """Create detailed dataset profile with ML insights"""
        # Remove target from feature lists
        numeric_features = [col for col in self.numeric_cols if col != target_col]
        categorical_features = [col for col in self.categorical_cols if col != target_col]
        
        profile = {
            'n_rows': len(self.data),
            'n_cols': len(self.data.columns),
            'numeric_cols': numeric_features,
            'categorical_cols': categorical_features,
            'missing_percentage': (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            'duplicates_count': self.data.duplicated().sum(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024 * 1024),
            'target_type': None,
            'class_imbalance_ratio': None,
            'feature_correlations': None,
            'high_cardinality_cols': []
        }
        
        # Analyze target variable
        if target_col and target_col in self.data.columns:
            target_unique = self.data[target_col].nunique()
            
            if self.data[target_col].dtype in ['int64', 'float64']:
                if target_unique <= 20:
                    profile['target_type'] = 'classification'
                    value_counts = self.data[target_col].value_counts()
                    profile['class_imbalance_ratio'] = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1
                else:
                    profile['target_type'] = 'regression'
            else:
                profile['target_type'] = 'classification'
                value_counts = self.data[target_col].value_counts()
                profile['class_imbalance_ratio'] = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1
        
        # High cardinality analysis
        for col in categorical_features:
            if self.data[col].nunique() > 50:
                profile['high_cardinality_cols'].append(col)
        
        # Feature correlations
        if len(numeric_features) > 1:
            profile['feature_correlations'] = self.data[numeric_features].corr()
        
        return profile
    
    def detect_data_quality_issues(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        issues = {
            'missing_values': {},
            'outliers': {},
            'duplicates': self.data.duplicated().sum(),
            'constant_features': [],
            'highly_correlated_pairs': [],
            'data_types_issues': []
        }
        
        # Missing values analysis
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                issues['missing_values'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(self.data)) * 100
                }
        
        # Outlier detection for numeric columns
        for col in self.numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    issues['outliers'][col] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / len(self.data)) * 100
                    }
        
        # Constant features
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                issues['constant_features'].append(col)
        
        # Highly correlated pairs
        if len(self.numeric_cols) > 1:
            corr_matrix = self.data[self.numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.95:
                        issues['highly_correlated_pairs'].append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
        
        return issues


class SmartPreprocessor:
    """Intelligent preprocessing with automatic decision making"""
    
    def __init__(self):
        self.transformers = {}
        self.preprocessing_steps = []
        
    def create_preprocessing_pipeline(self, data: pd.DataFrame, target_col: str, 
                                    profile: Dict[str, Any]) -> pd.DataFrame:
        """Create and apply intelligent preprocessing pipeline"""
        processed_data = data.copy()
        
        # Step 1: Handle missing values
        processed_data = self._handle_missing_values_intelligently(processed_data, profile)
        
        # Step 2: Handle outliers
        processed_data = self._handle_outliers_smartly(processed_data, target_col)
        
        # Step 3: Encode categorical variables
        processed_data = self._encode_categorical_intelligently(processed_data, target_col, profile)
        
        # Step 4: Feature scaling
        processed_data = self._apply_smart_scaling(processed_data, target_col)
        
        # Step 5: Feature engineering
        processed_data = self._engineer_features(processed_data, profile)
        
        return processed_data
    
    def _handle_missing_values_intelligently(self, data: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        """Smart missing value handling based on data characteristics"""
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            
            if missing_pct > 0:
                if missing_pct > 70:
                    # Drop columns with too many missing values
                    data = data.drop(col, axis=1)
                    self.preprocessing_steps.append(f"Dropped {col} (>{missing_pct:.1f}% missing)")
                
                elif data[col].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if missing_pct > 30:
                        # Use median for high missing percentage
                        imputer = SimpleImputer(strategy='median')
                    else:
                        # Use mean for low missing percentage
                        imputer = SimpleImputer(strategy='mean')
                    
                    data[col] = imputer.fit_transform(data[[col]]).ravel()
                    self.transformers[f'{col}_imputer'] = imputer
                    self.preprocessing_steps.append(f"Imputed {col} using {imputer.strategy}")
                
                else:
                    # Categorical columns
                    imputer = SimpleImputer(strategy='most_frequent')
                    data[col] = imputer.fit_transform(data[[col]]).ravel()
                    self.transformers[f'{col}_imputer'] = imputer
                    self.preprocessing_steps.append(f"Imputed {col} using mode")
        
        return data
    
    def _handle_outliers_smartly(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Smart outlier handling with capping"""
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_col]
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                outliers_before = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers_before > 0:
                    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    self.preprocessing_steps.append(f"Capped {outliers_before} outliers in {col}")
        
        return data
    
    def _encode_categorical_intelligently(self, data: pd.DataFrame, target_col: str, 
                                        profile: Dict[str, Any]) -> pd.DataFrame:
        """Smart categorical encoding based on cardinality and target relationship"""
        categorical_cols = [col for col in data.select_dtypes(include=['object', 'category']).columns 
                          if col != target_col]
        
        for col in categorical_cols:
            unique_count = data[col].nunique()
            
            if unique_count <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
                self.preprocessing_steps.append(f"One-hot encoded {col} ({unique_count} categories)")
            
            elif unique_count <= 50:
                # Label encoding for medium cardinality
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))
                self.transformers[f'{col}_encoder'] = encoder
                self.preprocessing_steps.append(f"Label encoded {col} ({unique_count} categories)")
            
            else:
                # For very high cardinality, consider frequency encoding
                freq_encoding = data[col].value_counts().to_dict()
                data[col] = data[col].map(freq_encoding)
                self.transformers[f'{col}_freq_encoder'] = freq_encoding
                self.preprocessing_steps.append(f"Frequency encoded {col} ({unique_count} categories)")
        
        return data
    
    def _apply_smart_scaling(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply feature scaling to numeric columns"""
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
        
        if numeric_cols:
            scaler = StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            self.transformers['scaler'] = scaler
            self.preprocessing_steps.append(f"Standard scaled {len(numeric_cols)} numeric features")
            
        return data
    
    def _engineer_features(self, data: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        """Create engineered features based on data characteristics"""
        # Create polynomial features for small datasets with few features
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns]
        
        if len(data) < 5000 and len(numeric_cols) <= 5 and len(numeric_cols) >= 2:
            # Create interaction terms
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
            
            self.preprocessing_steps.append("Created interaction features")
        
        return data


class MLModelRecommendationEngine:
    """Advanced ML model recommendation system"""
    
    def __init__(self, profile: Dict[str, Any], data_quality: Dict[str, Any]):
        self.profile = profile
        self.data_quality = data_quality
        
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive ML recommendations"""
        problem_type = self.profile.get('target_type', 'classification')
        n_rows = self.profile.get('n_rows', 0)
        n_features = len(self.profile.get('numeric_cols', [])) + len(self.profile.get('categorical_cols', []))
        missing_pct = self.profile.get('missing_percentage', 0)
        
        recommendation = {
            'problem_type': problem_type,
            'recommended_models': [],
            'model_explanations': {},
            'preprocessing_strategy': [],
            'evaluation_strategy': {},
            'performance_expectations': {},
            'next_steps': []
        }
        
        # Model recommendations based on comprehensive analysis
        if problem_type == 'classification':
            if n_rows < 1000:
                models = ['RandomForest', 'SVM', 'KNN']
                explanation = "Small dataset: Robust models that work well with limited data"
            elif n_rows > 100000:
                models = ['RandomForest', 'LogisticRegression', 'DecisionTree']
                explanation = "Large dataset: Scalable models for efficient training"
            else:
                models = ['RandomForest', 'SVM', 'LogisticRegression']
                explanation = "Medium dataset: Ensemble and linear models for balanced performance"
            
            # Adjust based on feature count
            if n_features > n_rows * 0.1:
                models = ['LogisticRegression', 'SVM'] + models
                recommendation['next_steps'].append("Consider feature selection due to high feature-to-sample ratio")
            
            recommendation['evaluation_strategy'] = {
                'primary_metric': 'f1_score',
                'cross_validation': 'StratifiedKFold',
                'additional_metrics': ['accuracy', 'precision', 'recall']
            }
            
        else:  # regression
            if n_rows < 1000:
                models = ['RandomForest', 'SVR', 'KNN']
                explanation = "Small dataset: Non-parametric models for robust prediction"
            elif n_rows > 100000:
                models = ['LinearRegression', 'RandomForest', 'DecisionTree']
                explanation = "Large dataset: Efficient models for scalable training"
            else:
                models = ['RandomForest', 'Ridge', 'SVR']
                explanation = "Medium dataset: Ensemble and regularized models"
            
            recommendation['evaluation_strategy'] = {
                'primary_metric': 'r2_score',
                'cross_validation': 'KFold',
                'additional_metrics': ['mean_squared_error', 'mean_absolute_error']
            }
        
        recommendation['recommended_models'] = models
        recommendation['model_explanations']['general'] = explanation
        
        # Preprocessing strategy
        strategy = ['missing_value_imputation', 'outlier_handling']
        
        if len(self.profile.get('categorical_cols', [])) > 0:
            strategy.append('categorical_encoding')
        
        if len(self.profile.get('numeric_cols', [])) > 1:
            strategy.append('feature_scaling')
        
        if missing_pct > 20:
            strategy.append('advanced_imputation')
            recommendation['next_steps'].append("Consider advanced imputation techniques")
        
        recommendation['preprocessing_strategy'] = strategy
        
        # Performance expectations
        if n_rows < 500:
            recommendation['performance_expectations'] = {
                'training_time': 'Fast (< 1 minute)',
                'expected_accuracy': 'Moderate (limited data)',
                'overfitting_risk': 'High'
            }
        elif n_rows > 50000:
            recommendation['performance_expectations'] = {
                'training_time': 'Moderate (1-5 minutes)',
                'expected_accuracy': 'High (sufficient data)',
                'overfitting_risk': 'Low'
            }
        else:
            recommendation['performance_expectations'] = {
                'training_time': 'Fast (1-2 minutes)',
                'expected_accuracy': 'Good',
                'overfitting_risk': 'Moderate'
            }
        
        # Additional recommendations
        class_imbalance_ratio = self.profile.get('class_imbalance_ratio') or 1
        if class_imbalance_ratio > 3:
            recommendation['next_steps'].append("Apply class balancing techniques (SMOTE)")
        
        if len(self.data_quality.get('highly_correlated_pairs', [])) > 0:
            recommendation['next_steps'].append("Consider removing highly correlated features")
        
        return recommendation


class ModelTrainingEngine:
    """Comprehensive model training and evaluation system"""
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        
    def get_model_instances(self, problem_type: str) -> Dict[str, Any]:
        """Get optimized model instances"""
        if problem_type == 'classification':
            return {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    min_samples_split=5,
                    random_state=42, 
                    n_jobs=-1
                ),
                'LogisticRegression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    solver='liblinear'
                ),
                'SVM': SVC(
                    random_state=42, 
                    probability=True,
                    kernel='rbf'
                ),
                'DecisionTree': DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                ),
                'KNN': KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance'
                ),
                'NaiveBayes': GaussianNB()
            }
        else:
            return {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    min_samples_split=5,
                    random_state=42, 
                    n_jobs=-1
                ),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(
                    random_state=42,
                    alpha=1.0
                ),
                'SVR': SVR(
                    kernel='rbf',
                    gamma='scale'
                ),
                'DecisionTree': DecisionTreeRegressor(
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                ),
                'KNN': KNeighborsRegressor(
                    n_neighbors=5,
                    weights='distance'
                )
            }
    
    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series, 
                                 recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model training and evaluation"""
        problem_type = recommendation['problem_type']
        
        # Adaptive train-test split
        test_size = min(0.3, max(0.1, 500 / len(X)))
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if problem_type == 'classification' and len(np.unique(y)) > 1 else None
            )
        except:
            # Fallback if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Get models
        available_models = self.get_model_instances(problem_type)
        recommended_models = recommendation['recommended_models']
        
        results = {}
        
        for model_name in recommended_models:
            if model_name in available_models:
                try:
                    model = available_models[model_name]
                    
                    # Training
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    metrics = self._calculate_comprehensive_metrics(y_test, y_pred, problem_type)
                    
                    # Cross-validation
                    cv_folds = min(5, len(X_train) // 20, 10)
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=cv_folds,
                        scoring='f1_weighted' if problem_type == 'classification' else 'r2'
                    )
                    
                    # Feature importance (if available)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(X.columns, abs(model.coef_.flatten())))
                    
                    results[model_name] = {
                        'model': model,
                        'metrics': metrics,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'training_time': training_time,
                        'feature_importance': feature_importance,
                        'predictions': y_pred.tolist(),
                        'model_params': model.get_params()
                    }
                    
                except Exception as e:
                    results[model_name] = {
                        'error': str(e),
                        'model_name': model_name
                    }
        
        return results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add macro averages for multiclass
            if len(np.unique(y_true)) > 2:
                metrics.update({
                    'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                    'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
                })
        
        else:  # regression
            mse = mean_squared_error(y_true, y_pred)
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'mean_squared_error': mse,
                'root_mean_squared_error': np.sqrt(mse),
                'mean_absolute_error': mean_absolute_error(y_true, y_pred)
            }
            
            # Add relative metrics
            if np.mean(y_true) != 0:
                metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return metrics


class AdvancedVisualizationEngine:
    """Comprehensive visualization system for ML insights"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def create_eda_suite(self, profile: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive EDA visualization suite"""
        figures = {}
        
        # 1. Dataset Overview Dashboard
        figures['overview'] = self._create_overview_dashboard(profile)
        
        # 2. Missing Values Analysis
        figures['missing_analysis'] = self._create_missing_values_analysis()
        
        # 3. Feature Distributions
        numeric_cols = profile.get('numeric_cols', [])
        if numeric_cols:
            figures['distributions'] = self._create_distribution_analysis(numeric_cols[:8])
        
        # 4. Correlation Analysis
        if len(numeric_cols) > 1:
            figures['correlations'] = self._create_correlation_analysis(numeric_cols)
        
        # 5. Categorical Analysis
        categorical_cols = profile.get('categorical_cols', [])
        if categorical_cols:
            figures['categorical'] = self._create_categorical_analysis(categorical_cols[:6])
        
        # 6. Outlier Detection Visualization
        if numeric_cols:
            figures['outliers'] = self._create_outlier_analysis(numeric_cols[:6])
        
        return figures
    
    def _create_overview_dashboard(self, profile: Dict[str, Any]) -> go.Figure:
        """Create comprehensive overview dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Data Types Distribution', 'Missing Values', 'Memory Usage',
                'Dataset Statistics', 'Feature Types', 'Data Quality Score'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "pie"}, {"type": "indicator"}]
            ]
        )
        
        # Data types pie chart
        dtype_counts = self.data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(
                labels=dtype_counts.index.astype(str),
                values=dtype_counts.values,
                name="Data Types"
            ),
            row=1, col=1
        )
        
        # Missing values bar chart
        missing_counts = self.data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            fig.add_trace(
                go.Bar(
                    x=missing_counts.index,
                    y=missing_counts.values,
                    name="Missing Values",
                    marker_color='red'
                ),
                row=1, col=2
            )
        
        # Memory usage
        memory_usage = self.data.memory_usage(deep=True) / 1024
        fig.add_trace(
            go.Bar(
                x=memory_usage.index,
                y=memory_usage.values,
                name="Memory (KB)",
                marker_color='blue'
            ),
            row=1, col=3
        )
        
        # Statistics table
        stats_data = [
            ['Total Rows', f"{profile['n_rows']:,}"],
            ['Total Columns', profile['n_cols']],
            ['Numeric Features', len(profile['numeric_cols'])],
            ['Categorical Features', len(profile['categorical_cols'])],
            ['Missing Data %', f"{profile['missing_percentage']:.2f}%"],
            ['Duplicate Rows', profile['duplicates_count']],
            ['Memory Usage', f"{profile['memory_usage_mb']:.2f} MB"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=list(zip(*stats_data)), fill_color='white')
            ),
            row=2, col=1
        )
        
        # Feature types pie
        feature_types = {
            'Numeric': len(profile['numeric_cols']),
            'Categorical': len(profile['categorical_cols'])
        }
        fig.add_trace(
            go.Pie(
                labels=list(feature_types.keys()),
                values=list(feature_types.values()),
                name="Feature Types"
            ),
            row=2, col=2
        )
        
        # Data quality indicator
        quality_score = self._calculate_data_quality_score(profile)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Dataset Analysis",
            showlegend=False
        )
        
        return fig
    
    def _calculate_data_quality_score(self, profile: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        score = 100
        
        # Penalize missing values
        missing_penalty = min(profile['missing_percentage'] * 2, 40)
        score -= missing_penalty
        
        # Penalize duplicates
        duplicate_penalty = min((profile['duplicates_count'] / profile['n_rows']) * 30, 20)
        score -= duplicate_penalty
        
        # Bonus for balanced features
        if len(profile['numeric_cols']) > 0 and len(profile['categorical_cols']) > 0:
            score += 5
        
        return max(score, 0)
    
    def _create_missing_values_analysis(self) -> go.Figure:
        """Create detailed missing values analysis"""
        missing_data = self.data.isnull().sum().sort_values(ascending=False)
        missing_percentages = (missing_data / len(self.data) * 100).round(2)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Missing Values Count', 'Missing Values Percentage'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Count
        fig.add_trace(
            go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                name="Count",
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Percentage
        fig.add_trace(
            go.Bar(
                x=missing_percentages.index,
                y=missing_percentages.values,
                name="Percentage",
                marker_color='orange'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Missing Values Analysis",
            showlegend=False
        )
        
        return fig
    
    def _create_distribution_analysis(self, numeric_cols: List[str]) -> go.Figure:
        """Create distribution analysis for numeric columns"""
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=self.data[col],
                    name=col,
                    showlegend=False,
                    nbinsx=30
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Feature Distributions",
            showlegend=False
        )
        
        return fig
    
    def _create_correlation_analysis(self, numeric_cols: List[str]) -> go.Figure:
        """Create correlation analysis"""
        corr_matrix = self.data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            width=700,
            height=600
        )
        
        return fig
    
    def _create_categorical_analysis(self, categorical_cols: List[str]) -> go.Figure:
        """Create categorical features analysis"""
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=categorical_cols
        )
        
        for i, col in enumerate(categorical_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            value_counts = self.data[col].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=col,
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Categorical Features Analysis",
            showlegend=False
        )
        
        return fig
    
    def _create_outlier_analysis(self, numeric_cols: List[str]) -> go.Figure:
        """Create outlier analysis using box plots"""
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Box(
                    y=self.data[col],
                    name=col,
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Outlier Detection (Box Plots)",
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive model performance dashboard"""
        # Extract data for visualization
        model_names = []
        metrics_data = {}
        
        for model_name, result in results.items():
            if 'error' not in result and 'metrics' in result:
                model_names.append(model_name)
                for metric, value in result['metrics'].items():
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
        
        if not model_names:
            return go.Figure()
        
        # Create subplots
        n_metrics = len(metrics_data)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=list(metrics_data.keys())
        )
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            row = i // n_cols + 1
            col_idx = i % n_cols + 1
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        return fig


@st.cache_data
def create_comprehensive_sample_dataset():
    """Create a comprehensive sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 2500
    
    # Generate realistic financial/business dataset
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int).clip(18, 80),
        'annual_income': np.random.lognormal(10.5, 0.8, n_samples).round(2),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
        'loan_amount': np.random.lognormal(9.5, 0.6, n_samples).round(2),
        'employment_years': np.random.exponential(3, n_samples).round(1).clip(0, 40),
        'debt_to_income_ratio': np.random.beta(2, 5, n_samples).round(3),
        'monthly_expenses': np.random.lognormal(8.5, 0.5, n_samples).round(2),
        'savings_account_balance': np.random.exponential(5000, n_samples).round(2),
        'number_of_dependents': np.random.poisson(1.2, n_samples).clip(0, 6),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                          n_samples, p=[0.3, 0.4, 0.25, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], 
                                         n_samples, p=[0.35, 0.5, 0.12, 0.03]),
        'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], 
                                         n_samples, p=[0.3, 0.2, 0.5]),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                          n_samples, p=[0.7, 0.15, 0.12, 0.03]),
        'loan_purpose': np.random.choice(['home_improvement', 'debt_consolidation', 'business', 'education', 'medical'], 
                                       n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'previous_defaults': np.random.poisson(0.3, n_samples),
        'number_of_bank_accounts': np.random.poisson(2.5, n_samples).clip(1, 8),
        'credit_utilization_ratio': np.random.beta(2, 3, n_samples).round(3),
        'years_at_current_address': np.random.exponential(4, n_samples).round(1).clip(0, 30)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable with complex dependencies
    risk_score = (
        (df['credit_score'] - 650) * 0.015 +
        (df['annual_income'] / 60000 - 1) * 0.4 +
        (df['debt_to_income_ratio'] - 0.3) * -3 +
        df['previous_defaults'] * -0.8 +
        (df['employment_years'] / 10) * 0.3 +
        (df['savings_account_balance'] / 10000) * 0.2 +
        np.random.normal(0, 0.25, n_samples)
    )
    
    # Convert to binary classification
    df['loan_approved'] = (risk_score > np.percentile(risk_score, 30)).astype(int)
    
    # Add realistic missing values
    missing_patterns = {
        'employment_years': 0.05,
        'annual_income': 0.02,
        'savings_account_balance': 0.08,
        'credit_utilization_ratio': 0.03
    }
    
    for col, missing_rate in missing_patterns.items():
        missing_mask = np.random.random(n_samples) < missing_rate
        df.loc[missing_mask, col] = np.nan
    
    # Add some high-cardinality categorical column
    df['customer_segment'] = [f'Segment_{i%100}' for i in range(n_samples)]
    
    return df


def main():
    st.title("🤖 AI Data Scientist Assistant")
    st.markdown("**Comprehensive automated machine learning platform with intelligent insights**")
    
    # Initialize session state
    if 'ai_loaded' not in st.session_state:
        st.session_state.ai_loaded = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🚀 AI Workflow")
        
        # Step 1: Data Loading
        st.subheader("1. Load Dataset")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Use Comprehensive Sample Dataset"]
        )
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel files (max 200MB)"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Loading and analyzing your dataset..."):
                        # Load data
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file)
                        else:
                            data = pd.read_excel(uploaded_file)
                        
                        # Store in session state
                        st.session_state.data = data
                        st.session_state.ai_loaded = True
                        
                        st.success(f"✅ Dataset loaded: {len(data):,} rows, {len(data.columns)} columns")
                
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        else:  # Sample dataset
            if st.button("🎯 Load Comprehensive Sample Dataset", type="primary"):
                with st.spinner("Generating comprehensive sample dataset..."):
                    data = create_comprehensive_sample_dataset()
                    st.session_state.data = data
                    st.session_state.ai_loaded = True
                    st.success(f"✅ Sample dataset loaded: {len(data):,} rows, {len(data.columns)} columns")
        
        # Step 2: Analysis Configuration
        if st.session_state.ai_loaded:
            st.subheader("2. Configure Analysis")
            
            available_columns = st.session_state.data.columns.tolist()
            target_column = st.selectbox(
                "🎯 Select target variable:",
                available_columns,
                help="Choose the variable you want to predict"
            )
            
            if st.button("🔍 Start AI Analysis", type="primary"):
                with st.spinner("AI is performing comprehensive analysis..."):
                    # Initialize analyzers
                    analyzer = AdvancedDataAnalyzer(st.session_state.data)
                    
                    # Create profile
                    profile = analyzer.create_comprehensive_profile(target_column)
                    st.session_state.profile = profile
                    st.session_state.target_column = target_column
                    
                    # Data quality analysis
                    data_quality = analyzer.detect_data_quality_issues()
                    st.session_state.data_quality = data_quality
                    
                    # Visualizations
                    viz_engine = AdvancedVisualizationEngine(st.session_state.data)
                    visualizations = viz_engine.create_eda_suite(profile)
                    st.session_state.visualizations = visualizations
                    
                    # Recommendations
                    recommender = MLModelRecommendationEngine(profile, data_quality)
                    recommendations = recommender.generate_recommendations()
                    st.session_state.recommendations = recommendations
                    
                    st.session_state.analysis_complete = True
                    st.success("✅ Comprehensive analysis complete!")
        
        # Step 3: Model Training
        if st.session_state.analysis_complete:
            st.subheader("3. Train AI Models")
            
            if st.button("🚀 Train Multiple Models", type="primary"):
                with st.spinner("Training and evaluating multiple ML models..."):
                    # Preprocessing
                    preprocessor = SmartPreprocessor()
                    
                    # Prepare data
                    feature_cols = [col for col in st.session_state.data.columns 
                                  if col != st.session_state.target_column]
                    X = st.session_state.data[feature_cols]
                    y = st.session_state.data[st.session_state.target_column]
                    
                    # Preprocess
                    X_processed = preprocessor.create_preprocessing_pipeline(
                        X, st.session_state.target_column, st.session_state.profile
                    )
                    
                    # Train models
                    trainer = ModelTrainingEngine()
                    model_results = trainer.train_and_evaluate_models(
                        X_processed, y, st.session_state.recommendations
                    )
                    
                    # Store results
                    st.session_state.model_results = model_results
                    st.session_state.preprocessing_steps = preprocessor.preprocessing_steps
                    
                    # Create performance visualization
                    viz_engine = AdvancedVisualizationEngine(st.session_state.data)
                    performance_viz = viz_engine.create_model_performance_dashboard(model_results)
                    st.session_state.performance_viz = performance_viz
                    
                    # Select best model
                    best_model = None
                    best_score = -float('inf')
                    primary_metric = st.session_state.recommendations['evaluation_strategy']['primary_metric']
                    
                    for model_name, result in model_results.items():
                        if 'error' not in result and 'metrics' in result:
                            if primary_metric in result['metrics']:
                                score = result['metrics'][primary_metric]
                                if score > best_score:
                                    best_score = score
                                    best_model = model_name
                    
                    st.session_state.best_model = best_model
                    st.session_state.models_trained = True
                    st.success("✅ Model training complete!")
    
    # Main content area
    if not st.session_state.ai_loaded:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### 🎯 Advanced AI Data Science Platform
            
            **Intelligent Automation for Complete ML Workflows**
            
            #### 🔍 **Comprehensive Data Analysis**
            - Automated data profiling with 15+ quality metrics
            - Advanced outlier detection and multicollinearity analysis
            - Interactive EDA with 6+ visualization types
            - Missing value patterns and data quality scoring
            
            #### 🧠 **Smart Preprocessing Engine**
            - Intelligent missing value imputation strategies
            - Adaptive categorical encoding (one-hot, label, frequency)
            - Robust outlier handling with IQR capping
            - Automatic feature engineering and scaling
            
            #### 🤖 **AI-Powered Model Recommendations**
            - Dataset-aware algorithm selection
            - Performance expectation modeling
            - Automated hyperparameter strategy suggestions
            - Class imbalance detection and handling
            
            #### 📊 **Advanced Machine Learning**
            - 6+ optimized algorithms per problem type
            - Cross-validation with adaptive fold selection
            - Feature importance analysis
            - Comprehensive performance metrics
            
            #### 💡 **Intelligent Insights Generation**
            - AI-generated data quality recommendations
            - Performance optimization suggestions
            - Next steps guidance for model improvement
            - Automated reporting capabilities
            
            ---
            
            **Ready to revolutionize your data science workflow?**  
            Start by loading your dataset or exploring with our comprehensive sample data.
            """)
    
    else:
        # Create tabs for analysis results
        if st.session_state.analysis_complete:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Dataset Overview", 
                "🔍 Data Quality Analysis", 
                "📈 Exploratory Analysis",
                "🤖 AI Recommendations", 
                "🚀 Model Results",
                "💡 AI Insights"
            ])
            
            with tab1:
                st.header("📊 Comprehensive Dataset Overview")
                
                # Key metrics
                profile = st.session_state.profile
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Rows", f"{profile['n_rows']:,}")
                with col2:
                    st.metric("Features", profile['n_cols'])
                with col3:
                    st.metric("Missing Data", f"{profile['missing_percentage']:.1f}%")
                with col4:
                    st.metric("Memory Usage", f"{profile['memory_usage_mb']:.1f} MB")
                with col5:
                    st.metric("Duplicates", profile['duplicates_count'])
                
                # Overview visualization
                if 'overview' in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations['overview'], width='stretch')
                
                # Data preview
                st.subheader("🔍 Data Preview")
                display_rows = st.slider("Rows to display:", 5, 100, 20)
                st.dataframe(st.session_state.data.head(display_rows), width='stretch')
                
                # Basic statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Numeric Features Summary")
                    if profile['numeric_cols']:
                        # Use sample for large datasets to avoid memory issues
                        sample_size = min(10000, len(st.session_state.data))
                        st.dataframe(
                            st.session_state.data[profile['numeric_cols']].iloc[:sample_size].describe(),
                            width='stretch'
                        )
                    else:
                        st.info("No numeric features found")
                
                with col2:
                    st.subheader("🏷️ Categorical Features Summary")
                    if profile['categorical_cols']:
                        cat_summary = []
                        for col in profile['categorical_cols'][:10]:
                            unique_count = st.session_state.data[col].nunique()
                            top_value = st.session_state.data[col].mode().iloc[0] if not st.session_state.data[col].mode().empty else "N/A"
                            cat_summary.append({
                                'Feature': col,
                                'Unique Values': unique_count,
                                'Top Value': str(top_value)
                            })
                        st.dataframe(pd.DataFrame(cat_summary), width='stretch')
                    else:
                        st.info("No categorical features found")
            
            with tab2:
                st.header("🔍 Data Quality Analysis")
                
                data_quality = st.session_state.data_quality
                
                # Quality score
                quality_score = (100 - profile['missing_percentage'] * 2 - 
                               min((profile['duplicates_count'] / profile['n_rows']) * 30, 20))
                quality_score = max(quality_score, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Quality Score", f"{quality_score:.1f}/100")
                with col2:
                    st.metric("Issues Detected", len([k for k, v in data_quality.items() if v]))
                with col3:
                    st.metric("High Cardinality Features", len(profile.get('high_cardinality_cols', [])))
                
                # Missing values analysis
                if 'missing_analysis' in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations['missing_analysis'], width='stretch')
                
                # Quality issues breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("❌ Data Quality Issues")
                    
                    if data_quality['missing_values']:
                        st.write("**Missing Values:**")
                        for col, info in data_quality['missing_values'].items():
                            st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                    
                    if data_quality['constant_features']:
                        st.write("**Constant Features:**")
                        for col in data_quality['constant_features']:
                            st.write(f"• {col}")
                    
                    if data_quality['duplicates'] > 0:
                        st.write(f"**Duplicate Rows:** {data_quality['duplicates']}")
                
                with col2:
                    st.subheader("⚠️ Statistical Issues")
                    
                    if data_quality['outliers']:
                        st.write("**Outliers Detected:**")
                        for col, info in data_quality['outliers'].items():
                            st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                    
                    if data_quality['highly_correlated_pairs']:
                        st.write("**Highly Correlated Features:**")
                        for pair in data_quality['highly_correlated_pairs']:
                            st.write(f"• {pair['feature1']} ↔ {pair['feature2']} (r={pair['correlation']:.3f})")
            
            with tab3:
                st.header("📈 Exploratory Data Analysis")
                
                # Feature distributions
                if 'distributions' in st.session_state.visualizations:
                    st.subheader("📊 Feature Distributions")
                    st.plotly_chart(st.session_state.visualizations['distributions'], width='stretch')
                
                # Correlation analysis
                if 'correlations' in st.session_state.visualizations:
                    st.subheader("🔗 Feature Correlations")
                    st.plotly_chart(st.session_state.visualizations['correlations'], width='stretch')
                
                # Categorical analysis
                if 'categorical' in st.session_state.visualizations:
                    st.subheader("🏷️ Categorical Features Analysis")
                    st.plotly_chart(st.session_state.visualizations['categorical'], width='stretch')
                
                # Outlier analysis
                if 'outliers' in st.session_state.visualizations:
                    st.subheader("📈 Outlier Detection")
                    st.plotly_chart(st.session_state.visualizations['outliers'], width='stretch')
            
            with tab4:
                st.header("🤖 AI Model Recommendations")
                
                recommendations = st.session_state.recommendations
                
                # Problem type and strategy
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Problem Analysis")
                    st.info(f"**Problem Type:** {recommendations['problem_type'].title()}")
                    
                    st.write("**Recommended Models:**")
                    for i, model in enumerate(recommendations['recommended_models'], 1):
                        st.write(f"{i}. {model}")
                    
                    st.write("**Preprocessing Strategy:**")
                    for step in recommendations['preprocessing_strategy']:
                        st.write(f"• {step.replace('_', ' ').title()}")
                
                with col2:
                    st.subheader("📊 Evaluation Strategy")
                    eval_strategy = recommendations['evaluation_strategy']
                    st.write(f"**Primary Metric:** {eval_strategy['primary_metric']}")
                    st.write(f"**Cross Validation:** {eval_strategy['cross_validation']}")
                    
                    st.write("**Additional Metrics:**")
                    for metric in eval_strategy['additional_metrics']:
                        st.write(f"• {metric}")
                
                # Performance expectations
                st.subheader("⏱️ Performance Expectations")
                perf_exp = recommendations['performance_expectations']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Time", perf_exp['training_time'])
                with col2:
                    st.metric("Expected Accuracy", perf_exp['expected_accuracy'])
                with col3:
                    st.metric("Overfitting Risk", perf_exp['overfitting_risk'])
                
                # AI explanation
                st.subheader("🧠 AI Explanation")
                st.success(recommendations['model_explanations']['general'])
                
                # Next steps
                if recommendations['next_steps']:
                    st.subheader("📋 Recommended Next Steps")
                    for step in recommendations['next_steps']:
                        st.write(f"• {step}")
            
            with tab5:
                st.header("🚀 Model Training Results")
                
                if st.session_state.models_trained:
                    # Best model highlight
                    if st.session_state.best_model:
                        st.success(f"🏆 **Best Model:** {st.session_state.best_model}")
                    
                    # Performance comparison
                    if 'performance_viz' in st.session_state:
                        st.plotly_chart(st.session_state.performance_viz, width='stretch')
                    
                    # Detailed results table
                    st.subheader("📊 Detailed Performance Metrics")
                    
                    results_data = []
                    for model_name, result in st.session_state.model_results.items():
                        if 'error' not in result:
                            row = {'Model': model_name}
                            row.update(result['metrics'])
                            row['CV Mean'] = f"{result['cv_mean']:.4f}"
                            row['CV Std'] = f"{result['cv_std']:.4f}"
                            row['Training Time'] = f"{result['training_time']:.2f}s"
                            results_data.append(row)
                    
                    if results_data:
                        results_df = pd.DataFrame(results_data)
                        
                        # Highlight best model
                        def highlight_best(row):
                            if row['Model'] == st.session_state.best_model:
                                return ['background-color: #90EE90'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(
                            results_df.style.apply(highlight_best, axis=1),
                            width='stretch'
                        )
                    
                    # Model details
                    st.subheader("🔍 Model Analysis")
                    
                    selected_model = st.selectbox(
                        "Select model for detailed analysis:",
                        [name for name in st.session_state.model_results.keys() 
                         if 'error' not in st.session_state.model_results[name]]
                    )
                    
                    if selected_model:
                        result = st.session_state.model_results[selected_model]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Performance Metrics:**")
                            for metric, value in result['metrics'].items():
                                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                        
                        with col2:
                            st.write("**Training Details:**")
                            st.metric("Cross-Validation Mean", f"{result['cv_mean']:.4f}")
                            st.metric("Cross-Validation Std", f"{result['cv_std']:.4f}")
                            st.metric("Training Time", f"{result['training_time']:.2f}s")
                        
                        # Feature importance
                        if result.get('feature_importance'):
                            st.subheader("📈 Feature Importance")
                            importance_df = pd.DataFrame(
                                list(result['feature_importance'].items()),
                                columns=['Feature', 'Importance']
                            ).sort_values('Importance', ascending=False).head(10)
                            
                            fig = px.bar(
                                importance_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title="Top 10 Most Important Features"
                            )
                            st.plotly_chart(fig, width='stretch')
                    
                    # Preprocessing summary
                    st.subheader("⚙️ Preprocessing Applied")
                    if 'preprocessing_steps' in st.session_state:
                        for step in st.session_state.preprocessing_steps:
                            st.write(f"• {step}")
                
                else:
                    st.info("Complete model training to view results")
            
            with tab6:
                st.header("💡 AI-Generated Insights")
                
                if st.session_state.models_trained:
                    # Generate comprehensive insights
                    insights = {}
                    
                    # Data quality insights
                    missing_pct = st.session_state.profile['missing_percentage']
                    if missing_pct > 20:
                        insights['Data Quality'] = f"High missing data ({missing_pct:.1f}%). Consider improving data collection processes."
                    elif missing_pct > 5:
                        insights['Data Quality'] = f"Moderate missing data ({missing_pct:.1f}%). Current preprocessing handled this well."
                    else:
                        insights['Data Quality'] = f"Excellent data quality with minimal missing values ({missing_pct:.1f}%)."
                    
                    # Model performance insights
                    if st.session_state.best_model:
                        best_result = st.session_state.model_results[st.session_state.best_model]
                        primary_metric = st.session_state.recommendations['evaluation_strategy']['primary_metric']
                        # Use .get() to safely access the metric with fallback
                        score = best_result['metrics'].get(primary_metric)
                        
                        if score is None:
                            # If primary metric not available, try common alternatives
                            for fallback_metric in ['accuracy', 'r2_score', 'f1_score']:
                                if fallback_metric in best_result['metrics']:
                                    score = best_result['metrics'][fallback_metric]
                                    break
                        
                        if score is not None:
                            if score > 0.8:
                                insights['Model Performance'] = f"Excellent performance achieved with {st.session_state.best_model} (score: {score:.3f}). Ready for deployment."
                            elif score > 0.6:
                                insights['Model Performance'] = f"Good performance with {st.session_state.best_model} (score: {score:.3f}). Consider hyperparameter tuning."
                            else:
                                insights['Model Performance'] = f"Moderate performance with {st.session_state.best_model} (score: {score:.3f}). Investigate feature engineering."
                        else:
                            insights['Model Performance'] = f"Model trained successfully with {st.session_state.best_model}. Check detailed metrics below."
                    
                    # Feature insights
                    n_features = len(st.session_state.profile['numeric_cols']) + len(st.session_state.profile['categorical_cols'])
                    n_rows = st.session_state.profile['n_rows']
                    
                    if n_features > n_rows * 0.1:
                        insights['Feature Engineering'] = "High feature-to-sample ratio detected. Feature selection could improve performance."
                    
                    # Class imbalance insights
                    class_imbalance_ratio = st.session_state.profile.get('class_imbalance_ratio') or 1
                    if class_imbalance_ratio > 3:
                        insights['Class Balance'] = f"Class imbalance detected (ratio: {class_imbalance_ratio:.1f}). Consider collecting more balanced data."
                    
                    # Display insights
                    for insight_type, insight_text in insights.items():
                        st.subheader(f"🎯 {insight_type}")
                        st.write(insight_text)
                    
                    # Actionable recommendations
                    st.subheader("📋 Next Steps Recommendations")
                    
                    recommendations_list = []
                    
                    if st.session_state.best_model:
                        recommendations_list.append(f"✅ Deploy {st.session_state.best_model} as your production model")
                    
                    if missing_pct > 10:
                        recommendations_list.append("⚠️ Improve data collection to reduce missing values")
                    
                    class_imbalance_ratio = st.session_state.profile.get('class_imbalance_ratio') or 1
                    if class_imbalance_ratio > 3:
                        recommendations_list.append("⚠️ Collect more balanced training data")
                    
                    if len(st.session_state.data_quality.get('highly_correlated_pairs', [])) > 0:
                        recommendations_list.append("💡 Remove highly correlated features to reduce multicollinearity")
                    
                    recommendations_list.append("🔧 Implement hyperparameter tuning for optimal performance")
                    recommendations_list.append("📊 Set up model monitoring and retraining pipeline")
                    
                    for rec in recommendations_list:
                        st.write(rec)
                    
                    # Model export section
                    st.subheader("💾 Model Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 Download Best Model"):
                            if st.session_state.best_model:
                                model_data = {
                                    'model': st.session_state.model_results[st.session_state.best_model]['model'],
                                    'preprocessing_steps': st.session_state.preprocessing_steps,
                                    'feature_importance': st.session_state.model_results[st.session_state.best_model].get('feature_importance'),
                                    'performance_metrics': st.session_state.model_results[st.session_state.best_model]['metrics']
                                }
                                
                                # Create download
                                model_filename = f"best_model_{st.session_state.best_model.lower()}.pkl"
                                joblib.dump(model_data, model_filename)
                                
                                st.success(f"Model saved as {model_filename}")
                    
                    with col2:
                        if st.button("📄 Generate Analysis Report"):
                            # Generate comprehensive report
                            report_lines = [
                                "AI DATA SCIENTIST - COMPREHENSIVE ANALYSIS REPORT",
                                "=" * 60,
                                f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                                "",
                                "DATASET OVERVIEW:",
                                f"- Rows: {st.session_state.profile['n_rows']:,}",
                                f"- Features: {st.session_state.profile['n_cols']}",
                                f"- Missing Data: {st.session_state.profile['missing_percentage']:.2f}%",
                                f"- Target Variable: {st.session_state.target_column}",
                                f"- Problem Type: {st.session_state.recommendations['problem_type']}",
                                "",
                                "BEST MODEL PERFORMANCE:",
                                f"- Algorithm: {st.session_state.best_model}",
                            ]
                            
                            if st.session_state.best_model:
                                best_result = st.session_state.model_results[st.session_state.best_model]
                                for metric, value in best_result['metrics'].items():
                                    report_lines.append(f"- {metric}: {value:.4f}")
                            
                            report_lines.extend([
                                "",
                                "PREPROCESSING APPLIED:",
                            ])
                            
                            for step in st.session_state.preprocessing_steps:
                                report_lines.append(f"- {step}")
                            
                            report_lines.extend([
                                "",
                                "KEY INSIGHTS:",
                            ])
                            
                            for insight_type, insight_text in insights.items():
                                report_lines.append(f"- {insight_type}: {insight_text}")
                            
                            report_content = "\n".join(report_lines)
                            
                            st.download_button(
                                label="📥 Download Report",
                                data=report_content,
                                file_name=f"ai_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                
                else:
                    st.info("Complete model training to receive AI-generated insights and recommendations")
        
        else:
            # Show basic data overview
            st.header("📊 Dataset Loaded")
            
            data = st.session_state.data
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(data):,}")
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", data.duplicated().sum())
            
            st.subheader("Data Preview")
            st.dataframe(data.head(10), width='stretch')
            
            st.info("👈 Complete the analysis configuration in the sidebar to proceed")


if __name__ == "__main__":
    main()
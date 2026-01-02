"""
Visualization Module for Breast Cancer Classification

Provides comprehensive visualizations for data exploration,
model evaluation, and result presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


class Visualizer:
    """
    Visualization toolkit for breast cancer classification.
    
    Provides publication-ready visualizations for:
    - Data exploration (distributions, correlations)
    - Outlier analysis
    - Model comparison
    - Confusion matrices
    - ROC curves
    - Feature importance
    
    Example:
        >>> viz = Visualizer()
        >>> viz.plot_correlation_heatmap(df)
        >>> viz.plot_confusion_matrix(cm, save_path='output/cm.png')
    """
    
    COLORS = {
        'primary': '#3498db',
        'secondary': '#2ecc71',
        'danger': '#e74c3c',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'benign': '#2ecc71',
        'malignant': '#e74c3c'
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize the Visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        self._setup_style(style)
    
    def _setup_style(self, style: str) -> None:
        """Configure matplotlib style."""
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8')
        
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
    
    # ==================== DATA EXPLORATION ====================
    
    def plot_target_distribution(self, y: np.ndarray, 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of target variable.
        
        Args:
            y: Target array
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        unique, counts = np.unique(y, return_counts=True)
        colors = [self.COLORS['benign'], self.COLORS['malignant']]
        
        axes[0].bar(['Benign (0)', 'Malignant (1)'], counts, color=colors)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution')
        
        for i, (u, c) in enumerate(zip(unique, counts)):
            axes[0].text(i, c + 5, str(c), ha='center', fontsize=12, fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts, labels=['Benign', 'Malignant'], autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=(0, 0.05))
        axes[1].set_title('Class Proportion')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                    target_col: str = 'diagnosis',
                                    n_cols: int = 4,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of all features.
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            n_cols: Number of columns in grid
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        feature_cols = [c for c in df.columns if c != target_col]
        n_features = len(feature_cols)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_cols):
            axes[i].hist(df[col], bins=20, color=self.COLORS['primary'], 
                        alpha=0.7, edgecolor='white')
            axes[i].set_title(col[:20], fontsize=10)
            axes[i].tick_params(labelsize=8)
        
        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (14, 12),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            df: DataFrame
            figsize: Figure size
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, annot_kws={'size': 8}, linewidths=0.5)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_boxplots_by_target(self, df: pd.DataFrame, 
                                 features: List[str],
                                 target_col: str = 'diagnosis',
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot boxplots of features grouped by target.
        
        Args:
            df: DataFrame
            features: List of feature columns
            target_col: Target column name
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = [self.COLORS['benign'], self.COLORS['malignant']]
        
        for i, col in enumerate(features):
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette=colors)
            axes[i].set_xticklabels(['Benign', 'Malignant'])
            axes[i].set_title(col)
        
        # Hide unused
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_outliers(self, df: pd.DataFrame, column: str,
                      outlier_indices: List[int],
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data with outliers highlighted.
        
        Args:
            df: DataFrame
            column: Column to plot
            outlier_indices: Indices of outliers
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(df[column].values, color=self.COLORS['primary'], 
               linewidth=0.8, alpha=0.7)
        
        if outlier_indices:
            ax.scatter(outlier_indices, df[column].iloc[outlier_indices],
                      color=self.COLORS['danger'], s=50, zorder=5, 
                      label=f'Outliers ({len(outlier_indices)})')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(column)
        ax.set_title(f'Outlier Detection: {column}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== MODEL EVALUATION ====================
    
    def plot_confusion_matrix(self, cm: np.ndarray,
                               class_names: List[str] = ['Benign', 'Malignant'],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   annot_kws={'size': 16})
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                               metric: str = 'CV Mean',
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of model performances.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric column to plot
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by metric
        df_sorted = results_df.sort_values(metric, ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Model'])
        ax.set_xlabel(metric)
        ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, df_sorted[metric]):
            ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontsize=10)
        
        ax.set_xlim(0, max(df_sorted[metric]) * 1.15)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                       auc_score: float, model_name: str = 'Model',
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score
            model_name: Name of the model
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color=self.COLORS['primary'], linewidth=2,
               label=f'{model_name} (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        
        ax.fill_between(fpr, tpr, alpha=0.2, color=self.COLORS['primary'])
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves_multiple(self, roc_data: Dict[str, Tuple],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple ROC curves for comparison.
        
        Args:
            roc_data: Dict of {model_name: (fpr, tpr, auc)}
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
        
        for (name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{name} (AUC = {auc_score:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                 top_n: int = 15,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with Feature and Importance columns
            top_n: Number of top features to show
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        df_top = importance_df.head(top_n).sort_values('Importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))
        
        ax.barh(range(len(df_top)), df_top['Importance'], color=colors)
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # ==================== DASHBOARD ====================
    
    def create_dashboard(self, 
                         cm: np.ndarray,
                         results_df: pd.DataFrame,
                         importance_df: Optional[pd.DataFrame] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive results dashboard.
        
        Args:
            cm: Confusion matrix
            results_df: Model comparison results
            importance_df: Optional feature importance
            save_path: Optional path to save
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Grid spec
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   annot_kws={'size': 14})
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('Confusion Matrix')
        
        # 2. Model Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        df_sorted = results_df.sort_values('CV Mean', ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
        ax2.barh(range(len(df_sorted)), df_sorted['CV Mean'], color=colors)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels(df_sorted['Model'])
        ax2.set_xlabel('CV Mean Score')
        ax2.set_title('Model Comparison')
        
        # 3. Feature Importance (if provided)
        ax3 = fig.add_subplot(gs[1, 0])
        if importance_df is not None:
            df_top = importance_df.head(10).sort_values('Importance', ascending=True)
            ax3.barh(range(len(df_top)), df_top['Importance'], 
                    color=self.COLORS['primary'])
            ax3.set_yticks(range(len(df_top)))
            ax3.set_yticklabels(df_top['Feature'], fontsize=9)
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Features')
        else:
            ax3.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4. Metrics Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Get best model metrics
        best_model = results_df.iloc[0]
        
        summary_text = f"""
        CLASSIFICATION RESULTS SUMMARY
        {'='*40}
        
        Best Model: {best_model['Model']}
        
        Cross-Validation Score: {best_model['CV Mean']:.4f}
        Test Score: {best_model.get('Test Score', 'N/A')}
        
        Confusion Matrix Analysis:
        - True Negatives:  {cm[0,0]}
        - False Positives: {cm[0,1]}
        - False Negatives: {cm[1,0]}
        - True Positives:  {cm[1,1]}
        
        Total Models Compared: {len(results_df)}
        {'='*40}
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Breast Cancer Classification - Results Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[OK] Dashboard saved to {save_path}")
        
        return fig

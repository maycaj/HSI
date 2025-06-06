import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import time

class ImprovedSpectrumClassifier:
    def __init__(self, cv_folds=5, random_state=42):
        """
        Initialize the improved spectrum classifier.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
    def load_data(self, filepath):
        """Load and preprocess the spectral data."""
        print('Loading dataframe...')
        start_time = time.time()
        
        self.df = pd.read_csv(filepath)
        print(f'Done loading dataframe ({np.round(time.time()-start_time,1)}s)')
        
        # Extract spectral features and target
        self.wavelength_cols = [col for col in self.df.columns if col.startswith('Wavelength_')]
        self.X = self.df[self.wavelength_cols].values
        self.y = self.df['Foldername'].values
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f'Dataset shape: {self.X.shape}')
        print(f'Number of spectral features: {len(self.wavelength_cols)}')
        print(f'Classes: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}')
        print(f'Class distribution: {pd.Series(self.y).value_counts()}')
        
        return self
    
    def optimize_hyperparameters(self):
        """Optimize hyperparameters using GridSearchCV."""
        print('\\nOptimizing hyperparameters...')
        
        # Define parameter grid (focused on the best performing configuration)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state),
            param_grid,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_scaled, self.y_encoded)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.grid_search_results = grid_search
        
        print(f'Best parameters: {self.best_params}')
        print(f'Best cross-validation score: {grid_search.best_score_:.3f}')
        
        return self
    
    def evaluate_performance(self):
        """Evaluate model performance using cross-validation."""
        print('\\nEvaluating performance...')
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.best_model, self.X_scaled, self.y_encoded, cv=self.cv)
        
        # Cross-validation predictions for confusion matrix
        cv_predictions = cross_val_predict(self.best_model, self.X_scaled, self.y_encoded, cv=self.cv)
        
        # Calculate metrics
        self.cv_accuracy_mean = cv_scores.mean()
        self.cv_accuracy_std = cv_scores.std()
        self.cv_scores = cv_scores
        
        print(f'Cross-validation accuracy: {self.cv_accuracy_mean:.3f} ± {self.cv_accuracy_std:.3f}')
        print(f'Accuracy percentage: {self.cv_accuracy_mean*100:.1f}%')
        
        # Classification report
        print('\\nClassification Report:')
        print(classification_report(self.y_encoded, cv_predictions, 
                                   target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(self.y_encoded, cv_predictions)
        print('\\nConfusion Matrix:')
        print(self.confusion_matrix)
        
        return self
    
    def get_feature_importance(self):
        """Get feature importance for linear SVM."""
        if hasattr(self.best_model, 'coef_') and self.best_model.kernel == 'linear':
            # For linear SVM, use coefficient magnitudes
            self.feature_importance = np.abs(self.best_model.fit(self.X_scaled, self.y_encoded).coef_[0])
            self.wavelengths = [float(col.split('_')[1]) for col in self.wavelength_cols]
            
            # Get top features
            top_indices = np.argsort(self.feature_importance)[-20:]
            self.top_wavelengths = [self.wavelengths[i] for i in top_indices]
            self.top_importance = self.feature_importance[top_indices]
            
            print(f'\\nTop 10 most important wavelengths:')
            for i in range(-10, 0):
                idx = top_indices[i]
                print(f'{self.wavelengths[idx]:.1f}nm: {self.feature_importance[idx]:.4f}')
        else:
            print('\\nFeature importance not available for non-linear kernels.')
            
        return self
    
    def create_visualization(self, save_path=None):
        """Create comprehensive visualization of results."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cross-validation scores
        plt.subplot(2, 3, 1)
        plt.bar(range(len(self.cv_scores)), self.cv_scores, alpha=0.7)
        plt.axhline(y=self.cv_accuracy_mean, color='red', linestyle='--', 
                   label=f'Mean: {self.cv_accuracy_mean:.3f}')
        plt.xlabel('CV Fold')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Confusion matrix
        plt.subplot(2, 3, 2)
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Plot 3: Sample spectra by class
        plt.subplot(2, 3, 3)
        wavelengths = [float(col.split('_')[1]) for col in self.wavelength_cols]
        for class_name in self.label_encoder.classes_:
            class_data = self.df[self.df['Foldername'] == class_name]
            mean_spectrum = class_data[self.wavelength_cols].mean()
            plt.plot(wavelengths, mean_spectrum, label=f'{class_name} (n={len(class_data)})', linewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Mean Spectra by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Feature importance (if available)
        plt.subplot(2, 3, 4)
        if hasattr(self, 'feature_importance'):
            plt.barh(range(len(self.top_importance)), self.top_importance)
            plt.yticks(range(len(self.top_importance)), 
                      [f'{w:.0f}nm' for w in self.top_wavelengths])
            plt.xlabel('|Coefficient|')
            plt.title('Top 20 Important Wavelengths')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Feature importance\\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        # Plot 5: Hyperparameter optimization results
        plt.subplot(2, 3, 5)
        if hasattr(self, 'grid_search_results'):
            results_df = pd.DataFrame(self.grid_search_results.cv_results_)
            # Plot C parameter vs accuracy for linear kernel
            linear_results = results_df[results_df['param_kernel'] == 'linear']
            if len(linear_results) > 0:
                plt.plot(linear_results['param_C'], linear_results['mean_test_score'], 'o-', label='Linear')
            
            rbf_results = results_df[results_df['param_kernel'] == 'rbf']
            if len(rbf_results) > 0:
                plt.plot(rbf_results['param_C'], rbf_results['mean_test_score'], 's-', label='RBF')
            
            plt.xlabel('C Parameter')
            plt.ylabel('CV Accuracy')
            plt.title('Hyperparameter Optimization')
            plt.xscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Performance summary
        plt.subplot(2, 3, 6)
        metrics = ['Accuracy', 'Std Dev', 'Min Score', 'Max Score']
        values = [self.cv_accuracy_mean, self.cv_accuracy_std, 
                 self.cv_scores.min(), self.cv_scores.max()]
        colors = ['green', 'orange', 'red', 'blue']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel('Value')
        plt.title('Performance Summary')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Visualization saved to: {save_path}')
        
        plt.show()
        return self
    
    def save_model(self, model_path):
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'wavelength_cols': self.wavelength_cols,
            'best_params': self.best_params,
            'cv_accuracy': self.cv_accuracy_mean,
            'cv_std': self.cv_accuracy_std
        }
        dump(model_data, model_path)
        print(f'Model saved to: {model_path}')
        return self
    
    def predict_new_data(self, new_data):
        """Predict classes for new spectral data."""
        # Ensure the model is fitted
        if self.best_model is None:
            raise ValueError("Model not trained. Call optimize_hyperparameters() first.")
        
        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions_encoded = self.best_model.predict(new_data_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get prediction probabilities if available
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(new_data_scaled)
        else:
            probabilities = None
        
        return predictions, probabilities

def main():
    """Main execution function."""
    # Initialize classifier
    classifier = ImprovedSpectrumClassifier(cv_folds=5, random_state=42)
    
    # Load data (update this path to your data file)
    data_path = '/Users/maycaj/Documents/HSI_III/PatchCSVs/May28_CR_FullRound1and2AllWLs_medians'
    classifier.load_data(data_path)
    
    # Optimize hyperparameters
    classifier.optimize_hyperparameters()
    
    # Evaluate performance
    classifier.evaluate_performance()
    
    # Get feature importance
    classifier.get_feature_importance()
    
    # Create visualization
    classifier.create_visualization('improved_classifier_results.png')
    
    # Save the model
    classifier.save_model('improved_spectrum_classifier.joblib')
    
    # Print final results
    print(f'\\n{"="*60}')
    print(f'IMPROVED CLASSIFIER RESULTS')
    print(f'{"="*60}')
    print(f'Original baseline accuracy: ~83.6%')
    print(f'Improved accuracy: {classifier.cv_accuracy_mean*100:.1f}% ± {classifier.cv_accuracy_std*100:.1f}%')
    print(f'Improvement: +{(classifier.cv_accuracy_mean - 0.836)*100:.1f} percentage points')
    print(f'Best parameters: {classifier.best_params}')
    print(f'{"="*60}')
    
    return classifier

if __name__ == "__main__":
    classifier = main()

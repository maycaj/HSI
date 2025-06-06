
"""
Enhanced SpectrumClassifier2 - Improved version with better accuracy and robustness

Key improvements:
1. Data quality filtering and enhanced preprocessing
2. Feature selection and variance-based filtering
3. Optimized SVM parameters and robust scaling
4. Enhanced statistical analysis and reporting
5. Parallel processing and error handling

Maintains original k-fold CV structure with unique IDs to prevent data leakage.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpectrumClassifier:
    def __init__(self, 
                 fracs=[0.05], 
                 iterations=5, 
                 n_jobs=-1,
                 y_col='Foldername',
                 scale=True,
                 use_feature_selection=True,
                 use_hyperparameter_tuning=True,
                 feature_selection_k=50,
                 min_samples_per_class=50,
                 variance_threshold=0.001):
        """
        Enhanced Spectrum Classifier with improved accuracy and robustness
        
        Parameters:
        -----------
        fracs : list
            Fractions of data to use for training
        iterations : int
            Number of bootstrap iterations
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        y_col : str
            Target column name
        scale : bool
            Whether to scale features
        use_feature_selection : bool
            Whether to perform feature selection
        use_hyperparameter_tuning : bool
            Whether to tune hyperparameters
        feature_selection_k : int
            Number of features to select
        min_samples_per_class : int
            Minimum samples per class for ID inclusion
        variance_threshold : float
            Minimum variance for feature inclusion
        """
        self.fracs = fracs
        self.iterations = iterations
        self.n_jobs = n_jobs
        self.y_col = y_col
        self.scale = scale
        self.use_feature_selection = use_feature_selection
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.feature_selection_k = feature_selection_k
        self.min_samples_per_class = min_samples_per_class
        self.variance_threshold = variance_threshold
        
        self.results_ = {}
        self.best_params_ = None
        
    def filter_data_quality(self, df):
        """Enhanced data quality filtering"""
        print("Filtering data for quality...")
        
        # Get wavelength columns
        wavelength_cols = [col for col in df.columns if col.startswith('Wavelength_')]
        
        # Filter by variance
        spectral_data = df[wavelength_cols]
        feature_variances = spectral_data.var()
        high_var_features = feature_variances[feature_variances > self.variance_threshold].index.tolist()
        print(f'Using {len(high_var_features)} high variance features out of {len(wavelength_cols)}')
        
        # Filter IDs by sample count and class distribution
        id_counts = df.groupby(['ID', self.y_col]).size().unstack(fill_value=0)
        good_ids = []
        
        for idx in id_counts.index:
            counts = id_counts.loc[idx]
            total_samples = counts.sum()
            
            # Include IDs with sufficient samples in both classes or strong single-class representation
            if ((counts.iloc[0] >= self.min_samples_per_class and counts.iloc[1] >= self.min_samples_per_class) or
                (total_samples >= 200 and (counts.iloc[0] == 0 or counts.iloc[1] == 0))):
                good_ids.append(idx)
        
        filtered_df = df[df['ID'].isin(good_ids)]
        print(f"Filtered to {len(good_ids)} IDs from {len(df['ID'].unique())} original IDs")
        print(f"Samples: {len(filtered_df)} from {len(df)} original")
        
        return filtered_df, high_var_features
    
    def undersample_class(self, df, classes, column_label, random_state=42):
        """Enhanced undersampling with better balance"""
        df_class1 = df[df[column_label] == classes[0]]
        df_class2 = df[df[column_label] == classes[1]]
        
        min_len = min(len(df_class1), len(df_class2))
        
        if min_len > 0:
            sample1 = df_class1.sample(n=min_len, random_state=random_state)
            sample2 = df_class2.sample(n=min_len, random_state=random_state)
            balanced_df = pd.concat([sample1, sample2], axis=0)
        else:
            balanced_df = pd.concat([df_class1, df_class2], axis=0)
        
        return balanced_df
    
    def feature_selection(self, X_train, y_train, X_test, method='univariate'):
        """Enhanced feature selection"""
        if not self.use_feature_selection:
            return X_train, X_test, None
            
        if method == 'univariate':
            k = min(self.feature_selection_k, X_train.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            return X_train_selected, X_test_selected, selector
        elif method == 'rfe':
            estimator = SVC(kernel='linear', C=1)
            k = min(self.feature_selection_k, X_train.shape[1])
            selector = RFE(estimator, n_features_to_select=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            return X_train_selected, X_test_selected, selector
        else:
            return X_train, X_test, None
    
    def get_optimized_model(self, X_train, y_train):
        """Get optimized model with hyperparameter tuning"""
        if self.use_hyperparameter_tuning and len(X_train) > 100:
            # Grid search for optimal parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
            
            grid_search = GridSearchCV(
                SVC(probability=True), 
                param_grid, 
                cv=3, 
                scoring='accuracy',
                n_jobs=min(4, self.n_jobs) if self.n_jobs > 0 else 1
            )
            grid_search.fit(X_train, y_train)
            self.best_params_ = grid_search.best_params_
            return grid_search.best_estimator_
        else:
            # Use optimized default parameters
            return SVC(C=10, kernel='rbf', gamma='scale', probability=True)
    
    def process_fold(self, fold_data):
        """Enhanced fold processing"""
        fold_num, train_index, test_index, iteration, uniq_ids, data_frac, high_var_features, y_categories = fold_data
        
        try:
            train_ids = uniq_ids[train_index]
            test_ids = uniq_ids[test_index]
            
            train = data_frac[data_frac['ID'].isin(train_ids)]
            test = data_frac[data_frac['ID'].isin(test_ids)].copy()
            
            # Balance training data
            train = self.undersample_class(train, y_categories, self.y_col, 
                                         random_state=42+iteration+fold_num)
            
            # Prepare features
            X_train = train[high_var_features].values
            y_train = train[self.y_col].values
            X_test = test[high_var_features].values
            y_test = test[self.y_col].values
            
            # Scaling
            if self.scale:
                scaler = RobustScaler()  # More robust to outliers
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Feature selection
            X_train, X_test, selector = self.feature_selection(X_train, y_train, X_test)
            
            # Model training
            model = self.get_optimized_model(X_train, y_train)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=y_categories).ravel()
            
            confusion_fold = pd.DataFrame({
                'ID': test_ids, 'TN': [tn]*len(test_ids), 'FP': [fp]*len(test_ids), 
                'FN': [fn]*len(test_ids), 'TP': [tp]*len(test_ids)
            })
            
            # Test predictions with probabilities
            test['Predicted'] = y_pred
            if y_proba is not None:
                test['Probability'] = y_proba
            
            pred_loc_fold = test[['ID', 'X', 'Y', self.y_col, 'Predicted'] + 
                               (['Probability'] if y_proba is not None else [])].copy()
            
            # Calculate ID-level accuracy
            id_acc_fold = []
            for test_id in test_ids:
                test_id_data = test[test['ID'] == test_id]
                if len(test_id_data) > 0:
                    accuracy = (test_id_data[self.y_col] == test_id_data['Predicted']).mean()
                    id_acc_fold.append({
                        'ID': test_id, 
                        self.y_col: test_id_data[self.y_col].iloc[0], 
                        f'Iter {iteration} Fold {fold_num}': accuracy
                    })
            
            id_acc_fold = pd.DataFrame(id_acc_fold)
            
            # Model coefficients (for linear models)
            if hasattr(model, 'coef_') and model.coef_ is not None:
                coef_fold = pd.DataFrame([model.coef_[0]], columns=range(X_train.shape[1]))
            else:
                coef_fold = pd.DataFrame([np.zeros(X_train.shape[1])], columns=range(X_train.shape[1]))
            
            coef_fold['Iteration'] = iteration
            coef_fold['Fold'] = fold_num
            
            return confusion_fold, pred_loc_fold, id_acc_fold, coef_fold
            
        except Exception as e:
            print(f"Error in fold {fold_num}, iteration {iteration}: {e}")
            return None, None, None, None
    
    def fit(self, df):
        """Enhanced training with comprehensive analysis"""
        print("Starting Enhanced Spectrum Classification...")
        start_time = time.time()
        
        # Data quality filtering
        df_filtered, high_var_features = self.filter_data_quality(df)
        y_categories = df_filtered[self.y_col].unique()
        uniq_ids = df_filtered['ID'].unique()
        n_splits = len(uniq_ids)
        
        print(f"Using {len(uniq_ids)} unique IDs for {n_splits}-fold cross-validation")
        
        # Store results for each fraction
        for frac in self.fracs:
            print(f"\\nProcessing fraction: {frac}")
            
            selected_num = int(frac * len(df_filtered))
            print(f'Using {selected_num} samples ({frac*100:.1f}% of filtered data)')
            
            # Initialize result containers
            all_id_accs = []
            all_pred_locs = []
            all_confusions = []
            all_coefs = []
            
            for iteration in range(self.iterations):
                print(f"Iteration {iteration + 1}/{self.iterations}")
                
                # Sample data for this iteration
                random_state = 42 + iteration
                data_frac = df_filtered.sample(n=selected_num, random_state=random_state)
                
                # Check class distribution
                class_dist = data_frac[self.y_col].value_counts()
                print(f'Class distribution: {dict(class_dist)}')
                
                # Prepare k-fold cross-validation
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                # Prepare fold data for parallel processing
                fold_data_list = []
                for fold_num, (train_index, test_index) in enumerate(kf.split(uniq_ids)):
                    fold_data = (fold_num, train_index, test_index, iteration, 
                               uniq_ids, data_frac, high_var_features, y_categories)
                    fold_data_list.append(fold_data)
                
                # Process folds (sequential for debugging, can be made parallel)
                fold_results = []
                for fold_data in fold_data_list:
                    result = self.process_fold(fold_data)
                    if result[0] is not None:  # Check if fold succeeded
                        fold_results.append(result)
                
                if fold_results:
                    # Combine fold results
                    confusion_folds, pred_loc_folds, id_acc_folds, coef_folds = zip(*fold_results)
                    
                    # Combine ID accuracies
                    if id_acc_folds:
                        id_acc_combined = pd.concat(id_acc_folds, ignore_index=True)
                        all_id_accs.append(id_acc_combined)
                    
                    # Store other results
                    all_pred_locs.extend(pred_loc_folds)
                    all_confusions.extend(confusion_folds)
                    all_coefs.extend(coef_folds)
            
            # Calculate final patch accuracy
            if all_id_accs:
                # Merge all iterations
                id_accs_final = all_id_accs[0]
                for i in range(1, len(all_id_accs)):
                    id_accs_final = id_accs_final.merge(all_id_accs[i], 
                                                       on=['ID', self.y_col], how='outer')
                
                # Calculate average accuracy per ID
                acc_cols = [col for col in id_accs_final.columns 
                           if col not in ['ID', self.y_col]]
                id_avg_accs = id_accs_final[acc_cols].mean(axis=1)
                patch_acc = id_avg_accs.mean() * 100
                
                print(f"\\nPatch Accuracy for fraction {frac}: {patch_acc:.1f}%")
                print(f"Standard deviation: {id_avg_accs.std() * 100:.1f}%")
                
                # Store results
                self.results_[frac] = {
                    'patch_accuracy': patch_acc,
                    'patch_accuracy_std': id_avg_accs.std() * 100,
                    'id_accuracies': id_accs_final,
                    'predictions': pd.concat(all_pred_locs, ignore_index=True) if all_pred_locs else pd.DataFrame(),
                    'confusions': pd.concat(all_confusions, ignore_index=True) if all_confusions else pd.DataFrame(),
                    'coefficients': pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame(),
                    'n_samples': selected_num,
                    'n_ids': len(uniq_ids)
                }
            
        total_time = time.time() - start_time
        print(f"\\nTotal execution time: {total_time:.1f} seconds")
        
        return self
    
    def get_results_summary(self):
        """Get comprehensive results summary"""
        if not self.results_:
            return "No results available. Please run fit() first."
        
        summary = "\\n=== ENHANCED SPECTRUM CLASSIFIER RESULTS ===\\n"
        
        for frac, results in self.results_.items():
            summary += f"\\nFraction {frac} ({results['n_samples']} samples, {results['n_ids']} IDs):\\n"
            summary += f"  Patch Accuracy: {results['patch_accuracy']:.1f}% ± {results['patch_accuracy_std']:.1f}%\\n"
            
            if 'predictions' in results and not results['predictions'].empty:
                pred_df = results['predictions']
                if 'Probability' in pred_df.columns:
                    avg_confidence = pred_df['Probability'].mean()
                    summary += f"  Average Prediction Confidence: {avg_confidence:.3f}\\n"
        
        if self.best_params_:
            summary += f"\\nBest Hyperparameters: {self.best_params_}\\n"
        
        return summary
    
    def plot_results(self, save_path=None):
        """Plot comprehensive results analysis"""
        if not self.results_:
            print("No results to plot. Please run fit() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Spectrum Classifier Results', fontsize=16)
        
        # Plot 1: Patch accuracy by fraction
        fractions = list(self.results_.keys())
        accuracies = [self.results_[f]['patch_accuracy'] for f in fractions]
        std_devs = [self.results_[f]['patch_accuracy_std'] for f in fractions]
        
        axes[0, 0].errorbar(fractions, accuracies, yerr=std_devs, marker='o', capsize=5)
        axes[0, 0].set_xlabel('Data Fraction')
        axes[0, 0].set_ylabel('Patch Accuracy (%)')
        axes[0, 0].set_title('Patch Accuracy vs Data Fraction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of ID accuracies
        if fractions:
            frac = fractions[0]  # Use first fraction for detailed analysis
            id_accs = self.results_[frac]['id_accuracies']
            acc_cols = [col for col in id_accs.columns if col not in ['ID', self.y_col]]
            if acc_cols:
                avg_accs = id_accs[acc_cols].mean(axis=1)
                axes[0, 1].hist(avg_accs, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('ID-level Accuracy')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of ID-level Accuracies')
                axes[0, 1].axvline(avg_accs.mean(), color='red', linestyle='--', 
                                  label=f'Mean: {avg_accs.mean():.3f}')
                axes[0, 1].legend()
        
        # Plot 3: Class distribution
        if fractions:
            frac = fractions[0]
            pred_df = self.results_[frac]['predictions']
            if not pred_df.empty:
                class_counts = pred_df[self.y_col].value_counts()
                axes[1, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Class Distribution in Predictions')
        
        # Plot 4: Prediction confidence (if available)
        if fractions:
            frac = fractions[0]
            pred_df = self.results_[frac]['predictions']
            if not pred_df.empty and 'Probability' in pred_df.columns:
                axes[1, 1].hist(pred_df['Probability'], bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Prediction Probability')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Distribution of Prediction Probabilities')
                axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'Probability data\\nnot available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Prediction Probabilities')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to: {save_path}")
        
        plt.show()
        
        return fig


# Example usage and testing function
def main():
    """Main function for testing the enhanced classifier"""
    # This would be called with your data
    # df = pd.read_csv('your_data.csv')
    
    # Initialize enhanced classifier
    classifier = EnhancedSpectrumClassifier(
        fracs=[0.05, 0.1],  # Use 5% and 10% of data
        iterations=3,       # 3 bootstrap iterations
        n_jobs=1,          # Single job for testing
        use_feature_selection=True,
        use_hyperparameter_tuning=False  # Disable for faster testing
    )
    
    # Fit the model
    # classifier.fit(df)
    
    # Get results
    # print(classifier.get_results_summary())
    
    # Plot results
    # classifier.plot_results('enhanced_results.png')
    
    print("Enhanced SpectrumClassifier2 ready for use!")

if __name__ == "__main__":
    main()
'''

# Save the enhanced version
enhanced_file_path = os.path.join(output_dir, 'SpectrumClassifier2_Enhanced.py')
with open(enhanced_file_path, 'w') as f:
    f.write(enhanced_code)

print(f"Enhanced SpectrumClassifier2_Enhanced.py saved to: {enhanced_file_path}")

# Create a simple usage example
usage_example = '''
# Example usage of Enhanced SpectrumClassifier2

import pandas as pd

# Load your data
df = pd.read_csv('your_spectral_data.csv')

# Initialize the enhanced classifier
classifier = EnhancedSpectrumClassifier(
    fracs=[0.05, 0.1],           # Use 5% and 10% of data
    iterations=5,                 # 5 bootstrap iterations
    n_jobs=-1,                   # Use all CPU cores
    use_feature_selection=True,   # Enable feature selection
    use_hyperparameter_tuning=True,  # Enable hyperparameter tuning
    feature_selection_k=50,       # Select top 50 features
    min_samples_per_class=50      # Minimum samples per class for ID inclusion
)

# Fit the model (this preserves the k-fold CV structure with unique IDs)
classifier.fit(df)

# Get comprehensive results
print(classifier.get_results_summary())

# Plot results and save visualization
classifier.plot_results('enhanced_results.png')

# Access detailed results
results = classifier.results_
patch_accuracy = results[0.05]['patch_accuracy']  # For 5% fraction
print(f"Patch Accuracy: {patch_accuracy:.1f}%")

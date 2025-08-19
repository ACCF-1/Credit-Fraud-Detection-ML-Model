"""ML model class"""

# In[0] Libraries
'''Import necessary libraries'''
# Standard library imports
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Union
import numbers

# Third-party imports
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as imb_pipe
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn import (
    calibration, discriminant_analysis, ensemble, feature_selection, 
    gaussian_process, linear_model, metrics, model_selection, 
    naive_bayes, neighbors, svm, tree
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline as skl_pipe
from xgboost import XGBClassifier

# Local imports
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'ml_scripts'))
import config as cfg
import specific_functions as sf
import utility_functions as uf

# Disable chained assignment warning
pd.options.mode.chained_assignment = None


#In[1] ML model class
'''machine learning model'''

class ML_Model():
    """
    A machine learning model class for fraud detection with comprehensive functionality
    including data handling, model training, tuning, evaluation, and deployment.
    """
    
    # Class-level constants
    VALID_PHASES = ['deploy', 'training']
    DEFAULT_SEED = 42
    DEFAULT_SPLIT_RATIO = 0.2
    MODEL_MAPPING = {
        'bl': DummyClassifier,
        'dt': tree.DecisionTreeClassifier,
        'knn': neighbors.KNeighborsClassifier,
        'rf': ensemble.RandomForestClassifier,
        'lr': LogisticRegression,
        'svm': svm.SVC,
        'gpc': gaussian_process.GaussianProcessClassifier,
        'xgb': XGBClassifier,
        'ada': AdaBoostClassifier,
    }
    
    def __init__(
            self,
            modeling_phase:str='training',
            target_col:str='is_fraud',
            input_file_name:str='credit_fraud_data_transformed',
            seed:int=DEFAULT_SEED
    ):
        """
        Initialize the ML model with configuration parameters.
        
        Args:
            modeling_phase: Either 'deploy' or 'training'
            target_col: Name of the target column
            input_file_name: Base name of the input data file
            seed: Random seed for reproducibility
        """
        self._validate_phase(modeling_phase)
        self._phase = modeling_phase
        self._target = target_col
        self.seed = seed
        
        # Set up paths
        self._parent_dir = os.path.dirname(os.getcwd()) if __name__ == "__main__" else os.getcwd()
        self.input_file_path = os.path.join(self._parent_dir, 'data', 'processed', f'{input_file_name}.csv')
        
        # Initialize model-related attributes
        self._data_for_mdl = self._load_data()
        self._features = self._extract_features()
        self._features_picked = None
        self._trained_model = None
        self._best_model = None
        self._mdl_param_grid = None
        self._mdl_predictions = None

        # Initialize data split attributes
        self._train_X = None
        self._test_X = None
        self._train_tgt_y = None
        self._test_tgt_y = None
        self._class_weight = None
        
        print('Model initialized successfully')

    # Properties and setters ===================================================
    
    @property
    def phase(self) -> str:
        return self._phase
    
    @phase.setter
    def phase(self, val: str):
        self._validate_phase(val)
        self._phase = val
        print("Refreshing model data...")
        self._data_for_mdl = self._load_data()
        self._features = self._extract_features()

    @property
    def data_for_mdl(self) -> pd.DataFrame:
        return self._data_for_mdl
    
    @data_for_mdl.setter
    def data_for_mdl(self, val: pd.DataFrame):
        if not isinstance(val, pd.DataFrame):
            raise ValueError("data_for_mdl must be a pandas DataFrame.")
        self._data_for_mdl = val
        self._features = self._extract_features()

    @property
    def trained_model(self):
        return self._trained_model
    
    @property
    def best_model(self):
        return self.best_model
    
    @best_model.setter
    def best_model(self, value):
        self._best_model = value

    @property
    def target(self) -> str:
        return self._target
    
    @property
    def features_picked(self) -> Optional[List[str]]:
        return self.features_picked
    
    @features_picked.setter
    def features_picked(self, val: Optional[List[str]]):
        self._features_picked = val

    @property
    def train_tgt_y(self):
        return self._train_tgt_y

    @train_tgt_y.setter
    def train_tgt_y(self, value):
        self._train_tgt_y = value
        # Update class_weight whenever train_tgt_y is set
        self._update_class_weight()

    @property
    def mdl_param_grid(self) -> Optional[Dict]:
        return self._mdl_param_grid

    @mdl_param_grid.setter
    def mdl_param_grid(self, val: Dict):
        self._mdl_param_grid = val

    # Private helper methods ===================================================

    def _update_class_weight(self):
        """
        Update the class weight attribute based on the training target values.
        """
        if self._train_tgt_y is not None:
            value_counts = self._train_tgt_y.value_counts()
            if len(value_counts) > 0:
                self._class_weight = len(self._train_tgt_y) / min(value_counts)
            else:
                self._class_weight = None  # or some default value
        else:
            self._class_weight = None

    def _validate_phase(self, phase: str):
        """
        Validate that the modeling phase is either 'deploy' or 'training'.

        Args:
            phase (str): The phase to validate.

        Raises:
            ValueError: If phase is not valid.
        """
        if phase not in self.VALID_PHASES:
            raise ValueError(f'Phase must be one of {self.VALID_PHASES}')
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load and return the model data based on the current phase.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"Data file not found at {self.input_file_path}")
        return pd.read_csv(self.input_file_path)
    
    def _extract_features(self) -> np.ndarray:
        """
        Extract feature names from the data, excluding the target column.

        Returns:
            np.ndarray: Array of feature names.
        """
        return self._data_for_mdl.columns[~(self._data_for_mdl.columns == self._target)].values
    
    def _func_disabler(self, warn_msg: str):
        """
        Raise an exception if called during deployment phase.

        Args:
            warn_msg (str): Warning message to display.

        Raises:
            Exception: If called in deployment phase.
        """
        if self.phase == 'deploy':
            raise Exception(warn_msg)

    def _initialize_model(self, model_name: str):
        """
        Get an instance of the specified model with random seed.

        Args:
            model_name (str): Short name of the model.

        Returns:
            Model instance.
        """
        if model_name not in self.MODEL_MAPPING:
            raise ValueError(f"Invalid model name. Choose from: {list(self.MODEL_MAPPING.keys())}")
        return self.MODEL_MAPPING[model_name](random_state=self.seed)
    
    def _get_param_grid(self, model_name:str, param_file_name:str='model_param_grids') -> Dict:
        """
        Return the parameter grid for the specified model.

        Args:
            model_name (str): Short name of the model.
            param_file_name (str): Name of the parameter grid file.

        Returns:
            Dict: Parameter grid.
        """
        df = pd.read_csv(os.path.join(self._parent_dir, 'configurations', f'{param_file_name}.csv'))
        param_grids = {}
        
        for model_name_in_df in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name_in_df]
            model_grid = {}
            
            for _, row in model_df.iterrows():
                param = row['parameter']
                values = row['values']
                value_type = row['value_type']
                special_type = row['special_type'] if pd.notna(row['special_type']) else None
                
                # Process based on value type
                if value_type == 'string':
                    model_grid[param] = [values]
                elif value_type == 'list':
                    items = [x.strip() for x in values.split('|')]
                    if special_type == 'int':
                        model_grid[param] = [int(x) for x in items]
                    elif special_type == 'float':
                        model_grid[param] = [float(x) for x in items]
                    elif special_type == 'str':
                        model_grid[param] = [str(x) for x in items]
                    elif special_type == 'bool':
                        model_grid[param] = [bool(x) for x in items]
                elif value_type == 'range':
                    start, stop, step = map(int, values.split('|'))
                    model_grid[param] = list(range(start, stop, step))
                elif value_type == 'logspace':
                    start, stop, num = map(float, values.split('|'))
                    model_grid[param] = list(np.logspace(start, stop, int(num)))
                elif value_type == 'bool':
                    model_grid[param] = [x.strip().lower() == 'true' for x in values.split('|')]
                    
            param_grids[model_name_in_df] = model_grid
        
        grid = param_grids.get(model_name, {})
        # Add random state to all grids
        if grid:
            grid['random_state'] = [self.seed]
        return grid
    
    def _get_cv_strategy(self, model_name: str):
        """
        Return appropriate cross-validation strategy.

        Args:
            model_name (str): Short name of the model.

        Returns:
            Cross-validation strategy.
        """
        if model_name == 'bl':
            return model_selection.KFold(n_splits=2, random_state=self.seed, shuffle=True)
        return model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.seed)
    
    def _save_model(self, model, version: int = 1):
        """
        Save the model to disk with versioning.

        Args:
            model: Model to save.
            version (int): Version number.
        """
        saves_dir = os.path.join(self._parent_dir, 'saves')
        os.makedirs(saves_dir, exist_ok=True)
        
        filename = f"{self._target}_v{version}.pkl" if version > 1 else f"{self._target}.pkl"
        with open(os.path.join(saves_dir, filename), 'wb') as f:
            dill.dump(model, f)
    
    def _save_params_info(self, params: Dict, version: int = 1):
        """
        Save model parameters and performance info to a text file.

        Args:
            params (Dict): Model parameters.
            version (int): Version number.
        """
        results_dir = os.path.join(self._parent_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        filename = f"{self._target}_v{version} - model params.txt" if version > 1 else f"{self._target} - model params.txt"
        with open(os.path.join(results_dir, filename), 'w') as f:
            for key, value in params.items():
                f.write(f'{key}: {value}\n')
            #f.write(f'Train/test split ratio used: {split_ratio}\n')
            f.write(f'Best params: {self._trained_model.best_params_}\n')
            f.write(f'Best score: {self._trained_model.best_score_}\n')
    
    def _get_model_version_count(self) -> int:
        """
        Count the number of saved model versions.

        Returns:
            int: Number of model versions.
        """
        saves_dir = os.path.join(self._parent_dir, 'saves')
        mdl_version_cnt = 1
        if os.path.exists(saves_dir):
            mdl_version_cnt += sum(
                1 for f in os.listdir(saves_dir) 
                if f.endswith('.pkl') and f.startswith(self._target)
            )
        return mdl_version_cnt

    def _build_feature_selector_by_stats(
            self,
            numerical_pipe_components: list = None,
            categorical_pipe_components: list = None,
            numerical_columns: list = ['auto_detect'],
            categorical_columns: list = ['auto_detect'],
    ):
        """
        Build a feature selector pipeline based on statistical methods.

        Args:
            numerical_pipe_components (list): Components for numerical pipeline.
            categorical_pipe_components (list): Components for categorical pipeline.
            numerical_columns (list): Numerical columns to use.
            categorical_columns (list): Categorical columns to use.

        Returns:
            ColumnTransformer: Feature selector pipeline.
        """
        numerical_columns = lambda X: sf.get_cols_for_selector(X.columns, 'num') if numerical_columns == ['auto_detect'] else numerical_columns
        categorical_columns = lambda X: sf.get_cols_for_selector(X.columns, 'cat') if categorical_columns == ['auto_detect'] else categorical_columns

        num_select_pipe = skl_pipe(numerical_pipe_components)
        cat_select_pipe = skl_pipe(categorical_pipe_components)
        col_trans_pipe = uf.ColumnTransformer([
            ('num fil', num_select_pipe, numerical_columns),
            ('cat fil', cat_select_pipe, categorical_columns)
        ], remainder='passthrough')
        return col_trans_pipe

    def _build_feature_selector_by_model(self, model_name:str='rf'):
        """
        Build a feature selector using a model.

        Args:
            model_name (str): Short name of the model.

        Returns:
            SelectFromModel: Feature selector.
        """
        model = self.MODEL_MAPPING[model_name]()
        if model is None:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {list(self.MODEL_MAPPING.keys())}")
        model.set_params(n_estimators=4000, max_depth=2, max_features=3, random_state=self.seed)

        return SelectFromModel(model)

    def _build_preprocessor(
            self, 
            components:list,
            params:list=None,
    ):
        """
        Build the data preprocessing pipeline. Return a pipeline with components.

        Args:
            components: List of tuples specifying the pipeline steps
            params: List of parameter tuples for pipeline components

        Returns:
            skl_pipe: A scikit-learn pipeline object
        """
        preprocessor = skl_pipe(components)

        # Dynamically construct the parameter name (e.g., 'DomainDrop__kw_args')
        if params:
            for idx, param in enumerate(params):
                component_name = param[0]
                param_name = f"{component_name}__kw_args"

                # Set the kwargs based on the component name
                preprocessor.set_params(
                    **{
                        param_name: param[1]
                    }
                )
        return preprocessor

    def _build_model_pipeline(self, model_name, model, resampling:tuple=None, *feature_selectors):
        """
        Build the full model pipeline with feature selection and rebalancing.

        Args:
            model_name: Name of the model
            model: The model instance
            resampling: Resampling method (optional)
            *feature_selectors: Variable number of feature selector objects

        Returns:
            Pipeline: Model pipeline.
        """
        components = []
        
        # Add all feature selectors with sequential numbering
        for i, selector in enumerate(feature_selectors, start=1):
            components.append((f'selector{i}', selector))
        
        # Add the remaining fixed components
        if resampling:
            components.append(('rebal', resampling)) # ('rebal', ADASYN(random_state=self.seed))

        components.append((model_name, model)) #CalibratedClassifierCV(model, method='isotonic', cv=3)
        
        return imb_pipe(components)

    def _convert_param_grid_for_pipeline(self, model_name, mdl_param_grid):
        """
        Convert parameter grid for use in a pipeline.

        Args:
            model_name (str): Short name of the model.
            mdl_param_grid (dict): Model parameter grid.

        Returns:
            dict: Converted parameter grid.
        """
        return {
            f'imba__{model_name}__{key}': value 
            for key, value in mdl_param_grid.items()
        }
    
    def _get_model_params(self):
        """
        Display and store the best parameters and performance metrics.
        """
        self._func_disabler("No model params export under deployment phase")
        
        if self._trained_model is None:
            raise Exception("Please either fit a new model or import an existing model first")
        
        # Print performance metrics
        print('Best: %f using params %s' % (self._trained_model.best_score_, self._trained_model.best_params_))
        print('Train score mean: {:.2f}'.format(self._trained_model.cv_results_['mean_train_score'][self._trained_model.best_index_]*100))
        print('Test score mean: {:.2f}'.format(self._trained_model.cv_results_['mean_test_score'][self._trained_model.best_index_]*100))
        print(f'F2 score: {metrics.fbeta_score(self._test_tgt_y, self._test_pred, average="binary", beta=2)}')
        print(f'Balanced accuracy: {metrics.balanced_accuracy_score(self._test_tgt_y, self._test_pred)}')
        print(f'Recall: {metrics.recall_score(self._test_tgt_y, self._test_pred, average="binary")}')
        
        # Store confusion matrix and classification report
        self.conf_matrix = pd.DataFrame(
            metrics.confusion_matrix(self._test_tgt_y, self._test_pred, labels=[1, 0]),
            columns=['Pred:1', 'Pred:0'],
            index=['True:1', 'True:0']
        )
        
        self.cls_report_df = pd.DataFrame(
            metrics.classification_report(
                self._test_tgt_y,
                self._test_pred,
                output_dict=True
            )
        )
        print(self.cls_report_df)

    def _setup_pipeline_for_tuning(self, model_name: str):
        """
        Build full model pipeline and parameter grid for tuning.

        Args:
            model_name (str): Short name of the model.

        Returns:
            tuple: (whole_pipeline, pipe_param_grid, mdl_param_grid)
        """
        model = self._initialize_model(model_name)
        mdl_param_grid = self._get_param_grid(model_name)

        # Preprocessing pipeline
        preprocessor = self._build_preprocessor([
            ('type convert', sf.data_type_prep),
            ('domain drop', sf.domain_drop_prep),
            ('feat engi', sf.feat_engi_prep),
            ('scale, bin, encode', sf.col_trans)
        ])

        # Feature selectors
        feature_selector_by_stats = self._build_feature_selector_by_stats(
            numerical_pipe_components=[
                ('var filter', uf.VarFilterTransformer()),
                ('corr filter', uf.CorrFilterTransformer()),
                ('num K Best', 'passthrough')
            ],
            categorical_pipe_components=[('cat K Best', 'passthrough')]
        )
        feature_selector_by_model = self._build_feature_selector_by_model('rf')

        # Model pipeline
        imba_mdl_pipeline = self._build_model_pipeline(
            model_name, 
            model,
            None,
            feature_selector_by_stats,
            feature_selector_by_model
        )
        #mdl_param_grid['imba__selector2__max_features'] = [6, 8, 10]

        # Wrap full pipeline
        whole_pipeline = skl_pipe([
            ('prep', preprocessor),
            ('imba', imba_mdl_pipeline)
        ])

        # Convert param grid to pipeline context
        pipe_param_grid = self._convert_param_grid_for_pipeline(model_name, mdl_param_grid)

        return whole_pipeline, pipe_param_grid, mdl_param_grid
    
    def _generate_visualizations(self): #FIXME
        """
        Generate model evaluation visualizations.
        """
        # Parameter combinations plot
        uf.param_combinations( 
            self._trained_model, 
            self._mdl_param_grid, 
            ['max_depth', 'n_estimators']
        )
        
        classifer = self._trained_model.best_estimator_[-1][-1]
        # XGBoost tree visualization if applicable
        '''
        if isinstance(classifer, (xgboost.Booster, xgboost.XGBModel)):
            fig, ax = plt.subplots(figsize=(30, 30))
            xgboost.to_graphviz(classifer, num_trees=0).render('xgboost_tree')
            xgboost.plot_tree(classifer, num_trees=0, rankdir='LR')
            plt.show()
        '''
        self.feature_importance_analysis()
        self.shap_analysis()

    # Public methods ===========================================================
    
    def split_data(
        self, 
        split_ratio: float = DEFAULT_SPLIT_RATIO,
        random_state: int = None
    ): #TODO add TimeSeriesSplit or other splitter for time series data
        """
        Split data into training and test sets.
        
        Args:
            split_ratio: Proportion of data to use for testing
            random_state: Random seed for reproducibility (uses class seed if None)
        """
        rs = random_state if random_state is not None else self.seed
        (self._train_X, self._test_X, 
         self._train_tgt_y, self._test_tgt_y) = model_selection.train_test_split(
            self._data_for_mdl[self._features],
            self._data_for_mdl[self._target],
            test_size=split_ratio,
            random_state=rs
        )
        
        # Reset indices
        self._train_X = self._train_X.reset_index(drop=True)
        self._test_X = self._test_X.reset_index(drop=True)
        self._train_tgt_y = self._train_tgt_y.reset_index(drop=True)
        self._test_tgt_y = self._test_tgt_y.reset_index(drop=True)
        
    def model_training_without_tuning(
        self, 
        model_name: str, 
        cv_or_not: bool = True
    ):
        """
        Train the specified model on the training data.

        Args:
            model_name: Short name of the model to train
            cv_or_not (bool): Whether to use cross-validation
        """
        print(f'Starting initial training for {model_name} model...')
        self._func_disabler('No training under model deployment phase')
        
        if self._train_X is None:
            raise RuntimeError("Cannot proceed because not yet split the data")

        model = self._initialize_model(model_name)

        #xgb_mdl = XGBClassifier(scale_pos_weight=13)
        model.fit(self._train_X, self._train_tgt_y)

        # Count model versions to determine save name
        mdl_version_cnt = self._get_model_version_count()
        
        # Save model and parameters
        self._save_model(model, mdl_version_cnt)
        
        # Load the trained model
        self.get_trained_model(mdl_version_cnt)
        
        print(f'Model {model_name} tuning completed and exported as version {mdl_version_cnt}')

    def model_training_with_tuning(self, model_name: str):
        """
        Perform hyperparameter tuning for the specified model.

        Args:
            model_name (str): Short name of the model.
        """
        self._func_disabler('No tuning under model deployment phase')
        print(f'Starting tuning for {model_name} model...')

        if self._train_X is None:
            raise RuntimeError("Cannot proceed because not yet split the data")

        whole_pipeline, pipe_param_grid, mdl_param_grid = self._setup_pipeline_for_tuning(model_name)

        f_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)
        cv = self._get_cv_strategy(model_name)

        grid_search = model_selection.GridSearchCV(
            whole_pipeline,
            param_grid=pipe_param_grid,
            scoring=f_scorer,
            return_train_score=True,
            cv=cv,
            n_jobs=1, # Must be 1 for GPU mode
            verbose=1
        )

        grid_search.fit(self._train_X, self._train_tgt_y)
        self._trained_model = grid_search
        self._test_pred = grid_search.predict(self._test_X)

        mdl_version_cnt = self._get_model_version_count()
        self._save_model(grid_search, mdl_version_cnt)
        self._save_params_info(mdl_param_grid, mdl_version_cnt)
        self.get_trained_model(mdl_version_cnt)
        self._get_model_params()
        
        print(f'Model {model_name} tuning completed and exported as version {mdl_version_cnt}')
    
    def get_trained_model(
            self, 
            model_version:int=1,
    ):
        """
        Load a trained model from disk.
        
        Args:
            model_version: Version number of the model to load
        """
        if model_version < 1:
            raise ValueError('Model version must be at least 1')
        
        model_path = os.path.join(
            self._parent_dir,
            'saves',
            f'{self._target}_v{model_version}.pkl' if model_version > 1 else f'{self._target}.pkl'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'No model found at {model_path}')
        
        self._trained_model = dill.load(open(model_path, 'rb'))
        self._mdl_param_grid = self._trained_model.get_params()['param_grid']
        self._best_model = self._trained_model.best_estimator_

        
        if hasattr(self._best_model, 'named_steps'):
            # Find the step before the classifier
            steps = list(self._best_model.named_steps.keys())
            classifier_pipe_name = steps[-1]  # typically last step
            preprocessor_pipe_name = steps[-2] if len(steps) > 1 else None
            
        try:  
            step_pre_classifier = self._best_model.named_steps[classifier_pipe_name][-2]  # step before classifier
            pipe_with_only_classifier = None # only assign value if error occurs
        except IndexError:
            step_pre_classifier = self._best_model.named_steps[preprocessor_pipe_name][-1] if preprocessor_pipe_name else None
            pipe_with_only_classifier = self._best_model.named_steps[classifier_pipe_name]
        
        try:
            if hasattr(pipe_with_only_classifier, 'feature_names_in_'):
                self._features_picked = pipe_with_only_classifier.feature_names_in_
            elif hasattr(step_pre_classifier, 'get_feature_names_out'):
                self._features_picked = step_pre_classifier.get_feature_names_out()
            elif hasattr(step_pre_classifier, 'feature_names_in_'): #assume in = out, so use input as picked features
                self._features_picked = step_pre_classifier.feature_names_in_
            else:
                self._features_picked = None
                print("No feature names found in the model's preprocessing steps.")
        except AttributeError as e:
            print(f"Error occurred while loading model features: {e}")
        print(f"Model loaded successfully:\nType: {type(self._trained_model)}\nVersion: {model_version}")

    def model_prediction(self, export_csv: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """
        Generate predictions using the trained model.
        
        Args:
            export_csv: Whether to save predictions to a CSV file
            
        Returns:
            Array of predictions or DataFrame if export_csv is True
        """
        if self._trained_model is None:
            raise Exception("Please either fit a new model or get an existing model first")
        
        if self._phase == 'deploy':
            self._mdl_predictions = self._trained_model.predict(self._data_for_mdl)
            preds_to_export = pd.DataFrame({
                'id': self._data_for_mdl.index.values,
                self._target: self._mdl_predictions
            })
        else:
            if self._test_X is None:
                raise RuntimeError("Cannot proceed because not yet split the data")
            self._mdl_predictions = self._trained_model.predict(self._test_X)
            
            try:
                mdl_proba = self._best_model.predict_proba(self._test_X)
                preds_to_export = pd.DataFrame({
                    'id': self._test_X.index.values,
                    self._target: self._mdl_predictions,
                    'Probability of Class 0': mdl_proba[:, 0],
                    'Probability of Class 1': mdl_proba[:, 1]
                })
            except AttributeError:
                preds_to_export = pd.DataFrame({
                    'id': self._test_X.index.values,
                    self._target: self._mdl_predictions
                })
        
        if export_csv:
            os.makedirs(os.path.join(self._parent_dir, 'results'), exist_ok=True)
            preds_to_export.to_csv(
                os.path.join(self._parent_dir, 'results', f'{self._target} - model predictions.csv'),
                index=False
            )
        
        print(f'Predictions generated successfully. {self._target} prediction: {self._mdl_predictions}')
        return self._mdl_predictions if not export_csv else preds_to_export
    
    def model_evaluation(self): #FIXME to review
        """
        Evaluate model performance with various metrics and visualizations.
        """
        self._func_disabler("No evaluation under deployment phase")
        
        if self._trained_model is None and self._best_model is None:
            raise Exception("Please either fit a new or import an existing model first.")

        # Get transformed data before modeling
        try: #FIXME
            if hasattr(self._best_model[:-1], 'transform'):
                if hasattr(self._best_model[-1][:-1], 'transform'):
                    # If the step before model a transformer, apply it to the transformed data
                    self._train_data_pre_mdl = pd.DataFrame(
                        self._best_model[-1][:-1].transform(
                            self._best_model[:-1].transform(self._train_X)
                        ),
                        columns=self._features_picked
                    )

                    self._prediction_data_pre_mdl = pd.DataFrame(
                        self._best_model[-1][:-1].transform(
                             self._best_model[:-1].transform(self._test_X)
                        ), 
                        columns=self._features_picked
                    )
                else:
                    self._train_data_pre_mdl = pd.DataFrame(
                        self._best_model[:-1].transform(self._train_X), 
                        columns=self._features_picked
                    )
                    self._prediction_data_pre_mdl = pd.DataFrame(
                        self._best_model[:-1].transform(self._test_X), 
                        columns=self._features_picked
                    )
        except AttributeError:
            pass

        # Confusion matrix and classification report
        self.conf_matrix = pd.DataFrame(
            metrics.confusion_matrix(self._test_tgt_y, self._mdl_predictions, labels=[1, 0]),
            columns=['Pred:1', 'Pred:0'],
            index=['True:1', 'True:0']
        )
        
        self.cls_report_df = pd.DataFrame(
            metrics.classification_report(
                self._test_tgt_y,
                self._mdl_predictions,
                output_dict=True
            )
        )

        print(metrics.fbeta_score(self._test_tgt_y, self._mdl_predictions, average="binary", beta=2))

        # Visualizations
        self._generate_visualizations()

    def feature_importance_analysis(self):
        """
        Perform feature importance analysis and plot results.
        """
        classifer = self._trained_model.best_estimator_[-1][-1]
        if hasattr(classifer, 'feature_importances_'):
            print("\nFeature importance analysis")
            importance_dict = dict(zip(self._features_picked, classifer.feature_importances_))
            sorted_importance = dict(sorted(importance_dict.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True))
            print(sorted_importance)
            
            fig, axs = plt.subplots(2, 1, figsize=(12, 14))
            plt.subplots_adjust(hspace=0.4)
            
            if isinstance(classifer, xgboost.XGBClassifier):
                if len(self._features_picked) == classifer.n_features_in_:
                    classifer.get_booster().feature_names = list(self._features_picked)

                    # Plot both graphs without default values
                    for ax, imp_type in zip(axs, ['weight', 'gain']):
                        xgboost.plot_importance(
                            classifer.get_booster() if imp_type == 'gain' else classifer,
                            ax=ax,
                            height=0.8,
                            title=f'Feature importance - {"appearance frequency" if imp_type == "weight" else "average gain of splits"}',
                            xlabel='F score' if imp_type == 'weight' else 'Gain',
                            importance_type=imp_type,
                            max_num_features=30,
                            grid=False,
                            show_values=False  # Disable default values
                        )
                        
                        # Add custom formatted values to each bar
                        for rect in ax.patches:
                            width = rect.get_width()
                            if width > 0:  # Only label bars with positive width
                                ax.text(width + 0.01,  # Small offset from bar end
                                    rect.get_y() + rect.get_height()/2, 
                                    f'{width:.2f}',
                                    ha='left',
                                    va='center',
                                    fontsize=8)
                    
                    # Improve overall readability
                    for ax in axs:
                        ax.tick_params(axis='y', labelsize=10)
                        ax.set_ylabel('Features', fontsize=10)
                        ax.title.set_size(12)

                    if any(len(f) > 15 for f in self._features_picked):
                        for ax in axs:
                            ax.tick_params(axis='y', rotation=30)
            
            # Save high quality image
            output_dir = os.path.join(self._parent_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(
                os.path.join(output_dir, 'feature_importance_analysis.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            plt.show()
            plt.close(fig)
        else:
            print("Warning: Feature importance not available for this model type.")
        
    def shap_analysis(self):
        """Perform SHAP analysis and generate summary plot."""
        # SHAP analysis
        print('SHAP analysis')
        # Skip SHAP for DummyClassifier
        classifier = self._trained_model.best_estimator_[-1][-1]
        if isinstance(classifier, DummyClassifier):
            print("Warning: SHAP analysis not supported for DummyClassifier")
            return None

        explainer = shap.Explainer(classifier)
        shap_values = explainer.shap_values(self._prediction_data_pre_mdl)
        
        fig = plt.figure()
        shap.summary_plot(shap_values, self._prediction_data_pre_mdl)
        fig.savefig(os.path.join(self._parent_dir, 'results', 'SHAP analysis.png'))
        plt.close(fig)
    
    def pr_auc_analysis(self): #FIXME to review
        """Perform PR AUC analysis and generate plot."""
        print("\nPR AUC analysis")
        
        # Baseline model
        bl_mdl = DummyClassifier(strategy='stratified')
        bl_mdl.fit(self._train_data_pre_mdl, self._train_tgt_y)
        naive_probs = bl_mdl.predict_proba(self._prediction_data_pre_mdl)[:, 1]
        bl_pr_auc = metrics.average_precision_score(self._test_tgt_y, naive_probs)
        print(f'No skill PR AUC: {bl_pr_auc:.3f}')
        
        # Model performance
        mdl_probs = self._trained_model.predict_proba(self._test_X)[:, 1]
        mdl_pr_auc = metrics.average_precision_score(self._test_tgt_y, mdl_probs)
        print(f"Model PR AUC: {mdl_pr_auc:.3f}")
        
        # Plot PR curve
        uf.plot_pr_curve(
            self._test_tgt_y, 
            'custom model', 
            naive_probs, 
            mdl_probs, 
            self._parent_dir
        )


class ML_Model_Panel_Data(ML_Model):
    """
    A specialized ML model class for handling panel data.
    
    Inherits from ML_Model and overrides methods to accommodate panel data structure.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the ML_Model_Panel_Data class.
        """
        super().__init__(*args, **kwargs)
        print('Model for panel data')

    def split_data(self, split_ratio=ML_Model.DEFAULT_SPLIT_RATIO, random_state=None):
        """
        Split panel data into training and test sets.

        Args:
            split_ratio (float): Proportion of data to use for testing.
            random_state (int, optional): Random seed for reproducibility.
        """
        rs = random_state if random_state is not None else self.seed
        self._func_disabler('No data split under model deployment phase')
        print(f'Splitting data with ratio {split_ratio}...')

        # Split the data into training and test sets
        self._train_X = self._data_for_mdl[:int(len(self._data_for_mdl)*(1-split_ratio))]
        self._test_X = self._data_for_mdl[int(len(self._data_for_mdl)*(1-split_ratio)):]
        self._train_X = self._train_X.drop(columns=[self.target])
        self._test_X = self._test_X.drop(columns=[self.target])

        self._train_tgt_y = self._data_for_mdl[:int(len(self._data_for_mdl)*(1-split_ratio))][self.target]
        self._test_tgt_y = self._data_for_mdl[int(len(self._data_for_mdl)*(1-split_ratio)):][self.target]

        # Reset indices
        self._train_X = self._train_X.reset_index(drop=True)
        self._test_X = self._test_X.reset_index(drop=True)
        self._train_tgt_y = self._train_tgt_y.reset_index(drop=True)
        self._test_tgt_y = self._test_tgt_y.reset_index(drop=True)

    def _setup_pipeline_for_tuning(self, model_name: str):
        """
        Build full model pipeline and parameter grid for tuning.

        Args:
            model_name (str): Short name of the model.

        Returns:
            tuple: (whole_pipeline, pipe_param_grid, mdl_param_grid)
        """
        model = self._initialize_model(model_name)
        mdl_param_grid = self._get_param_grid(model_name)

        # Preprocessing pipeline
        domain_drop_file_name = "domain_drop"
        irrelevant_cols = pd.read_csv(os.path.join(self._parent_dir, 'configurations', f'{domain_drop_file_name}.csv')).values.flatten().tolist()

        preprocessor = self._build_preprocessor(
            components=[
                #('TypeConvert', sf.data_type_prep),
                ('DomainDrop', sf.domain_drop_prep),
            ],
            params=[('DomainDrop', {"irrelevant_cols": irrelevant_cols})]
        )

        # Model pipeline
        imba_mdl_pipeline = self._build_model_pipeline(
            model_name, 
            model,
            None,
            self._build_feature_selector_by_model('rf')
        )

        # Wrap full pipeline
        whole_pipeline = skl_pipe([
            ('prep', preprocessor),
            ('imba', imba_mdl_pipeline)
        ])

        # Convert param grid to pipeline context
        pipe_param_grid = self._convert_param_grid_for_pipeline(model_name, mdl_param_grid)

        return whole_pipeline, pipe_param_grid, mdl_param_grid


# In[9] Main execution block

def main(model_to_train_or_get=None, tuning:bool=False, split_ratio=0.2, random_state=42):
    """
    Main function to execute the ML model training and evaluation.

    Args:
        model_to_train_or_get: Model name or version to train or load.
        tuning (bool): Whether to perform hyperparameter tuning.
        split_ratio (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    """
    
    if model_to_train_or_get is None:
        raise ValueError("Please specify either a model to train or a model version to load.")

    ml_model = ML_Model_Panel_Data(modeling_phase='training', target_col='is_fraud')
    
    if isinstance(model_to_train_or_get, str) and model_to_train_or_get not in ml_model.MODEL_MAPPING.keys():
        raise ValueError(f"Invalid model name: {model_to_train_or_get}. Choose from {list(ml_model.MODEL_MAPPING.keys())}")

    # Example usage
    ml_model.split_data(split_ratio=split_ratio, random_state=random_state)

    if isinstance(model_to_train_or_get, numbers.Number):
        ml_model.get_trained_model(model_version=model_to_train_or_get)
    elif isinstance(model_to_train_or_get, str) and tuning == True:
        ml_model.model_training_with_tuning(model_name=model_to_train_or_get)
    elif isinstance(model_to_train_or_get, str) and tuning == False:
        ml_model.model_training_without_tuning(model_name=model_to_train_or_get)

    ml_model.model_prediction(export_csv=True)
    ml_model.model_evaluation()
    

# In[0]

if __name__ == '__main__':
    main(model_to_train_or_get=8)
    #main(model_to_train_or_get='xgb', tuning=True, split_ratio=0.2, random_state=42)


# %%

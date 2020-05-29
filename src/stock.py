#import os
import random
import numpy as np
import pandas as pd
#import pandas_profiling as pp
#import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
#import scikitplot as skplt

from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import PowerTransformer
from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from collections import defaultdict
from warnings import filterwarnings

class Stock:
    def __init__(self, path: str='./', seed: int=1234, gridsearch: bool=None, save: bool=False):
        self.ROOT_PATH = path
        self.DATA_PATH = self.ROOT_PATH + 'data/'
        self.TRAIN_PATH = self.ROOT_PATH + 'data/train.csv'
        self.TEST_PATH = self.ROOT_PATH + 'data/test.csv'
        self.PROFILE_REPORT_PATH = self.DATA_PATH
        self.RESULT_CSV_PATH = self.DATA_PATH + 'submission.csv'
        self.GRIDSEARCH = gridsearch
        self.SAVE = save

        self.data_plot_no_nan = None

        self.num_features_final = None
        self.cat_features_nominal_final = None
        self.cat_features_ordinal_final = None
        self.cat_transformer_nominal = None

        self.num_transformer = None

        self.preprocessor_Y = None
        self.preprocessor_X = None

        self.best_estimator = None

        self.SEED = seed
        random.seed(self.SEED)

        print("Loading data...")
        self.train_df = pd.read_csv(self.TRAIN_PATH)
        # We need to sort by 'Feature_7'. See Data Analysis section.
        self.train_df.sort_values(by=['Feature_7'])
        self.test_df = pd.read_csv(self.TEST_PATH)

        # Aggregate intraday returns
        intraday_rets = []
        rets = ['Ret_MinusTwo', 'Ret_MinusOne']
        train_aggregated_rets = pd.DataFrame(columns=['Ret_Agg', 'Ret_Agg_Std', 'Ret_Std', ])
        test_aggregated_rets = pd.DataFrame(columns=['Ret_Agg', 'Ret_Agg_Std', 'Ret_Std'])

        for i in range(2, 121): intraday_rets.append(f'Ret_{i}')

        train_aggregated_rets['Ret_Agg'] = self.train_df[intraday_rets].sum(axis=1)
        train_aggregated_rets['Ret_Agg_Std'] = self.train_df[intraday_rets].std(axis=1)
        train_aggregated_rets['Ret_Std'] = self.train_df[rets].std(axis=1)
        self.train_df = pd.concat([self.train_df, train_aggregated_rets], axis=1)

        test_aggregated_rets['Ret_Agg'] = self.test_df[intraday_rets].sum(axis=1)
        test_aggregated_rets['Ret_Agg_Std'] = self.test_df[intraday_rets].std(axis=1)
        test_aggregated_rets['Ret_Std'] = self.test_df[rets].std(axis=1)
        self.test_df = pd.concat([self.test_df, test_aggregated_rets], axis=1)

        # Prepare train, validation and test data
        self.features = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6','Feature_7', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12','Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18','Feature_19', 'Feature_20', 'Feature_21', 'Feature_22', 'Feature_23', 'Feature_24','Feature_25', 'Ret_MinusTwo', 'Ret_MinusOne', 'Ret_Agg', 'Ret_Agg_Std', 'Ret_Std', ]
        self.targets = ['Ret_PlusOne', 'Ret_PlusTwo']
        weights_intraday = 'Weight_Intraday'
        weights_daily = 'Weight_Daily'
        weights = [weights_intraday, weights_daily]
        self.features_targets = self.features + self.targets

        self.train_X_Y_df = self.train_df[self.features + self.targets]
        self.train_X_df = self.train_df[self.features]
        self.train_Y_df = self.train_df[self.targets]
        self.train_weights_daily_df = self.train_df[weights_daily]
        self.test_X_df = self.test_df[self.features]

        print('Data loaded')
        print(f'Shape of training feature data: {self.train_X_df.shape}')
        print(f'Shape of training target data: {self.train_Y_df.shape}')
        print(f'Shape of test feature data: {self.test_X_df.shape}')

    def __outliers(self, col) -> int:
        std3 = col.std() * 3
        mean = col.mean()
        c = 0
        for row in col:
            if (abs(row - mean) > std3): c = c + 1
        return c

    def __analyze_df(self, name: str, df_train: pd.DataFrame, df_test: pd.DataFrame=None, percentage: bool=True) -> pd.DataFrame:
        test_set = ()
        vals = []
        vals_percent = []
        for col in df_train:
            if df_test is not None: test_set = set(df_test[col])
            switcher = {
                'Missing': sum(df_train[col].isnull()),
                'Unique': len(df_train[col].unique()),
                'Imbalance': df_train[col].value_counts().values[0],
                'Outlier': self.__outliers(df_train[col]),
                'Disjoint': set(df_train[col]).isdisjoint(test_set)
            }
            val = switcher.get(name)
            vals.append(val)
            vals_percent.append(val/len(df_train[col])*100)
        if percentage: res_df = pd.DataFrame(list(zip(vals, vals_percent)), columns=[name, f'{name} %'])
        else: res_df = pd.DataFrame(list(zip(vals)), columns=[name])

        return res_df

    def analysis1(self):
        # OBS: This is very compute intensive
        print('Analyzing data (this will take a while)...')
        missing_data = self.__analyze_df('Missing', self.train_X_Y_df)
        unique_data = self.__analyze_df('Unique', self.train_X_Y_df)
        balanced_data = self.__analyze_df('Imbalance', self.train_X_Y_df)
        outlier_data = self.__analyze_df('Outlier', self.train_X_Y_df)
        disjoint_data = self.__analyze_df('Disjoint', self.train_X_df, self.test_X_df, False)
        analyze_data = pd.concat([pd.DataFrame(self.train_X_Y_df.columns), missing_data, unique_data, balanced_data, outlier_data, disjoint_data], axis=1)
        print("Data Analysis:")
        print(analyze_data)

        return self

    def analysis2(self):
        feature_groups = defaultdict(list)
        for i, row in self.train_df.iterrows():
            val = row['Ret_PlusOne']
            if val > 0: feature_groups[row['Feature_7']].append(1)
            else: feature_groups[row['Feature_7']].append(-1)

        freq = 0
        for key, val in feature_groups.items():
            frq0 = val.count(1)/len(val)
            frq1 = len(val) - frq0
            frq = max(frq0, frq1)
            freq += frq/len(val)
        freq = freq / len(feature_groups)

        print(f'Frequence of return signs: {freq}')

        return self

    def __plot(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, transformer: Pipeline = None, frac: float = 0.1, label: str = ''):

        train_data = train_df.sample(frac=frac, random_state=0)
        test_data = test_df.sample(frac=frac, random_state=0)

        if transformer is not None:
            train_data = transformer.fit_transform(train_data[features])
            train_data_df = pd.DataFrame(train_data, columns=features)
            test_data = transformer.transform(test_data[features])
            test_data_df = pd.DataFrame(test_data, columns=features)
        else:
            imputer = SimpleImputer(strategy='constant')
            train_data_df = pd.DataFrame(imputer.fit_transform(
                train_df[features]), columns=features)
            test_data_df = pd.DataFrame(imputer.fit_transform(
                test_df[features]), columns=features)

        print(label)
        fig, ax = plt.subplots(round(len(features) / 3), 3, figsize=(15, 15))
        for i, ax in enumerate(fig.axes):
            if i < len(features):
                sns.distplot(train_data_df[features[i]], color='blue', ax=ax)
                sns.distplot(test_data_df[features[i]], color='red', ax=ax)

    def __initialize_pre_plot(self):
        data_plot = self.train_X_Y_df.sample(frac=0.1, random_state=0)
        imputer = SimpleImputer(strategy='constant')
        self.data_plot_no_nan = pd.DataFrame(imputer.fit_transform(data_plot), columns=self.features_targets)
        self.num_features = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_6','Feature_11', 'Feature_14', 'Feature_15','Feature_17', 'Feature_18', 'Feature_19','Feature_21', 'Feature_22', 'Feature_23', 'Feature_24', 'Feature_25','Ret_MinusTwo', 'Ret_MinusOne', 'Ret_Agg']
        self.cat_features = ['Feature_5', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10','Feature_12', 'Feature_13', 'Feature_16', 'Feature_20']

    def pre_plot1(self):
        # Correlation heatmap for all features and targets
        print("Correlation heatmap:")
        if not self.data_plot_no_nan: self.__initialize_pre_plot()
        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(self.data_plot_no_nan.corr(), annot=True, ax=ax)
        plt.show()

    def pre_plot2(self):
        # Correlation heatmap for numerical features
        print("Correlation heatmap for numerical features:")
        if not self.data_plot_no_nan: self.__initialize_pre_plot()
        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(self.data_plot_no_nan[self.num_features].corr(), annot=True, ax=ax)
        plt.show()

    def pre_plot3(self):
        # Correlation heatmap for categorical features
        print("Correlation heatmap for categorical features:")
        if not self.data_plot_no_nan: self.__initialize_pre_plot()
        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(self.data_plot_no_nan[self.cat_features].corr(), annot=True, ax=ax)
        plt.show()

    def pre_plot4(self):
        # Distributions
        self.__plot(self.train_X_df, self.test_X_df, features=self.features, label='Raw distributiions: Features')
        plt.show()

    def pre_plot5(self):
        # Regression plots
        print("Regression plots:")
        fig, ax = plt.subplots(round(len(self.features) / 3), 3, figsize=(15, 15))
        if not self.data_plot_no_nan: self.__initialize_pre_plot()
        for i, ax in enumerate(fig.axes):
            if i < len(self.features): sns.regplot(x=self.features[i], y=self.targets[0], data=self.data_plot_no_nan, ax=ax)
        plt.show()

    def select_features(self):
        # Define final features
        print("Selecting features...")
        self.num_features_final = ['Feature_2', 'Feature_3', 'Feature_4', 'Feature_6','Feature_11', 'Feature_14','Feature_17', 'Feature_18', 'Feature_19','Feature_21', 'Feature_22', 'Feature_23', 'Feature_24', 'Feature_25','Ret_MinusTwo', 'Ret_MinusOne', 'Ret_Agg', 'Ret_Agg_Std','Ret_Std']

        self.cat_features_ordinal_final = ['Feature_13']

        self.cat_features_nominal_final = ['Feature_1', 'Feature_5', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10','Feature_12', 'Feature_15', 'Feature_16', 'Feature_20']

        self.cat_features_final = self.cat_features_ordinal_final + self.cat_features_nominal_final
        self.features_final = self.num_features_final + self.cat_features_final

        self.train_X_df = self.train_df[self.features_final]
        self.test_X_df = self.test_df[self.features_final]

        return self

    class CutOff(TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            X[X > 3] = 3
            X[X < -3] = -3
            return X

    def data_processing(self):
        # Preprocessing for numerical data
        print("Begin Data Processing...")
        if not (self.num_features_final and self.cat_features_nominal_final and self.cat_features_ordinal_final): self.select_features()

        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('scale', RobustScaler(quantile_range=[5, 95])),
            ('quantile', QuantileTransformer(n_quantiles=300, output_distribution='normal', random_state=0)),
            ('cutoff', self.CutOff()),  # Cut off at 3 standard deviations
            ('norm', Normalizer(norm='l2'))
        ])

        # Preprocessing for nominal categorical data
        self.cat_transformer_nominal = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('pca', PCA(whiten=True, random_state=0)),
            ('bins', KBinsDiscretizer(n_bins=100, encode='onehot', strategy='quantile')),
            ('norm', Normalizer(norm='l2')),
        ])

        # Preprocessing for ordinal categorical data
        self.cat_transformer_ordinal = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('bins', KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')),
            ('norm', Normalizer(norm='l2')),
        ])

        # Combined preprocessing for numerical and categorical data
        self.preprocessor_X = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, self.num_features_final),
                ('cat_nom', self.cat_transformer_nominal, self.cat_features_nominal_final),
                ('cat_ord', self.cat_transformer_ordinal, self.cat_features_ordinal_final)
            ])

        # Testing preprocessor
        preprocessor_X_shape = self.preprocessor_X.fit_transform(self.train_X_df).shape
        print(f'preprocessor_X output shape: {preprocessor_X_shape}')

        # Target transformer
        self.preprocessor_Y = Pipeline(steps=[('quantile', QuantileTransformer(n_quantiles=300, output_distribution='normal', random_state=0))])

        return self

    def __initialize_post_plot(self):
        # Transform numerical features and targets
        if not (self.num_transformer and self.cat_transformer_nominal): self.data_processing()

        data_plot = self.train_X_Y_df.sample(frac=0.1, random_state=0)
        imputer = SimpleImputer(strategy='constant')
        data_plot_no_nan = pd.DataFrame(imputer.fit_transform(data_plot), columns=self.features_targets)

        num_transformed = self.num_transformer.fit_transform(data_plot_no_nan[self.num_features_final])
        viz_train_X_num_df = pd.DataFrame(num_transformed, columns=self.num_features_final)
        self.viz_train_Y_df = pd.DataFrame(self.preprocessor_Y.fit_transform(self.train_Y_df), columns=self.targets)
        self.viz_train_X_Y_num_df = pd.concat([viz_train_X_num_df, self.viz_train_Y_df], axis=1)

        cat_transformed = self.cat_transformer_nominal.fit_transform(self.num_transformer.fit_transform(data_plot_no_nan[self.cat_features_final]))
        self.dim = 20
        self.cat_features_pca = []
        for i in range(0, self.dim): self.cat_features_pca.append(f'{i}')
        cat_transformed_pca = TruncatedSVD(n_components=self.dim).fit_transform(cat_transformed)
        self.viz_train_X_cat_df = pd.DataFrame(cat_transformed_pca, columns=self.cat_features_pca)
        self.viz_train_X_Y_cat_df = pd.concat([self.viz_train_X_cat_df, self.viz_train_Y_df], axis=1)

    def post_plot1(self):
        # Distributions of numerical features after preprocssesing
        if not self.num_transformer: self.data_processing()
        self.__plot(self.train_X_df, self.test_X_df, features=self.num_features_final, transformer=self.num_transformer,label='After preproccssing numerical features: Distributions:')
        plt.show()

    def post_plot2(self):
        # Regression plots of numerical features after preprocssesing
        print("After preproccssing numerical features: Regression plots:")
        self.__initialize_post_plot()
        fig, ax = plt.subplots(
            round(len(self.num_features_final) / 3), 3, figsize=(15, 15))
        for i, ax in enumerate(fig.axes):
            if i < len(self.num_features_final): sns.regplot(x=self.num_features_final[i], y=self.targets[0], data=self.viz_train_X_Y_num_df, ax=ax)
        plt.show()

    def post_plot3(self):
        # Distributions of categorical features after preprocssesing
        print("After preproccssing categorical features: Distributions:")
        self.__initialize_post_plot()
        fig, ax = plt.subplots(round(self.dim / 3), 3, figsize=(15, 15))
        for i, ax in enumerate(fig.axes):
            if i < self.dim: sns.distplot(self.viz_train_X_cat_df.iloc[i], ax=ax)
        plt.show()

    def post_plot4(self):
        # Regression plots categorical features after preprocssesing
        print("After preproccssing categorical features: Regression plots:")
        self.__initialize_post_plot()
        fig, ax = plt.subplots(round(self.dim / 3), 3, figsize=(15, 15))
        for i, ax in enumerate(fig.axes):
            if i < self.dim: sns.regplot(x=self.cat_features_pca[i], y=self.targets[0], data=self.viz_train_X_Y_cat_df, ax=ax)
        plt.show()

    def post_plot5(self):
        # Distributions of targets after preprocssesing
        print("After preproccssing targets: Distributions:")
        self.__initialize_post_plot()
        fig, ax = plt.subplots(round(len(self.targets) / 3), 3, figsize=(15, 5))
        for i, ax in enumerate(fig.axes):
            if i < len(self.targets): sns.distplot(self.viz_train_Y_df[self.targets[i]], ax=ax)
        plt.show()

    def __WA(self, a, axis, weight):
        # Adapted from function_base.py
        a = np.asanyarray(a)
        wgt = np.asanyarray(weight)
        wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
        wgt = wgt.swapaxes(-1, axis)
        n = len(a)
        avg = np.multiply(a, wgt).sum(axis)/n

        return avg

    def __WMAE(self, y_true, y_pred, sample_weight):
        # Adapted from regrssion.py
        output_errors = self.__WA(np.abs(y_pred - y_true), weight=sample_weight, axis=0)
        avg = np.average(output_errors)

        return avg

    def build_model(self, GRIDSEARCH : bool = False):

        print('Building model...')
        if not (self.preprocessor_X and self.preprocessor_Y): self.data_processing()

        # Define initial model
        model = LinearSVR(epsilon=0.0, C=0.0005, loss='squared_epsilon_insensitive', random_state=0)  # 1727.860

        # Define model pipeline for multi output regression
        multi_out_reg = MultiOutputRegressor(model)
        model_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor_X), ('multioutreg', multi_out_reg)])
        estimator = TransformedTargetRegressor(regressor=model_pipeline, transformer=self.preprocessor_Y)

        if GRIDSEARCH:
            # Define grid parameters to search
            grid_params = {'regressor__multioutreg__estimator__C': [0.0005, 0.001, 0.0015, 0.002]}

            # Define CV by grouping on 'Feature_7'
            # See: https://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat
            #      http://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
            group = self.train_X_df['Feature_7'].values
            cv = list(GroupKFold(n_splits=5).split(self.train_X_df, self.train_Y_df, group))

            # Define grid search scoring metric
            scoring = 'neg_mean_absolute_error'

            # Define grid search specified scoring and cross-validation generator
            print('Running grid searc CV...')
            gd_sr = GridSearchCV(estimator=estimator, param_grid=grid_params, scoring=scoring, cv=cv, refit=True)

            # Apply grid search and get parameters for best result
            gd_sr.fit(self.train_X_df, self.train_Y_df)
            best_params = gd_sr.best_params_
            self.best_estimator = gd_sr.best_estimator_
            score = -gd_sr.best_score_

            print(f'Best parameters = {gd_sr.best_params_}')
            print(f'Best MAE = {score}')

        else:
            estimator.fit(self.train_X_df, self.train_Y_df)
            self.best_estimator = estimator

        print('Done building model')

        return self

    def evaluate_model(self):
        # Predict on train and validation data
        print('Evaluating model...')
        if not self.best_estimator: self.build_model()
        pred_train_Y = self.best_estimator.predict(self.train_X_df)

        # Evaluate predictions on train and validation data and compare with baseline mean prediction
        mean_Y = [0, 0]
        mean_Y[0] = self.train_df[self.targets[0]].mean()
        mean_Y[1] = self.train_df[self.targets[1]].mean()

        train_mae = self.__WMAE(self.train_Y_df, pred_train_Y, sample_weight=self.train_weights_daily_df)
        mean_Y_np = np.concatenate((np.full((self.train_Y_df.shape[0], 1), mean_Y[0]), np.full((self.train_Y_df.shape[0], 1), mean_Y[1])), axis=1)
        mean_mae = self.__WMAE(self.train_Y_df, mean_Y_np, sample_weight=self.train_weights_daily_df)

        # Print scores
        print('WMAE score: LOWER is BETTER')
        print(f'WMAE of fitted model: {train_mae}')
        print(f'WMAE of baseline model: {mean_mae}')

        # Predict on test data
        self.pred_test_Y = self.best_estimator.predict(self.test_X_df)

        if self.SAVE: self.save()

        return self

    def save(self):

        # Create submission data
        print("Saving Results...")
        if not self.pred_test_Y.any(): self.evaluate_model()
        ids = []
        preds = []
        for i, row in self.test_df.iterrows():
            for j in range(1, 61):
                ids.append(f'{i+1}_{j}')
                # OBS! We predict i_1 - i_60 as 0
                preds.append(0)
            ids.append(f'{i+1}_61')
            preds.append(self.pred_test_Y[i][0])  # D+1
            ids.append(f'{i+1}_62')
            preds.append(self.pred_test_Y[i][1])  # D+2

        submission_df = pd.DataFrame(list(zip(ids, preds)), columns=['Id', 'Predicted'])
        print(submission_df[(submission_df.Predicted != 0)].head(5))

        # Save submission to csv file
        submission_df.to_csv(self.RESULT_CSV_PATH, index=False)

        return self

filterwarnings("ignore")

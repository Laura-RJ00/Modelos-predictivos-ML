import json
import pandas as pd
import numpy as np
from itiia.data_preparation import DataPreparator
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


class ModelSepsisDiagnosis():
    def _get_var_names(self, model_json, typ):
        return  [v['name'] for v in model_json['input'] if v['type'] == typ]
    
    def __init__(self, src='model.json'):
        # load data description model 
        with open(src, 'r') as f:
            model_json = json.load(f)
        # get lists of variable name per type
        LIST_NUMVARS = self._get_var_names(model_json, 'numerical')
        LIST_ORDVARS = self._get_var_names(model_json, 'ordinal')
        LIST_BOOLVARS = self._get_var_names(model_json, 'boolean')
        # define pipeline
        pipeline = [
            ('impute booleans', SimpleImputer(strategy='most_frequent'), {'varnames': LIST_BOOLVARS}),
            # ('impute ordinals', SimpleImputer(strategy='most_frequent'), {'varnames': LIST_ORDVARS}),
            ('impute numericals', SimpleImputer(strategy='median'), {'varnames': LIST_NUMVARS}),
            # ('transform ordinals', OrdinalEncoder(), {'varnames': LIST_ORDVARS})
            # standardscaler segun el estimador 
        ]
        self.prep = DataPreparator(pipeline=pipeline)
        self.model = GradientBoostingRegressor(random_state=0)
        self.columns = []

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X not a dataframe')
        self.columns = X.columns
        xprep = self.prep.fit_transform(X)
        self.model.fit(xprep, y)
        return self
        
    def predict(self, X):
        # En el caso de que sea una lista, asume que el orden de los valores es 
        # el mismo que en tiempo de entrenamiento.
        # En el caso de ser un diccionario, ordena las columnas y añade las 
        # posibles faltantes
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
            
        # unify missing indicator from None to nan
        X = X.applymap(lambda x: np.nan if x is None else x)
        
        xprep = self.prep.transform(X)
        sofatot3 = self.model.predict(xprep)
        delta_sofa = sofatot3 - X['SOFATOT1']

        return delta_sofa

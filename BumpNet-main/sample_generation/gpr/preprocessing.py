import numpy as np

from numbers import Real
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, _name_estimators, _final_estimator_has

class LogitScaler(MinMaxScaler):
    _parameter_constraints: dict = {
        "epsilon": [Real],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, epsilon_0=0, epsilon_1=0, copy=True, clip=False):
        self.epsilon_0 = epsilon_0
        self.epsilon_1 = epsilon_1
        self.copy = copy
        self.clip = clip
        super().__init__(feature_range=(0+epsilon_0, 1-epsilon_1),
                         copy=copy, clip=clip)

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        z = logit(super().transform(X))
        z[np.isinf(z)] = np.nan
        return z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return super().inverse_transform(expit(X))
    
    def error_prop(self, er, X):
        z = super().transform(X)
        zer = er/self.data_range_ # error prop for min/max scaler
        return zer * (1/(z*(1-z)))


class ExtStandardScaler(StandardScaler):
    def error_prop(self, er, X):
        return er / self.scale_

class ExtPipeline(Pipeline):
    def error_propagation(self, er, X):
        for name, step in self.steps:
            if hasattr(step, 'error_prop'):
                er = step.error_prop(er, X)
                X = step.transform(X)  # Update X to the transformed data
            else:
                raise AttributeError(
                    f"Step '{name}' does not support error propagation. "
                    f"Ensure it has an 'error_prop' method."
                )
        return er

def make_ext_pipeline(*steps, memory=None, verbose=False):
    return ExtPipeline(_name_estimators(steps), memory=memory, verbose=verbose)

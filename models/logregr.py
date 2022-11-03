import numpy as np

from sklearn.linear_model import LogisticRegression


class MyAmazingLogisticRegression:
    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args,**kwargs)

    def train(self, inputs, outputs):
        words = np.array(inputs)
        tags = np.ravel(np.array(outputs))
        self.model.fit(words, tags)
        return self.model.coef_()

    def predict(self, inputs):
        words = np.array(inputs)
        return self.model.predict(words)

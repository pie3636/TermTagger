import numpy as np

from sklearn.linear_model import LogisticRegression


class MyLogisticRegression:
    def __init__(self, *args, **kwargs):
        self.classifier = LogisticRegression(*args, **kwargs)

    def train(self, inputs, outputs):
        words = np.array(inputs)
        tags = np.ravel(np.array(outputs))
        self.classifier.fit(words, tags)
        return self.classifier.coef_

    def predict(self, inputs):
        words = np.array(inputs)
        return self.classifier.predict(words)

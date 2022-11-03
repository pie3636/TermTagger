import numpy as np

from sklearn import svm


class SupportVectorMachine:

    def __init__(self, method='svc', *args, **kwargs):
        if method == 'svc':
            self.classifier = svm.SVC(*args, **kwargs)
        elif method == 'linear_svc':
            self.classifier = svm.LinearSVC(*args, **kwargs)
        elif method == 'nu_svc':
            self.classifier = svm.NuSVC(*args, **kwargs)
        else:
            raise ValueError(f'`method` parameter can only be one of [svc, linear_svc, nu_svc]. Got [{method}]')


    def train(self, inputs, outputs):
        inputs = np.array(inputs)
        outputs = np.ravel(np.array(outputs))
        self.classifier.fit(inputs, outputs)


    def predict(self, inputs):
        return self.classifier.predict(inputs)

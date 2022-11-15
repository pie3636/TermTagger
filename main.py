from datasets import SentenceDataset, SlidingWindowDataset, WordDataset
from models.svm import SupportVectorMachine
from models.logregr import MyLogisticRegression
from sklearn.metrics import classification_report

import datasets
import gensim
import numpy as np
import zipfile

from tqdm import tqdm

WINDOW_SIZE = 3

"""
Input feature selection
-----------------------

Possible values:
- POS: Part-of-speech tags (one-hot encoded)
- WL:  Word length
"""
features = ['POS', 'WL']


def get_embeds(dataset, embeds, single_pred=False):
    inputs = []
    outputs = []
    for tokens, *features, tags in tqdm(dataset):
        current_input = np.empty(vector_size*WINDOW_SIZE)
        current_input_features = []
        for i, token in enumerate(tokens):
            if token in embeds:
                current_input[vector_size*i:vector_size*(i+1)] = embeds[token]
            else:
                current_input[vector_size*i:vector_size*(i+1)] = np.zeros(vector_size)
            for feature in features:
                current_input_features.append(feature[i])
        if current_input_features:
            current_input = np.concatenate([current_input, *current_input_features])
        inputs.append(current_input)
        outputs.append(tags[0] if single_pred else tags)
    return inputs, outputs



print('Reading embeddings')
embeds = gensim.models.KeyedVectors.load('embeds/english_fasttext_2017_10', mmap='r')
vector_size = embeds.vector_size

print('Loading and preparing dataset')

train_set = SlidingWindowDataset('data/train', WINDOW_SIZE, uncap_first=True, features=features)
dev_set = SlidingWindowDataset('data/dev', WINDOW_SIZE, uncap_first=True, features=features)
test_set = SlidingWindowDataset('data/test', WINDOW_SIZE, uncap_first=True, features=features)


clfs = [
        ('SVM', SupportVectorMachine()),
        ('LR', MyLogisticRegression())
]

print('Computing input embeddings')
train_inputs, train_outputs = get_embeds(train_set, embeds, single_pred=True)

print('Computing dev embeddings')
dev_inputs, dev_outputs = get_embeds(dev_set, embeds, single_pred=True)

preds = []
for name, clf in tqdm(clfs):
    print(f'Training {name} model')
    clf.train(train_inputs, train_outputs)
    print(f'Computing {name} predictions')
    preds.append(clf.predict(dev_inputs))

dev_outputs = np.ravel(dev_outputs)

for i, (name, _) in enumerate(clfs):
    print(f'{name} results:')
    print(classification_report(dev_outputs, preds[i], target_names=datasets.tag_names, digits=4, zero_division=0))


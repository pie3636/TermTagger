from datasets import SentenceDataset, SlidingWindowDataset, TARSDataset, WordDataset
from evaluation import convert_span_into_iob, get_span_based_scores
from flair.data import Sentence
from models.logregr import MyLogisticRegression
from models.svm import SupportVectorMachine
from models.tarstr import TARSModel, flair_tag_type
from sklearn.metrics import classification_report

import copy
import datasets
import flair
import gensim
import numpy as np
import torch
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

#flair.device = torch.device('cpu')

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

print('Loading and preparing datasets')

train_set = SlidingWindowDataset('data/train', WINDOW_SIZE, uncap_first=True, features=features)
dev_set = SlidingWindowDataset('data/dev', WINDOW_SIZE, uncap_first=True, features=features)
test_set = SlidingWindowDataset('data/test', WINDOW_SIZE, uncap_first=True, features=features)

tars_train_set = TARSDataset('data/train')
tars_dev_set = TARSDataset('data/dev')
tars_test_set = TARSDataset('data/test')
tars_test_set_eval = copy.deepcopy(tars_test_set)

for sent in tars_dev_set:
    sent.remove_labels(flair_tag_type)

for sent in tars_test_set:
    sent.remove_labels(flair_tag_type)

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

print('Training few-shot TARS model')
tars_model = TARSModel([tars_train_set, [Sentence('')], [Sentence('')]])
tars_model.train()

print('Computing TARS predictions')
all_pred = []
all_true = []
for sent_pred, sent_true in zip(tars_test_set, tars_test_set_eval):
    tars_model.predict(sent_pred)
    all_pred.extend(convert_span_into_iob(sent_pred))
    all_true.extend(convert_span_into_iob(sent_true))

for i, (name, _) in enumerate(clfs):
    print(f'{name} results:')
    print(classification_report(dev_outputs, preds[i], target_names=datasets.tag_names, digits=4, zero_division=0))
    precision, recall, f1score = get_span_based_scores(preds[i], dev_outputs)
    print(f'[Span-based] Precision: {precision:.2f}, recall: {recall:.2f}, F-score: {f1score:.2f}')


print('TARS results:')
print(classification_report(all_true, all_pred, digits=4, zero_division=0))

precision, recall, f1score = get_span_based_scores(all_pred, all_true)
print(f'[Span-based] Precision: {precision:.2f}, recall: {recall:.2f}, F-score: {f1score:.2f}')
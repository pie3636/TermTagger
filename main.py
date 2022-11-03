from datasets import SentenceDataset, SlidingWindowDataset, WordDataset
from models.svm import SupportVectorMachine
from sklearn.metrics import classification_report

import datasets
import gensim
import numpy as np
import zipfile

from tqdm import tqdm

WINDOW_SIZE = 3

def get_embeds(dataset, embeds, single_pred=False):
    inputs = []
    outputs = []
    for tokens, tag in tqdm(dataset):
        current_input = np.empty(vector_size*WINDOW_SIZE)
        for i, token in enumerate(tokens):
            if token in embeds:
                current_input[vector_size*i:vector_size*(i+1)] = embeds[token]
            else:
                current_input[vector_size*i:vector_size*(i+1)] = np.zeros(vector_size)
        inputs.append(current_input)
        outputs.append(tag[0] if single_pred else tag)
    return inputs, outputs



print('Reading embeddings')
embeds = gensim.models.KeyedVectors.load('embeds/english_fasttext_2017_10', mmap='r')
vector_size = embeds.vector_size

print('Loading dataset')
train_set = SlidingWindowDataset('data/train', WINDOW_SIZE, uncap_first=True)
dev_set = SlidingWindowDataset('data/dev', WINDOW_SIZE, uncap_first=True)
test_set = SlidingWindowDataset('data/test', WINDOW_SIZE, uncap_first=True)


svm_clf = SupportVectorMachine()

print('Computing input embeddings')
train_inputs, train_outputs = get_embeds(train_set, embeds, single_pred=True)

print('Training model')
svm_clf.train(train_inputs, train_outputs)

print('Computing test embeddings')
test_inputs, test_outputs = get_embeds(test_set, embeds, single_pred=True)

print('Computing predictions')
pred = svm_clf.predict(test_inputs)

test_outputs = np.ravel(test_outputs)
print(classification_report(test_outputs, pred, target_names=datasets.tag_names, digits=4, zero_division=0))


import flair
import torch

from flair.data import Sentence
from flair.models import TARSTagger

flair.device = torch.device('cpu')

print('Loading TARSTagger, this may take a minute...')

tagger = TARSTagger.load('output/final-model.pt')
print()

print('Interactive mode: Type a sentence. The tagged version of the sentence will then be outputed. Stop at any time using Ctrl-C')
print()

while True:
    sent = Sentence(input('Input: '))
    tagger.predict(sent)
    print('Prediction: ', sent)
    print()

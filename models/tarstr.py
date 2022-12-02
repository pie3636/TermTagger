from flair.data import Corpus, Dictionary
from flair.models import TARSTagger
from flair.trainers import ModelTrainer

import torch

flair_tag_type = 'ner'

class TARSModel:
    def __init__(self, corpus, *args, **kwargs):
        self.dictionary = Dictionary(add_unk=False)
        self.dictionary.add_item('term')
        self.tagger = TARSTagger.load('tars-ner')
        self.corpus = Corpus(*corpus)
        self.trainer = ModelTrainer(self.tagger, self.corpus)

    def train(self):
        self.tagger.add_and_switch_to_new_task(task_name='Term tagging', label_dictionary=self.dictionary, label_type=flair_tag_type)
        self.trainer.train('output', optimizer=torch.optim.AdamW, learning_rate=5e-6, max_epochs=3, embeddings_storage_mode="none", weight_decay=0)

    def predict(self):
        for sent in self.corpus.test:
            tars.predict(sent)

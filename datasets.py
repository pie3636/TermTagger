from flair.data import FlairDataset, Sentence
from models.tarstr import flair_tag_type
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

import numpy as np
import spacy

udp_pos = ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ', 'X', 'NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ'] # https://universaldependencies.org/u/pos/

tag_names = ['B', 'I', 'O']
tag2idx = {tag: i for i, tag in enumerate(tag_names)}


nlp = spacy.load("en_core_web_sm")

def pad(data, length, value):
    if len(data) < length:
        data += [value] * (length - len(data))
    return data


class ClassmateError(Exception):
    def __init__(self, message, line_content, file, line):
        year = file.parts[2]
        super().__init__(f'In {file.parts[-1]:25} at {line:4} [{repr(line_content) + "]":25}: ' + message)


class SentenceDataset(Dataset):
    def __init__(self, data_dir, features=None):
        if features is None:
            features = []
        self.data = []
        self.feature_list = features
        if 'POS' in features:
            self.pos_enc = OneHotEncoder(sparse=False).fit([[pos] for pos in udp_pos])
        for file in Path(data_dir).rglob('*final'):
            lines = open(file, encoding='utf-8').read().splitlines()
            tokens = []
            tags = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    split = line.strip().split()
                    token, *_, tag = split

                    if tag not in 'BIO':
                        raise ClassmateError(f'Label is [{repr(tag)}]', line, file, i)
                    if not isinstance(token, str):
                        raise ClassmateError(f'Token [{repr(token)}] is a {type(token)}', line, file, i)
                    if not isinstance(tag, str):
                        raise ClassmateError(f'Tag [repr({tag})] is a {type(tag)}', line, file, i)
                    if not tokens:
                        token = token.lower() # TODO: Improve (only lowercase non proper nouns)
                    if token == "'s":
                        tokens.append("'")
                        tokens.append('s')
                        tags.append(tag2idx[tag])
                        tags.append(tag2idx[tag])
                    elif token.endswith('-') and token != '-':
                        tokens.append(token[:-1])
                        tokens.append('-')
                        tags.append(tag2idx[tag])
                        tags.append(tag2idx[tag])
                    else:
                        tokens.append(token)
                        tags.append(tag2idx[tag])
                elif tokens:
                    data_to_add = [tokens, tags]
                    doc = nlp(' '.join(tokens))
                    if 'POS' in self.feature_list:
                        pos = []
                        for token in doc:
                            pos.append(self.pos_enc.transform([[token.pos_]])[0])
                        data_to_add = data_to_add[:-1] + [pos] + [data_to_add[-1]]
                    if 'WL' in self.feature_list:
                        lengths = []
                        for token in doc:
                            lengths.append(np.array([len(token.text)]))
                        data_to_add = data_to_add[:-1] + [lengths] + [data_to_add[-1]]
                    self.data.append(data_to_add)
                    tokens = []
                    tags = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WordDataset(SentenceDataset):
    def __init__(self, data_dir, uncap_first=False, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.inputs = []
        self.outputs = []
        new_data = []
        for tokens, tags in self.data:
            for token, tag in zip(tokens, tags):
                new_data.append((token, tag))
                self.inputs.append(token)
                self.outputs.append(tag)
        self.data = new_data


class SlidingWindowDataset(SentenceDataset):
    def __init__(self, data_dir, window_size=3, uncap_first=False, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.window_size = window_size
        self.uncap_first = uncap_first
        new_data = []
        for tokens, *features, tags in self.data:
            if len(tokens) < window_size: # If the sentence doesn't have enough words, fill it with dots
                tokens = pad(tokens, window_size, '.')
                feature_idx = 0
                if 'POS' in self.feature_list:
                    features[feature_idx] = np.array(pad(features[feature_idx], 3, np.zeros(len(udp_pos))))
                    feature_idx += 1
                if 'WL' in self.feature_list:
                    features[feature_idx] = np.array(pad(features[feature_idx], 3, np.zeros(1)))
                    feature_idx += 1
                tags = pad(tags, window_size, tag2idx['O'])
                new_data.append((tokens, *features, tags))
            else:
                for i in range(0, len(tokens) - window_size + 1):
                    new_data.append((
                        tokens[i:i+window_size],
                        *[feature[i:i+window_size] for feature in features],
                        tags[i:i+window_size // 2]
                        ))
        self.data = new_data


def create_fsentence(_input, verbose=False) -> Sentence:
    tokens, *features, tags = _input
    sentence = Sentence(" ".join(tokens))
    if len(sentence) != len(tokens):
        print("Trouble: ", sentence)
        return sentence
    for i, token in enumerate(tokens):
        if tags[i] == 2:
            continue
        if tags[i] == 0:
            onset = i
            offset = i+1
            for tag in tags[i+1:]:
                if tag == 1:
                    offset += 1
                else:
                    break
            if verbose: print(onset, offset)
            sentence[onset:offset].add_label(flair_tag_type, value="term")
    return sentence


class TARSDataset(FlairDataset):
    def __init__(self, data_dir):
        self.data = [create_fsentence(sent) for sent in SentenceDataset(data_dir)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

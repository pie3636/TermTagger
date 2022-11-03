from torch.utils.data import Dataset
from pathlib import Path

tag_names = ['B', 'I', 'O']
tag2idx = {tag: i for i, tag in enumerate(tag_names)}


def pad(data, length, value):
    if len(data) < length:
        data += [value] * (length - len(data))
    return data


class ClassmateError(Exception):
    def __init__(self, message, line_content, file, line):
        year = file.parts[2]
        super().__init__(f'In {file.parts[-1]:25} at {line:4} [{repr(line_content) + "]":25}: ' + message)


class SentenceDataset(Dataset):
    def __init__(self, data_dir, uncap_first=False):
        self.data = []
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
                    tokens.append(token)
                    tags.append(tag2idx[tag])
                else:
                    self.data.append((tokens, tags))
                    tokens = []
                    tags = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WordDataset(SentenceDataset):
    def __init__(self, data_dir, uncap_first=False):
        super().__init__(data_dir, uncap_first)
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
    def __init__(self, data_dir, window_size=3, uncap_first=False):
        super().__init__(data_dir, uncap_first)
        self.window_size = window_size
        self.uncap_first = uncap_first
        new_data = []
        for tokens, tags in self.data:
            if len(tokens) < window_size: # If the sentence doesn't have enough words, fill it with dots
                new_data.append((pad(tokens, window_size, '.'), pad(tags, window_size, tag2idx['O'])))
            else:
                for i in range(0, len(tokens) - window_size + 1):
                    new_data.append((tokens[i:i+window_size], tags[i:i+window_size // 2]))
        self.data = new_data

from torch.utils.data import Dataset
from pathlib import Path


class ClassmateError(Exception):
    def __init__(self, message, line_content, file, line):
        year = file.parts[2]
        super().__init__(f'In {file.parts[-1]:25} at {line:4} [{repr(line_content) + "]":25}: ' + message)


class SentenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for file in Path(data_dir).rglob('*final'):
            lines = open(file, encoding='utf-8').read().splitlines()
            tokens = []
            labels = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    split = line.strip().split()
                    token, *_, label = split

                    if label not in 'BIO':
                        raise ClassmateError(f'Label is [{repr(label)}]', line, file, i)
                    if not isinstance(token, str):
                        raise ClassmateError(f'Token [{repr(token)}] is a {type(token)}', line, file, i)
                    if not isinstance(label, str):
                        raise ClassmateError(f'Label [repr({label})] is a {type(label)}', line, file, i)
                    tokens.append(token)
                    labels.append(label)
                else:
                    self.data.append((tokens, labels))
                    tokens = []
                    labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_set = SentenceDataset('data/train')
dev_set = SentenceDataset('data/dev')
test_set = SentenceDataset('data/test')


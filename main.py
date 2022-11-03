from torch.utils.data import Dataset
from pathlib import Path


groups = ['Nora & Soklong', 'Duy & Maxime', 'Joe and PA', 'Karolin and Omar', 'Mathilde and Dimitra', 'Silviya and Averie', 'Adriana and Jimmy', 'Hossain and Andres', 'Madi and Kevin', 'Jorge and Dalila', 'Shahzaib, Roham and Scott', 'Maeva and Louis', 'Abir and Camille']


class ClassmateError(Exception):
    def __init__(self, message, line_content, file, line):
        year = file.parts[2]
        authors = groups[2021 - int(year)]
        reviewers = groups[int(year) - 2009]
        super().__init__(f'In {file.parts[-1]:25} at {line:4} [{repr(line_content) + "]":25} (blame {authors:25} OR {reviewers + ")":25}: ' + message)


class SentenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.errors = []
        for file in Path(data_dir).rglob('*final'):
            lines = open(file, encoding='utf-8').read().splitlines()
            tokens = []
            labels = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    tabs = line.count('\t')
                    if tabs != 1:
                        self.errors.append(ClassmateError(f'Encountered [{repr(line)}] with {tabs} tabulations', line, file, i))
                        continue
                    split = line.strip().split()
                    try:
                        token, *_, label = split
                    except:
                        print(file, i)
                        raise


                    if label not in 'BIO':
                        self.errors.append(ClassmateError(f'Label is [{repr(label)}]', line, file, i))
                    if not isinstance(token, str):
                        self.errors.append(ClassmateError(f'Token [{repr(token)}] is a {type(token)}', line, file, i))
                    if not isinstance(label, str):
                        self.errors.append(ClassmateError(f'Label [repr({label})] is a {type(label)}', line, file, i))
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
for error in train_set.errors:
    print(repr(error))
print(len(train_set.errors), sum(len(x[0]) for x in train_set.data))

dev_set = SentenceDataset('data/dev')
for error in dev_set.errors:
    print(repr(error))
print(len(dev_set.errors), sum(len(x[0]) for x in dev_set.data))

test_set = SentenceDataset('data/test')
for error in test_set.errors:
    print(repr(error))
print(len(test_set.errors), sum(len(x[0]) for x in test_set.data))

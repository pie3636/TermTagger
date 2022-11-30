from flair.models import TARSTagger
from flair.data import Sentence

tag_type = "ner"

def create_fsentence(_input) -> Sentence:
    tokens, tags = _input
    sentence = Sentence(" ".join(tokens))
    print(sentence)
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
            print(onset, offset)
            sentence[onset:offset].add_label(tag_type, value="term")
    return sentence

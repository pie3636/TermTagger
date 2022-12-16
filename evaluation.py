from flair.data import Sentence
from models.tarstr import flair_tag_type


def convert_span_into_iob(_in:Sentence) -> list:
    tokens = [tok for tok in _in]
    spans = _in.get_spans(flair_tag_type)
    iob = ["O"]*len(tokens)
    for span in spans:
        for i, token in enumerate(span):
            if i == 0:
                iob[token.idx] = "B"
            else:
                iob[token.idx] = "I"
    return iob


def get_spans_from_iobs(tags:list, verbose=False):
    spans = []
    for i, tag in enumerate(tags):
        if tag == 2:
            continue
        if tag == 0:
            onset = i
            offset = i+1
            for t in tags[i+1:]:
                if t == 1:
                    offset += 1
                else:
                    break
            if verbose: print(onset, offset)
            spans.append([onset,offset])
    return spans
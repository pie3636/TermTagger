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
            spans.append((onset,offset))
    return spans


def get_span_based_scores(hyps, refs):
    h_spans = set(get_spans_from_iobs(hyps))
    r_spans = set(get_spans_from_iobs(refs))
    intersec = h_spans.intersection(r_spans)
    p = len(intersec)/len(h_spans)
    r = len(intersec)/len(r_spans)
    f1 = (2*p*r)/(p+r)
    return p, r, f1





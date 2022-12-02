from flair.data import Sentence

flair_tag_type = "ner"

def convert_span_into_iob(_in:Sentence) -> list:
    tokens = [tok for tok in _in]
    spans = _in.get_spans()
    iob = ["O"]*len(tokens)
    for span in spans:
        for i, token in enumerate(span):
            if i == 0:
                iob[token.idx] = "B"
            else:
                iob[token.idx] = "I"
    return iob


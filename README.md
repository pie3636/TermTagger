# TermTagger

## Usage instructions

- Install the required packages by using `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.
- Download FastText embeddings at http://vectors.nlpl.eu/repository/20/10.zip and convert them into two files (`embeds/english_fasttext_2017_10.vectors.npy`) and (`embeds/english_fasttext_2017_10`) using the procedure described in https://stackoverflow.com/a/43067907.
- Run `main.py` to start the training and execution process.

**Warning:** TARS requires a lot of VRAM for both training and inference. We recommend using Grid5k or any other similar resource.

## Results

The results of running the fine-tuned TARS tagger on the test set can be found in `output/tars_results.txt`.

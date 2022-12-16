# TermTagger

## Usage instructions

- Install the required packages by using `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`.
- Download the required [additional files](https://drive.google.com/drive/folders/1UV_nYGVGYtCEvvubijR6IZAdsp2ztKD9?usp=sharing) and move them into the folders `embeds/` (for `english_fasttext_2017_10.vectors.npy` and `english_fasttext_2017_10`) and `output/` (for `final-model.pt`).
- The full training and inference process can be triggered by running `main.py`.
- Alternatively, `sandbox.py` can be run as a standalone program to play with TARS' capabilities.

**Warning:** TARS requires a lot of VRAM for both training and inference. We recommend using Grid5k or any other similar resource.

## Results

The results of running the fine-tuned TARS tagger on the test set can be found in `output/tars_results.txt`.

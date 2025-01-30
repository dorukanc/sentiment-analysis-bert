Sentiment Analysis with DistilBERT

This project fine-tunes DistilBERT for sentiment analysis on the IMDb dataset, predicting whether a movie review is positive or negative. It includes a Gradio interface for easy real-time testing.

The model is available on Hugging Face: [DistilBERT IMDb Model](https://huggingface.co/dorukan/distil-bert-imdb)

Project Structure:
------------------
dorukanc-sentiment-analysis-bert/
    ├── LICENSE
    ├── gradio-interface.py
    ├── train.py
    └── dataset/
        └── IMDB.csv

- gradio-interface.py: Gradio interface for testing the model.
- train.py: Script for fine-tuning the model.
- dataset/IMDB.csv: IMDb dataset for training.

Usage:
------
Train the model:

python train.py

Run the Gradio interface:

python gradio-interface.py

License:
--------
This project is licensed under the MIT License. See the LICENSE file for details.


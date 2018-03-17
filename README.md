# twitter-sentiment
Scripts for training a sentiment analysis model and serving it.

## Installation
1) Clone the repo:
```
git clone https://github.com/social-barometer/twitter-sentiment.git
```
2) Create a virtual env: 
```
python3 -m venv twitter-emotion
```
3) cd to the folder:
```
cd twitter-emotion
```
4) Activate teh virtual env:
```
source bin/activate
```
5) Install requirements:
```
pip3 install requirements.txt
```
6) Run the server:
```
pip3 server.py
```

## Training
A trained model is already included in the _models_ folder but you can tweak the _train.py_ script to make a better one!

To do so you must download the _data_.

### Training data
Download the _Twitter Emotion Corpus_ from: http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html and save unzip it to root of the project.

### Word Embeddings
The training script also needs GloVe embeddings to work. Download the _glove.twitter.27B.zip embeddings_ from https://nlp.stanford.edu/projects/glove/ and unzip it to the root of the project.

#### All done!
Now you are ready to train your own models. Have fun!

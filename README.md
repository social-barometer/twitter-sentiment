# twitter-sentiment
Scripts for training a sentiment analysis model and serving it.

## Demo
Test it out here: https://twitter-emotion.herokuapp.com/

## API docs
Query https://twitter-emotion.herokuapp.com/emotion-analysis
with a _POST_ with a body like this:
```json
{
	"tweets": [
		"Tweet",
		"Another tweet",
		"Are we done with the tweets already?"
		]
}
```
Remeber to use _"content-type": "application/json"_ in the header!

And it will return something like this:
```json
[
    {"anger": 6.688, "disgust": 3.302, "fear": 16.631, "joy": 21.477, "sadness": 22.82, "surprise": 19.229},
    {"anger": 9.33, "disgust": 6.366, "fear": 15.042, "joy": 22.213, "sadness": 25.334, "surprise": 22.911},
    {"anger": 13.476, "disgust": 13.434, "fear": 8.879, "joy": 13.032, "sadness": 28.745, "surprise": 22.029}
]
```

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
Now you are ready to train your own models. Once you're done with tweaking the _train.py_ script run the following command to train the model:
```
pip3 train.py
```

Have fun!

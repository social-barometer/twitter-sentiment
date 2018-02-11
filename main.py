import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data.preprocess import clean
from classifier.train import train_svm_model, train_rf_model, train_nb_model

df = clean('./training_data/Jan9-2012-tweets-clean.txt')

X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'],
    df['emotion'],
    test_size=0.2
)

clf = train_svm_model(X_train, y_train)

predicted = clf.predict(X_test)
accuracy = np.mean(predicted == y_test)
print(accuracy)
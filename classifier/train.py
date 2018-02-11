from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def make_pipeline(classifier):
  """Makes a nice pipeline ready to be fit."""
  return Pipeline([('vect', CountVectorizer(stop_words='english')),
                ('tfidf', TfidfTransformer()),
                ('clf', classifier)])

def train_svm_model(X, y):
  """Trains a SVM model for emotion detection."""

  clf = make_pipeline(SGDClassifier(loss='hinge',
                                    penalty='l2',
                                    alpha=1e-3,
                                    random_state=42,
                                    max_iter=5,
                                    tol=None))
  clf.fit(X, y)
  return clf

def train_rf_model(X, y):
  """Trains a random forest model form emotion detection."""
  clf = make_pipeline(RandomForestClassifier())
  clf.fit(X, y)
  return clf

def train_nb_model(X, y):
  """Trains a Naive Bayes model."""
  clf = make_pipeline(MultinomialNB())
  clf.fit(X, y)
  return clf
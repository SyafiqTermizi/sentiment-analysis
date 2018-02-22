from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
import numpy as np

x, y = [], []

print('Reading Training data')

counter = 0
with open('./train_dataset_sentiment.txt') as f:
    for i in f:
        y.append(i)
        counter += 1
        if counter >= 50000:
            break
f.close()

y_int = np.array(y, dtype=int)

counter = 0
with open('./train_dataset_text.txt') as f:
    for i in f:
        x.append(i)
        counter += 1
        if counter >= 50000:
            break
f.close()

print('Vectorizing words')

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(x)
features_array =  features.toarray()

print('Training classifier')

clf = MultinomialNB()
trained_clf = clf.fit(features_array, y_int)
# joblib.dump(trained_clf, 'tweet_sental.pkl') 

print('Classifier trained and ready to use')

# text = ['i hate you so much']
# text_features = cv.fit_transform(text)
# text_array = text_features.toarray()

# print(trained_clf.predict([text_array]))
# print(features_array[9:10])

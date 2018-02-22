from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


while True:
    x = []
    counter = 0
    with open('./train_dataset_text.txt') as f:
        for i in f:
            x.append(i)
            counter += 1
            if counter >= 50000:
                break
    f.close()

    x.append(input('Enter a text: '))

    cv = CountVectorizer(stop_words='english')
    features = cv.fit_transform(x)
    x =  features.toarray()

    clf = joblib.load('sentalNB.pkl') 
    print(clf.predict(x[len(x)-1:]))
    print(len(x))

# 1. baca original text
# 2. append input ke text
# 3. vectorize
# 4. classify

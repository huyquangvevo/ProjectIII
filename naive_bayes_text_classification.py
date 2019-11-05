from sklearn.feature_extraction.text import CountVectorizer

corpus = [ 'This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
X = X.toarray()
# print(X.toarray())
print(len(X))
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('fake_or_real_news.csv')

dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

dataset.head()

title = np.array(dataset["title"])
label = np.array(dataset["label"])

cv = CountVectorizer()
title = cv.fit_transform(title)

titletrain, titletest, labeltrain, labeltest = train_test_split(title, label, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(titletrain, labeltrain)

headline = "Former Russian colonel criticizes invasion of Ukraine on state TV"

data = cv.transform([headline]).toarray()
print(model.predict(data))
from django.shortcuts import render

from django.http import HttpResponse
from .forms import SearchForm
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def Home(request):
    form = SearchForm(request.POST or None)
    response = None
    if form.is_valid():
        value = form.cleaned_data.get("q")

        df = pd.read_csv('spam.csv', encoding="latin-1")
        df['label'] = df['Label'].map({'ham': 0, 'spam': 1})
        X = df['EmailText']
        y = df['label']
        cv = CountVectorizer()
        X = cv.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = MultinomialNB()
        clf.fit(X_train,y_train)
        print(clf.score(X_test,y_test))
        y_pred = clf.predict(X_test)
        message = value
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        if(my_prediction== 1):
            print("Spam")
            response = "Spam"
        else:
            print("Ham")
            response = "Ham"

        return render(request, 'result.html', {"response": response})
    return render(request, 'form.html', {"form": form})




# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB, GaussianNB
# from sklearn import svm
# from sklearn.model_selection import GridSearchCV
#
# ##Step1: Load Dataset
# dataframe = pd.read_csv("spam.csv")
# print(dataframe.describe())
#
# ##Step2: Split in to Training and Test Data
#
# x = dataframe["EmailText"]
# y = dataframe["Label"]
#
# x_train,y_train = x[0:4457],y[0:4457]
# x_test,y_test = x[4457:],y[4457:]
#
# ##Step3: Extract Features
# cv = CountVectorizer()
# features = cv.fit_transform(x_train)
#
# ##Step4: Build a model
# tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]}
#
# model = GridSearchCV(svm.SVC(), tuned_parameters)
#
# model.fit(features,y_train)
#
# print(model.best_params_)
# #Step5: Test Accuracy
# print(model.score(cv.transform(x_test),y_test))

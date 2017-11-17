#!/usr/bin/python

import sys
import time
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### Load the Dictionary Containing the Dataset

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# poi should be first    
#features_list = ['poi'] + [i for i in all_features if i not in ['poi', 'email_address', 'name']]
features_list = ['poi', 'fraction_from_poi_email', 'fraction_to_poi_email', 'total_payments', 
                 'total_stock_value', 'salary', 'bonus', 'director_fees']
### Task 2: Remove outliers

# remove total:
del data_dict['TOTAL']
import pandas as pd
import numpy as np

pl = []
for k, v in data_dict.items():
    l = v
    l.update({'name': k})
    pl.append(l)
df = pd.DataFrame(pl)

# look into data:
print(df.describe().T)

### remove NAN's from dataset, show top 5 salary
top_salary = df[df['salary'] != 'NaN'].sort_values('salary', ascending=False)['name'][:3]

for name in top_salary:
    del data_dict[name]

### plot features
data = featureFormat(data_dict, features_list)

"""
import seaborn
for point in data:
    plt.scatter( point[0], point[1] )
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
"""

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


### new features: fraction_to_poi_email,fraction_from_poi_email
df.loc[df['from_poi_to_this_person'] == 'NaN', 'from_poi_to_this_person'] = 0.0
df.loc[df['from_this_person_to_poi'] == 'NaN', 'from_this_person_to_poi'] = 0.0
df.loc[df['to_messages'] == 'NaN', 'to_messages'] = -1
df.loc[df['from_messages'] == 'NaN', 'from_messages'] = -1

df['fraction_from_poi_email'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi_email'] = df['from_this_person_to_poi'] / df['from_messages']

name_to_fraction_from_poi_email = zip(df['name'], df['fraction_from_poi_email'])
name_to_fraction_to_poi_email = zip(df['name'], df['fraction_to_poi_email'])

for k, v in dict(name_to_fraction_from_poi_email).items():
    if k not in data_dict:
        continue
    data_dict[k]['fraction_from_poi_email'] = v
for k, v in dict(name_to_fraction_to_poi_email).items():
    if k not in data_dict:
        continue
    data_dict[k]['fraction_to_poi_email'] = v

my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                             test_size=0.1,
                                                                                             random_state=42)


from sklearn.tree import DecisionTreeClassifier
import time

t0 = time.time()

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print('accuracy before tuning ', score)

print('consume time(Decision tree):', round(time.time() - t0, 3), 's')

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

### try Naive Bayes for prediction
t0 = time.time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print(accuracy)


# bernoulliNB
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print(accuracy)



# print("consume time (naive bayes):", round(time.time() - t0, 3), "s")
from sklearn import grid_search

# too slow 
#svc_clf = SVC()
#parameters = {'kernel': ['rbf', 'linear', 'sigmoid']}

#clf = grid_search.GridSearchCV(svc_clf, parameters)
#clf = clf.fit(features_train, labels_train)
#clf = clf.best_estimator_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# disable jupyer notebook warning
#import warnings
#warnings.filterwarnings('ignore')


tuned_parameters = {'alpha': [0.5, 0.7, 1.0, 0], 'fit_prior': [True, False],
                     'binarize': [0, 0.2, 0.5, 0.8, 1.0]}

scores = ['precision', 'recall']
from sklearn.model_selection import GridSearchCV


for score in scores:
    print("# 基于 %s 调参" % score)

    bnb  = BernoulliNB()

    clf = GridSearchCV(bnb, tuned_parameters,
                       scoring='%s_macro' % score)

    clf.fit(features_train, labels_train)

    print("最佳参数组合:")
    print()

    print(clf.best_params_)

    print()
    print("网格排名:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']


    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred,labels_test)

    print("accuracy: ", accuracy)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


# 最终参数组合
clf=  BernoulliNB(**{'binarize': 0.2, 'fit_prior': True, 'alpha': 0.5})
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print(time.time() - t0)
print(accuracy)

dump_classifier_and_data(clf, my_dataset, features_list)
